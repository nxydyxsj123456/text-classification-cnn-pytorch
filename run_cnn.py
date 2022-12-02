#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy
import numpy as np
import sklearn
import torch.nn
from torch.utils.data import Dataset, DataLoader



# print(keras.__version__)

from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

# base_dir = 'data/cnews'
# train_dir = os.path.join(base_dir, 'cnews.train.txt')
# test_dir = os.path.join(base_dir, 'cnews.test.txt')
# val_dir = os.path.join(base_dir, 'cnews.val.txt')
# vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')


base_dir = 'data/CSIC2010'
train_dir = os.path.join(base_dir, 'train.txt')
test_dir = os.path.join(base_dir, 'test.txt')
val_dir = os.path.join(base_dir, 'val.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')



save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict
class MyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


def evaluate(model,loss_func, x_, y_):
    """评估在某一数据上的准确率和损失"""

    model.eval()
    data_len = len(x_)
    dataset = MyDataset(x_, y_)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size)
    #TODO
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for  x_batch, y_batch in dataloader:
            lable_idx = torch.max(y_batch, dim=1)[1]
            outputs =model(x_batch)
            loss =loss_func(outputs,lable_idx)
            total_loss += loss
            preds=torch.max(outputs,dim=1)[1]
            total_acc += torch.sum(preds==lable_idx).item()

    return total_loss / data_len, total_acc / data_len


def train():

    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    # tf.summary.scalar("loss", model.loss)
    # tf.summary.scalar("accuracy", model.acc)
    # merged_summary = tf.summary.merge_all()
    # writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    # saver = tf.train.Saver()
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    #x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_train= numpy.load("x_train.npy")
    y_train= numpy.load("y_train.npy")


    #x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    x_val= numpy.load("x_val.npy")
    y_val= numpy.load("y_val.npy")


    time_dif = get_time_dif(start_time)
    print("数据集处理时间Time usage:", time_dif)

    # 创建session
    # session = tf.Session()
    # session.run(tf.global_variables_initializer())
    # writer.add_graph(session.graph)
    #

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=config.dropout_keep_prob)

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        dataset = MyDataset(x_train, y_train)
        dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size)
        for x_batch, y_batch in dataloader:
            #feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            #if total_batch % config.save_per_batch == 0:
                #print(1)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                loss_train, acc_train = evaluate(model, loss_func, x_batch, y_batch)
                loss_val, acc_val = evaluate(model, loss_func,x_val, y_val)
                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    #saver.save(sess=session, save_path=save_path)
                    torch.save(model, save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            model.train()
            outputs = model(x_batch)
            lable_idx = torch.max(y_batch, dim=1)[1]
            loss = loss_func(outputs, lable_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")

    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))
    print("AUC...")
    print(sklearn.metrics.roc_auc_score(y_test_cls, y_pred_cls))





    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test_cls, y_pred_cls)
    ttt=0
    for i,rate in enumerate(tpr):
        if rate > 0.05:
            ttt=thresholds[i]
            break
    print(ttt)


    roc_auc = auc(fpr, tpr)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.axvline(0.05,c='r',ls='-.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()



    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def loadData(file):
    with open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if (len(d) > 0):
            result.append(d)
    return result



if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")

    print('Configuring CNN model...')
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()  #获得名称到id的映射
    words, word_to_id = read_vocab(vocab_dir)#获得词汇到id的映射


    config.vocab_size = len(words)
    config.num_classes=len(categories)
    model = TextCNN(config)

    if sys.argv[1] == 'train':
        train()
    else:
        test()
