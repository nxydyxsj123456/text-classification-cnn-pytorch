# coding: utf-8
import torch
import torch.nn as nn
class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 500  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

class TextCNN(nn.Module):
    def __init__(self,config):
        super(TextCNN, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)

        self.conv1 = nn.Conv1d(in_channels=self.config.embedding_dim,out_channels = self.config.num_filters, kernel_size = self.config.kernel_size)

        self.fc1 = nn.Linear(self.config.num_filters, self.config.hidden_dim)
        self.dropout = nn.Dropout(p=self.config.dropout_keep_prob)
        self.relu =  nn.ReLU()
        self.fc2 = nn.Linear(self.config.hidden_dim, self.config.num_classes)



    def forward(self, x):
        x = self.embedding(x)
        x = torch.transpose(x, 1, 2)

        x = self.conv1(x)
        x = torch.max(x, 2)[0]  #全局池化

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)

        # 分类器
        x = self.fc2(x)
        y_pred_class=torch.argmax(x,dim=1)



        return x




