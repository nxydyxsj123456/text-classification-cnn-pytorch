# This is a sample Python script.
import torch

from cnn_model import TextCNN_pytorch, TCNNConfig


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    config=TCNNConfig()
    model=TextCNN_pytorch(config)


    input = torch.randint(500,(64, 500))
    # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len

    model.forward(input)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
