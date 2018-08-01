from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # 输入：1通道，28*28图片
        layer1 = nn.Sequential()
        layer1.add_module('Conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=1, padding=2))
        # 输出：6通道，28*28图片
        layer1.add_module('MaxPool1', nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer1 = layer1
        # 输出：6通道，14*14图片

        layer2 = nn.Sequential()
        layer2.add_module('Conv2', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5))
        # 输出：16通道，10*10图片
        layer2.add_module('MaxPool2', nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = layer2
        # 输出：16通道，5*5图片

        layer3 = nn.Sequential()
        layer3.add_module('FC1', nn.Linear(16 * 5 * 5, 200))
        layer3.add_module('FC2', nn.Linear(200, 84))
        layer3.add_module('FC3', nn.Linear(84, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x
