import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyperparameters
batch_size = 64
learning_rate = 1e-2


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


data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_set = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = datasets.MNIST(root='./data', train=False, transform=data_tf)
test_data_loader = DataLoader(test_set, batch_size=batch_size)

my_net = LeNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on " + str(device))
my_net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_net.parameters(), lr=learning_rate)

for epoch in range(50):
    print("Training...epoch %d" % epoch)
    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = my_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print("[%d, %d] loss:%.3f" % (epoch, i+1, running_loss / 200))
            running_loss = 0
print('Finish training the network')

correct = 0
total = 0
correct_num = [0]*10
total_num = [0]*10

with torch.no_grad():
    for data in test_data_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = my_net(inputs)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_num[label] += 1
                correct += 1
            total_num[label] += 1
            total += 1

print("Total Accuracy: %.3f%%" % (100 * correct / total))
for num in range(10):
    print("The Accuracy of number %d is: %.3f%%" % (num, (100 * correct_num[num] / total_num[num])))
