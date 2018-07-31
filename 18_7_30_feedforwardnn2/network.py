import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 自己尝试写一下最简单的全相连前馈神经网络

# Hyper Parameters
batch_size = 64
learning_rate = 1e-2


class MyNet(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim):
        super(MyNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_dim1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU(True))
        self.layer3 = nn.Linear(hidden_dim2, out_dim)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


my_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_data = datasets.MNIST(root='./data', train=True, transform=my_tf, download=True)
train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.MNIST(root='./data', train=False, transform=my_tf)
test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size)

network = MyNet(28 * 28, 300, 100, 10)
optimizer = optim.SGD(network.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()  # 这个括号好像必须加，不然报错

for epoch in range(3):
    print("Training......  epoch:{}".format(epoch))
    for data in train_data_loader:
        inputs, labels = data
        outputs = network(inputs)
        _, predicted = torch.max(outputs, 1)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
print("Training finished")

correct = 0
total = 0
correct_num = [0]*10  # 初始化的方法学习一下
total_num = [0]*10

with torch.no_grad():
    for data in test_data_loader:
        inputs, labels = data
        outputs = network(inputs)
        _, predicted = torch.max(outputs, 1)
        for (label, pred) in zip(labels, predicted):
            if label == pred:
                correct_num[label] += 1
                correct += 1
            total_num[label] += 1
            total += 1

print("Total Accuracy: %d%%" % (100 * correct / total))
for num in range(10):
    print("The Accuracy of number %d is: %d%%" % (num, (100 * correct_num[num] / total_num[num])))
