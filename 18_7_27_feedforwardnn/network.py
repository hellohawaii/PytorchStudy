
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyper parameters
batch_size = 64
learning_rate = 1e-2


class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, out_dim):
        super(SimpleNet, self).__init__()
        # have to write 'SimpleNet)
        self.my_layer1 = nn.Linear(input_dim, hidden1_dim)
        self.my_layer2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.my_layer3 = nn.Linear(hidden2_dim, out_dim)

    def forward(self, x):
        x = x.reshape([-1, 784])
        # 黄熠华加了这个改尺寸
        x = self.my_layer1(x)
        # x = nn.ReLU(True)
        x = self.my_layer2(x)
        # x = nn.ReLU(True)
        x = self.my_layer3(x)
        return x


data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_data_set = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
test_data_set = datasets.MNIST(root='./data', train=False, transform=data_tf)
train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data_set, batch_size=batch_size)

net_model = SimpleNet(28 * 28, 300, 100, 10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net_model.parameters(), lr=learning_rate)

for epoch in range(1):
    for data in train_loader:
        # 这里取到的数据可能batch_size是64，一次取的data有64张图片
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print(loss)
    print(epoch)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net_model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network is %d %%' % (100 * correct / total))
