import json

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from logger import Logger
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(

            nn.Linear(16 * 11 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 19),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 11 * 4)
        x = self.classifier(x)
        return x


def test_MNIST():
    confusion = np.zeros((10, 10), dtype=np.uint32)  # First index actual, second index predicted
    correct = 0
    n = 0
    N = len(test_dataset)
    for idx,(d, l) in enumerate(test_dataset):
        #if idx == 1:
            #print(d.shape)
            #print(d.unsqueeze(0).shape)
        d = Variable(d.unsqueeze(0))
        outputs = net.forward(d)
        _, out = torch.max(outputs.data, 1)
        c = int(out.squeeze())
        confusion[l, c] += 1
        if c == l:
            correct += 1
        n += 1
    acc = correct / n
    #print(confusion)
    F1 = 0
    for nr in range(10):
        TP = confusion[nr, nr]
        FP = sum(confusion[:, nr]) - TP
        FN = sum(confusion[nr, :]) - TP
        F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N
    print('F1: ', F1)
    print('Accuracy: ', acc)
    return [('F1',F1), ('Accuracy', acc)]


class MNIST_Addition(Dataset):

    def __init__(self, dataset, examples):
        self.data = list()
        self.dataset = dataset
        with open(examples) as f:
            for line in f:
                line = line.strip().split(' ')
                self.data.append(tuple([int(i) for i in line]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        i1, i2, l = self.data[index]
        return torch.cat((self.dataset[i1][0], self.dataset[i2][0]), 1), l


if __name__=='__main__':

    net = Net()
    net.load_state_dict(torch.load('sd_nn_cnn.pt'))
    net.classifier[4] = nn.Linear(84,10)
    #net.classifier[5] = nn.LogSoftmax(1)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    criterion = nn.NLLLoss()



    train_dataset = MNIST_Addition(
        torchvision.datasets.MNIST(root='../../../data/MNIST', train=True, download=True, transform=transform),
        'train_data.txt')
    test_dataset = MNIST_Addition(
        torchvision.datasets.MNIST(root='../../../data/MNIST', train=False, download=True, transform=transform),
        'test_data.txt')
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)

    i = 1
    test_period = 500
    log_period = 50
    running_loss = 0.0
    log = Logger()

    for epoch in range(1):

        for data in trainloader:
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()#data[0]
            if i % log_period == 0:
                print('Iteration: ', i * 2, '\tAverage Loss: ', running_loss / log_period)
                log.log('loss', i * 2, running_loss / log_period)
                running_loss = 0
            if i % test_period == 0:
                log.log_list(i * 2, test_MNIST())
            i += 1

    with open('tl_add2abs_nn.json', 'w') as outfile:
        json.dump(log.log_data, outfile)