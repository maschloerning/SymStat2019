import torch
import torchvision
import torchvision.transforms as transforms



transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='../../../data/MNIST', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='../../../data/MNIST', train=False, download=True, transform=transform)

#59719,45022 7
print(train_dataset[59719][1])
print(train_dataset[45022][1])

"""
with open('train_data_abs.txt', "r") as f:
    train = f.read()

train = list(train.split('\n'))
train = [t.split(' ') for t in train if t!='']


with open('train_data_abs_dpl.txt', "w") as f:
    for line in train:
        f.write("abs({},{},{}).\n".format(int(line[0]),int(line[1]),int(line[2])))


with open('test_data_abs.txt', "r") as f:
    test = f.read()

test = list(test.split('\n'))
test = [t.split(' ') for t in test if t!='']

with open('test_data_abs_dpl.txt', "w") as f:
    for line in test:
        f.write("abs({},{},{}).\n".format(int(line[0]),int(line[1]),int(line[2])))
"""
"""

# creating txt-file containing data and labels for absolute values (training data)
with open('train_data.txt') as f:
    train = f.read()

train = list(train.split('\n'))
train = [t.split(' ') for t in train if t != ''] #indices
train[-1]

with open('train_data_abs.txt',"w") as f:
    train[-1]
    for id1,id2, _ in train:
        val1 = train_dataset[int(id1)][1]
        val2 = train_dataset[int(id2)][1]
        f.write("{} {} {}\n".format(id1,id2,abs(val1-val2)))

# for test_data now
with open('test_data.txt') as f:
    test = f.read()

test = list(test.split('\n'))
test = [t.split(' ') for t in test if t!=''] #indices

with open('test_data_abs.txt',"w") as f:
    for id1,id2, _ in test:
        val1 = test_dataset[int(id1)][1]
        val2 = test_dataset[int(id2)][1]
        f.write("{} {} {}\n".format(id1,id2,abs(val1-val2)))
"""