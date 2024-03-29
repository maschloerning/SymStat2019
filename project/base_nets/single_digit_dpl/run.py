from train import train_model
from data_loader import load
from examples.NIPS.MNIST.mnist import test_MNIST, MNIST_Net, neural_predicate
from model import Model
from optimizer import Optimizer
from network import Network
import torch


queries = load('train_data.txt')

with open('addition.pl') as f:
    problog_string = f.read()


network = MNIST_Net()
net = Network(network, 'mnist_net', neural_predicate)
net.optimizer = torch.optim.Adam(network.parameters(),lr = 0.001)
model = Model(problog_string, [net], caching=False)
optimizer = Optimizer(model, 2)


# for transfer learning i need to be able to save the mnist_net model. where can I find it?


train_model(model,queries, 1, optimizer,test_iter=1000,test=test_MNIST,snapshot_iter=10000)


trained_model = list(model.networks.values())[0].net

torch.save(trained_model.state_dict(), 'sd_dpl_cnn.pt')