import json

from train import train_model
from data_loader import load
from examples.NIPS.MNIST.mnist import test_MNIST, MNIST_Net, MNIST_Net2, neural_predicate
from model import Model
from optimizer import Optimizer
from network import Network
import torch


queries = load('train_data.txt')

with open('abs.pl') as f:
    problog_string = f.read()


network = MNIST_Net()
network.load_state_dict(torch.load('sd_rec_cnn.pt'))
network.eval()
net = Network(network, 'mnist_net', neural_predicate)
net.optimizer = torch.optim.Adam(network.parameters(),lr = 0.001)
model = Model(problog_string, [net], caching=False)
optimizer = Optimizer(model, 2)

log = {}

logs = test_MNIST(model)
for e in logs:
    log[e[0]] = e[1]

with open('notl_rec2abs_dpl.json', 'w') as outfile:
    json.dump(log, outfile)

#train_model(model,queries, 1, optimizer,test_iter=500,test=test_MNIST,snapshot_iter=10000)

