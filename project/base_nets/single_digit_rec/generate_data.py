import torchvision
import random

trainset = torchvision.datasets.MNIST(root='../../../data/MNIST', train=True, download=True)
testset = torchvision.datasets.MNIST(root='../../../data/MNIST', train=False, download=True)



def next_example(dataset,i):
    x = next(i)
    (_,c1) = dataset[x]
    return x, c1

def gather_examples(dataset,filename):
    examples = list()
    i = list(range(len(dataset)))
    random.shuffle(i)
    i = iter(i)
    while(True):
        try:
            examples.append(next_example(dataset,i))
        except StopIteration:
            break 

    with open(filename,'w') as f:
        for example in examples:
            f.write('{} {}\n'.format(example[0],example[1]))

gather_examples(trainset,'train_data.txt')
gather_examples(testset,'test_data.txt')
