import json
import matplotlib.pyplot as plt
import numpy as np
import pprint

labels = {
    'notl_abs_dpl': 'DPL',
    'notl_abs_nn': 'CNN',
    #'notl_add2abs_dpl': 'DPL: NN from ADD',
    #'notl_rec2abs_dpl': 'DPL: NN from REC',
    'tl_add2abs_dpl': 'TL-DPL: NN from ADD',
    'tl_add2abs_nn': 'TL-CNN: NN from ADD',
    'tl_rec2abs_dpl': 'TL-DPL: NN from REC'}

data = {
    'notl_abs_dpl': {},
    'notl_abs_nn': {},
    #'notl_add2abs_dpl': {},
    #'notl_rec2abs_dpl': {},
    'tl_add2abs_dpl': {},
    'tl_add2abs_nn': {},
    'tl_rec2abs_dpl': {}
    }



acc_steps = [str(i*1000) for i in range(1,30)]
loss_steps = [str(i*1000) for i in range(1,30)]


pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(data)

# Acc Plot
for key,value in data.items():
    #print()
    #print(key)
    #print(value['Accuracy'])
    plt.plot(acc_steps, [value['Accuracy'][step] for step in acc_steps], label=labels[key])
plt.xlabel('Iterations', fontsize=18) # Batch size = 2
plt.ylabel('Accuracy', fontsize=18)
plt.xticks(['10000','20000','30000'],['10000','20000','30000'],fontsize=16)
plt.yticks(fontsize=14)
plt.title('Accuracies of the different models', fontsize=20)
plt.legend(fontsize=14)
plt.savefig('acc_plot')
plt.show()
#plt.clf()

# Loss plot
for key,value in data.items():
    print(key)
    plt.plot(loss_steps, [value['loss'][step] for step in loss_steps], label=labels[key])
plt.xlabel('Iterations', fontsize=18)
plt.ylabel('NLL-Loss', fontsize=18)
plt.xticks(['10000','20000','30000'],['10000','20000','30000'], fontsize=16)
plt.yticks(fontsize=14)
plt.title('Negative Log Likelihood Loss of different models', fontsize=20)
plt.legend(fontsize=14)
plt.savefig('loss_plot')
plt.show()
