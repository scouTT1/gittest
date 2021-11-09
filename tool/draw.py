import os
import re
import csv
from matplotlib import pyplot as plt
import numpy as np
    
with open("checkpoint/copy_train_summary.csv", 'r') as f:
    reader = csv.reader(f)
    x=[]
    y=[]
    for line in reader:
        x.append(int(line[0]))
        list_y = re.split('\[|\]', line[2])[1]
        list_y = list_y.split()
        print(list_y)
        list_y = [float(list_y[i]) for i in range(len(list_y))]
        y.append(max(list_y))
    
    plt.plot(x, y)
    plt.show()
