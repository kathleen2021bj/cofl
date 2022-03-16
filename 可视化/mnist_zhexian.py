#   -*- coding = utf-8 -*-
#   @time : 2021/9/12 12:07
#   @ File : zhexiantu.py
#   @Software: PyCharm

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
# list1 = [140, 140, 140, 140, 140, 140, 140, 136, 124, 117, 106, 113, 94, 93, 83, 90, 79, 70, 69, 61, 59, 73, 61, 64, 56, 52, 45, 48, 49, 50, 41, 43, 44, 33, 34, 38, 44, 31, 47, 45, 47, 34, 43, 30, 34, 30, 24, 32, 28, 29, 29, 19, 31, 30, 19, 32, 28, 23, 29, 24, 17, 21, 27, 22, 18, 25, 27, 24, 21, 21, 18, 16, 21, 26, 26, 17, 27, 20, 10, 24, 20, 18, 19, 16, 22, 19, 16, 17, 23, 18, 18, 13, 16, 17, 16, 10, 15, 18, 19, 16, 8, 24, 12, 12, 18, 11, 19, 17, 17, 19, 4, 7, 11, 12, 17, 15, 15, 8, 14, 10]

plt.style.use('fivethirtyeight')
data_1 = []

data_2 = []

data_3 = []
data_4 = []
data_3_list = [140, 140, 140, 140, 140, 140, 140, 136, 124, 117, 106, 113, 94, 93, 83, 90, 79, 70, 69, 61, 59, 73, 61, 64, 56, 52, 45, 48, 49, 50, 41, 43, 44, 33, 34, 38, 44, 31, 47, 45, 47, 34, 43, 30, 34, 30, 24, 32, 28, 29, 29, 19, 31, 30, 19, 32, 28, 23, 29, 24, 17, 21, 27, 22, 18, 25, 27, 24, 21, 21, 18, 16, 21, 26, 26, 17, 27, 20, 10, 24, 20, 18, 19, 16, 22, 19, 16, 17, 23, 18, 18, 13, 16, 17, 16, 10, 15, 18, 19, 16, 8, 24, 12, 12, 18, 11, 19, 17, 17, 19, 4, 7, 11, 12, 17, 15, 15, 8, 14, 10]
for i in range(len(data_3_list)):
    data_3.append(data_3_list[i]*6.34)
    data_4.append(140*6.34)
    data_2.append(140*6.34)
    data_1.append(i)
# labels = ['HL', 'ED', 'HD_2', 'HD_all', 'Voting', 'ME']

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
width_1 = 0.4
x_major_locator = MultipleLocator(10)
ax.plot(data_1, data_2, color='g', linestyle='-.', label="FedAvg")
ax.plot(data_1, data_3, color='chocolate', label="AlCofl")
ax.plot(data_1, data_4, color='darkblue', linestyle=':', label="FedProx")
ax.xaxis.set_major_locator(x_major_locator)
ax.set_ylim([1, 140*6.34 + 50])
ax.set_xlim([0, 121])
# x_values=list(range(11))
plt.ylabel('Communication cost(MB)')
plt.xlabel('Communication Rounds')
ax.legend()
plt.savefig("./mnist_zhexian.png", bbox_inches='tight')
plt.savefig("./mnist_zhexian.jpg", bbox_inches='tight')
plt.tight_layout()
plt.show()