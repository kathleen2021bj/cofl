#   -*- coding = utf-8 -*-
#   @time : 2021/9/12 12:07
#   @ File : zhexiantu.py
#   @Software: PyCharm

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
plt.style.use('fivethirtyeight')
data_1 = []

data_2 = []

data_3 = []
data_4 = []
data_3_list = [22, 22, 22, 22, 22, 22, 21, 17, 17, 16, 14, 15, 11, 16, 14, 9, 9, 10, 8, 9, 4, 5, 6, 7, 4, 5, 5, 2, 5, 5, 5, 5, 4, 4, 5, 5, 5, 4, 6, 3, 2, 4, 5, 1, 3, 3, 0, 2, 2, 6, 6, 4, 3, 6, 5, 2, 2, 1, 4, 4, 1, 1, 1, 3, 0, 5, 0, 2, 2, 1, 3, 0, 1, 2, 3, 1, 1, 0, 3, 1]
for i in range(len(data_3_list)):
    data_3.append(data_3_list[i]*13.4)
    data_4.append(22*13.4)
    data_2.append(22*13.4)
    data_1.append(i)
# labels = ['HL', 'ED', 'HD_2', 'HD_all', 'Voting', 'ME']

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
width_1 = 0.4
x_major_locator = MultipleLocator(10)
ax.plot(data_1, data_2, color='g', linestyle='-.', label="FedAvg")
ax.plot(data_1, data_3, color='chocolate', label="AlCofl")
ax.plot(data_1, data_4, color='darkblue', linestyle=':', label="FedProx")
ax.xaxis.set_major_locator(x_major_locator)
ax.set_ylim([0.001, 22*13.4 + 15])
ax.set_xlim([0, 81])
# x_values=list(range(11))
plt.ylabel('Communication cost(MB)')
plt.xlabel('Communication Rounds')
ax.legend()
plt.savefig("./zishi_zhexian.png", bbox_inches='tight')
plt.savefig("./zishi_zhexian.jpg", bbox_inches='tight')
plt.tight_layout()
plt.show()