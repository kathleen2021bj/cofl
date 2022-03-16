#   -*- coding = utf-8 -*-
#   @time : 2021/9/12 10:46
#   @ File : 柱状图.py
#   @Software: PyCharm

# !/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
# plt.style.use('fivethirtyeight')
data_1 = [8, 18, 28, 38, 48, 58, 68, 78, 88, 98, 108, 118]
data_4 = [12, 22, 32, 42, 52, 62, 72, 82, 92, 102, 112, 122]
data_6 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
data_2 = [1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400]
data_5 = [1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400]
data_3 = [1339, (2203-1339), (2766-2203), (3160-2766), (3449-3160), (3770-3449), (3984-3770), (4142-3984), (4309-4142), (4472-4309), (4627-4472), (4759-4627)]


# labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
width_1 = 4
x_major_locator = MultipleLocator(10)
ax.bar(data_1, data_2, width=width_1, color='g', label="FedAvg and FedProx")
# ax.bar(data_6, data_5, width=width_1, color='darkblue', label="FedProx")

ax.bar(data_4, data_3, width=width_1, color='chocolate', label="AIcofl")
# ax.bar(data_1, data_2, width=width_1, color='g')
# # ax.bar(data_6, data_5, width=width_1, color='darkblue', label="FedProx")
#
# ax.bar(data_4, data_3, width=width_1, color='chocolate')
# ax.set_xticks([])
ax.xaxis.set_major_locator(x_major_locator)
ax.set_ylim([1, 1410])
ax.set_xlim([0, 136])
plt.ylabel('Number of participating nodes', fontsize=20)
plt.xlabel('Communication Rounds', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.legend(fontsize=20)
plt.savefig("./mnist_tongxun3.png", bbox_inches='tight')
plt.savefig("./mnist_tongxun3.jpg", bbox_inches='tight')
plt.tight_layout()
plt.show()