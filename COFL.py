#   -*- coding = utf-8 -*-
#   @time : 2021/8/6 11:52
#   @ File : 10node cheshi.py
#   @Software: PyCharm

import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras import regularizers
from Block import Block
import datetime

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import metrics
from tensorflow.keras.models import model_from_json

from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import sys




class Data():
    def __init__(self, data):
        self.data = data

    def sample(self, start, end):
        size = len(self.data)
        return self.data[int(size * start):int(size * end)]

    def sample_smote(self, start, end):
        data = self.sample(start, end)
        dataSmote = np.array(data.drop("Class", axis=1))
        y = np.array(data[['Class']])
        smt = SMOTE()
        dataSmote, y = smt.fit_resample(dataSmote, y)  # SMOTE 重采样
        dataSmote = pd.DataFrame(dataSmote)
        y = pd.DataFrame(y)
        dataSmote['Class'] = y
        dataSmote = dataSmote.sample(frac=1)  # 打乱顺序
        dataSmote.columns = list(data.columns)  # 将原始数据 data 的列名设置为 dataSmote 的列名
        return dataSmote


class Aggregator():

    def __init__(self):
        self.wB1 = 0.05
        self.wB2 = 0.15
        self.wB3 = 0.05
        self.wB4 = 0.05
        self.wB5 = 0.08
        self.wB6 = 0.06
        self.wB7 = 0.10
        self.wB8 = 0.07
        self.wB9 = 0.09
        self.wB10 = 0.30

    def aggregate(self, delta, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10):
        delta = np.array(delta, dtype=object)
        temp = (self.wB1 * np.array(B1, dtype=object) + self.wB2 * np.array(B2, dtype=object) + self.wB3 * np.array(B3,
                                                                                                                    dtype=object)) + self.wB4 * np.array(
            B4, dtype=object) + self.wB5 * np.array(B5, dtype=object) + self.wB6 * np.array(B6,
                                                                                            dtype=object) + self.wB7 * np.array(
            B7, dtype=object) + self.wB8 * np.array(B8, dtype=object) + self.wB9 * np.array(B9,
                                                                                            dtype=object) + self.wB10 * np.array(
            B10, dtype=object)
        temp -= delta
        delta += temp

        return delta


class Model():

    def __init__(self):
        self.input_shape = (30,)
        self.model = Sequential()
        self.model.add(
            Dense(32, activation='relu', input_shape=self.input_shape, kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam',  # rmsprop
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def saveModel(self):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")

    def loadModel(self):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return loaded_model

    def run(self, X, Y, validation_split=0.1, load=True):
        if (load):
            self.model = self.loadModel()
        self.model.fit(X, Y, epochs=2, validation_split=validation_split, verbose=1)        # 修改epochs = 2

    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y)[1] * 100

    def loss(self, X, Y):
        return self.model.evaluate(X, Y)[0]

    def predict(self, X):
        return self.model.predict(X)

    def getWeights(self):
        return self.model.get_weights()

    def setWeights(self, weight):
        self.model.set_weights(weight)


class Bank(Model):

    def __init__(self, data, split_size=0):
        super().__init__()
        self.data = data
        self.split(split_size)

    def setData(self, data, split_size=0):
        self.data = data
        self.split(split_size)

    def getData(self):
        return self.data

    def split(self, split_size):
        X = self.data.copy()
        X.drop(['Class'], axis=1, inplace=True)
        Y = self.data[['Class']]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=split_size)


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


#
# ticks = time.time()
# sys.stdout = Logger('%f.txt' %(ticks))

data = pd.read_csv('creditcard.csv')
data.head()

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))  # 标准化 'Amount' 列到 'scaled_amount' 列
data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))  # 标准化 'Time' 列到 'scaled_time' 列
data.drop(['Time','Amount'], axis=1, inplace=True)  # 去掉原来的 'Amount' 列和 'Time' 列

scaled_amount = data['scaled_amount']
scaled_time = data['scaled_time']

data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
data.insert(0, 'scaled_amount', scaled_amount)  # 将 'scaled_amount' 列放到 data 中第一列
data.insert(1, 'scaled_time', scaled_time)  # 将 'scaled_time' 列放到 data 中第二列

data.head()

data = data.sample(frac=1)  # 对 data 打乱顺序

# amount of fraud classes 492 rows.
fraud_data = data.loc[data['Class'] == 1]  # 取出 data 中所有 1 类样本放到 fraud_data 中
non_fraud_data = data.loc[data['Class'] == 0]  # 取出 data 中所有 0 类样本放到 non_fraud_data 中

normal_distributed_data = pd.concat([fraud_data, non_fraud_data])  # 将 fraud_data 表格和 non_fraud_data 表格整合

# Shuffle dataframe rows
new_data = normal_distributed_data.sample(frac=1, random_state=42)  # 对 normal_distributed_data 打乱顺序并赋给 new_data

new_data.head()


results = {}
aggregator = Aggregator()

datum=Data(data)

Data_Global = datum.sample_smote(0, 0.1)     # 将数据集前百分之10作为初始数据集
Data_Model_1 = datum.sample_smote(0.1, 0.13)
Data_Model_2 = datum.sample_smote(0.13, 0.22)
Data_Model_3 = datum.sample_smote(0.22, 0.25)
Data_Model_4 = datum.sample_smote(0.25, 0.3)
Data_Model_5 = datum.sample_smote(0.3, 0.35)            # 将中间百分之80，不同比例分给10个联邦节点
Data_Model_6 = datum.sample_smote(0.35, 0.4)
Data_Model_7 = datum.sample_smote(0.4, 0.5)
Data_Model_8 = datum.sample_smote(0.5, 0.55)
Data_Model_9 = datum.sample_smote(0.55, 0.6)
Data_Model_10 = datum.sample_smote(0.6, 0.9)
Data_Test = datum.sample_smote(0.9, 1.0)    # 将数据集后百分之10作为测试数据集


GlobalBank=Bank(Data_Global,0.2)  # 初始化服务器模型
GlobalBank.run(GlobalBank.X_train, GlobalBank.Y_train, load=False)

results['BankG.1']=GlobalBank.evaluate(GlobalBank.X_test,GlobalBank.Y_test)

GlobalBank.saveModel()

# 初始化三个客户端模型
Bank1 = Bank(Data_Model_1, 0.2)   # Bank1初始化，将百分之80的数据训练，百分之20的数据测试
Bank2 = Bank(Data_Model_2, 0.2)
Bank3 = Bank(Data_Model_3, 0.2)
Bank4 = Bank(Data_Model_4, 0.2)
Bank5 = Bank(Data_Model_5, 0.2)
Bank6 = Bank(Data_Model_6, 0.2)
Bank7 = Bank(Data_Model_7, 0.2)
Bank8 = Bank(Data_Model_8, 0.2)
Bank9 = Bank(Data_Model_9, 0.2)
Bank10 = Bank(Data_Model_10, 0.2)

Bank1_acc = Bank1.evaluate(Bank1.X_test, Bank1.Y_test)
Bank2_acc = Bank2.evaluate(Bank2.X_test, Bank2.Y_test)
Bank3_acc = Bank3.evaluate(Bank3.X_test, Bank3.Y_test)
Bank4_acc = Bank4.evaluate(Bank4.X_test, Bank4.Y_test)
Bank5_acc = Bank5.evaluate(Bank5.X_test, Bank5.Y_test)
Bank6_acc = Bank6.evaluate(Bank6.X_test, Bank6.Y_test)
Bank7_acc = Bank7.evaluate(Bank7.X_test, Bank7.Y_test)
Bank8_acc = Bank8.evaluate(Bank8.X_test, Bank8.Y_test)
Bank9_acc = Bank9.evaluate(Bank9.X_test, Bank9.Y_test)
Bank10_acc = Bank10.evaluate(Bank10.X_test, Bank10.Y_test)

join_sum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
join_sum1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
join_sum2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
join_sum3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
join_sum4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
join_sum5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
join_sum6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
join_sum7 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
join_sum8 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
join_sum9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
join_sum10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
i = 1

for j in range(10):
    # 将服务器模型参数传递给客户端模型
    Bank1.setWeights(GlobalBank.getWeights())
    Bank2.setWeights(GlobalBank.getWeights())
    Bank3.setWeights(GlobalBank.getWeights())
    Bank4.setWeights(GlobalBank.getWeights())
    Bank5.setWeights(GlobalBank.getWeights())
    Bank6.setWeights(GlobalBank.getWeights())
    Bank7.setWeights(GlobalBank.getWeights())
    Bank8.setWeights(GlobalBank.getWeights())
    Bank9.setWeights(GlobalBank.getWeights())
    Bank10.setWeights(GlobalBank.getWeights())

    # 将服务器模型测试本地数据
    Bank1_benlun_acc = Bank1.evaluate(Bank1.X_test, Bank1.Y_test)
    Bank2_benlun_acc = Bank2.evaluate(Bank2.X_test, Bank2.Y_test)
    Bank3_benlun_acc = Bank3.evaluate(Bank3.X_test, Bank3.Y_test)
    Bank4_benlun_acc = Bank4.evaluate(Bank4.X_test, Bank4.Y_test)
    Bank5_benlun_acc = Bank5.evaluate(Bank5.X_test, Bank5.Y_test)
    Bank6_benlun_acc = Bank6.evaluate(Bank6.X_test, Bank6.Y_test)
    Bank7_benlun_acc = Bank7.evaluate(Bank7.X_test, Bank7.Y_test)
    Bank8_benlun_acc = Bank8.evaluate(Bank8.X_test, Bank8.Y_test)
    Bank9_benlun_acc = Bank9.evaluate(Bank9.X_test, Bank9.Y_test)
    Bank10_benlun_acc = Bank10.evaluate(Bank10.X_test, Bank10.Y_test)

    if Bank1_acc <= Bank1_benlun_acc:
        Bank1.run(Bank1.X_train, Bank1.Y_train)
        Bank1_acc = Bank1_benlun_acc
        print('第' + str(i) + '次' + 'Bank1更新模型精度高，进行训练')
        join_sum[j] = join_sum[j] + 1
        join_sum1[j] = join_sum1[j] + 1
    else:
        print('第' + str(i) + '次' + 'Bank1更新模型精度低，未进行训练')

    if Bank2_acc <= Bank2_benlun_acc:
        Bank2.run(Bank2.X_train, Bank2.Y_train)
        Bank2_acc = Bank2_benlun_acc
        print('第' + str(i) + '次' + 'Bank2更新模型精度高，进行训练')
        join_sum[j] = join_sum[j] + 1
        join_sum2[j] = join_sum2[j] + 1
    else:
        print('第' + str(i) + '次' + 'Bank2更新模型精度低，未进行训练')

    if Bank3_acc <= Bank3_benlun_acc:
        Bank3.run(Bank3.X_train, Bank3.Y_train)
        Bank3_acc = Bank3_benlun_acc
        print('第' + str(i) + '次' + 'Bank3更新模型精度高，进行训练')
        join_sum[j] = join_sum[j] + 1
        join_sum3[j] = join_sum3[j] + 1
    else:
        print('第' + str(i) + '次' + 'Bank3更新模型精度低，未进行训练')

    if Bank4_acc <= Bank4_benlun_acc:
        Bank4.run(Bank4.X_train, Bank4.Y_train)
        Bank4_acc = Bank4_benlun_acc
        print('第' + str(i) + '次' + 'Bank4更新模型精度高，进行训练')
        join_sum[j] = join_sum[j] + 1
        join_sum4[j] = join_sum4[j] + 1
    else:
        print('第' + str(i) + '次' + 'Bank4更新模型精度低，未进行训练')

    if Bank5_acc <= Bank5_benlun_acc:
        Bank5.run(Bank5.X_train, Bank5.Y_train)
        Bank5_acc = Bank5_benlun_acc
        print('第' + str(i) + '次' + 'Bank5更新模型精度高，进行训练')
        join_sum[j] = join_sum[j] + 1
        join_sum5[j] = join_sum5[j] + 1
    else:
        print('第' + str(i) + '次' + 'Bank5更新模型精度低，未进行训练')

    if Bank6_acc <= Bank6_benlun_acc:
        Bank6.run(Bank6.X_train, Bank6.Y_train)
        Bank6_acc = Bank6_benlun_acc
        print('第' + str(i) + '次' + 'Bank6更新模型精度高，进行训练')
        join_sum[j] = join_sum[j] + 1
        join_sum6[j] = join_sum6[j] + 1
    else:
        print('第' + str(i) + '次' + 'Bank6更新模型精度低，未进行训练')

    if Bank7_acc <= Bank7_benlun_acc:
        Bank7.run(Bank7.X_train, Bank7.Y_train)
        Bank7_acc = Bank7_benlun_acc
        print('第' + str(i) + '次' + 'Bank7更新模型精度高，进行训练')
        join_sum[j] = join_sum[j] + 1
        join_sum7[j] = join_sum7[j] + 1
    else:
        print('第' + str(i) + '次' + 'Bank7更新模型精度低，未进行训练')

    if Bank8_acc <= Bank8_benlun_acc:
        Bank8.run(Bank8.X_train, Bank8.Y_train)
        Bank8_acc = Bank8_benlun_acc
        print('第' + str(i) + '次' + 'Bank8更新模型精度高，进行训练')
        join_sum[j] = join_sum[j] + 1
        join_sum8[j] = join_sum8[j] + 1
    else:
        print('第' + str(i) + '次' + 'Bank8更新模型精度低，未进行训练')

    if Bank9_acc <= Bank9_benlun_acc:
        Bank9.run(Bank9.X_train, Bank9.Y_train)
        Bank9_acc = Bank9_benlun_acc
        print('第' + str(i) + '次' + 'Bank9更新模型精度高，进行训练')
        join_sum[j] = join_sum[j] + 1
        join_sum9[j] = join_sum9[j] + 1
    else:
        print('第' + str(i) + '次' + 'Bank9更新模型精度低，未进行训练')

    if Bank10_acc <= Bank10_benlun_acc:
        Bank10.run(Bank10.X_train, Bank10.Y_train)
        Bank10_acc = Bank10_benlun_acc
        print('第' + str(i) + '次' + 'Bank10更新模型精度高，进行训练')
        join_sum[j] = join_sum[j] + 1
        join_sum10[j] = join_sum10[j] + 1
    else:
        print('第' + str(i) + '次' + 'Bank10更新模型精度低，未进行训练')

    # # 使用对应数据训练客户端模型
    # Bank1.run(Bank1.X_train, Bank1.Y_train)
    # Bank2.run(Bank2.X_train, Bank2.Y_train)
    # Bank3.run(Bank3.X_train, Bank3.Y_train)

    # 集成客户端模型参数
    delta = aggregator.aggregate(GlobalBank.getWeights(), Bank1.getWeights(), Bank2.getWeights(), Bank3.getWeights(),
                                 Bank4.getWeights(), Bank5.getWeights(), Bank6.getWeights(), Bank7.getWeights(),
                                 Bank8.getWeights(), Bank9.getWeights(), Bank10.getWeights())

    # 将集成后的客户端模型参数分配到服务器模型上
    GlobalBank.setWeights(delta)
    i = i + 1
GlobalBank.saveModel()


results['Bank1.1']=Bank1.evaluate(Bank1.X_test,Bank1.Y_test)  # 用 Bank1 客户端数据测试 Bank1 模型
results['Bank2.1']=Bank2.evaluate(Bank2.X_test,Bank2.Y_test)  # 用 Bank2 客户端数据测试 Bank2 模型
results['Bank3.1']=Bank3.evaluate(Bank3.X_test,Bank3.Y_test)  # 用 Bank3 客户端数据测试 Bank3 模型
results['Bank4.1']=Bank4.evaluate(Bank4.X_test,Bank4.Y_test)  # 用 Bank4 客户端数据测试 Bank4 模型
results['Bank5.1']=Bank5.evaluate(Bank5.X_test,Bank5.Y_test)  # 用 Bank5 客户端数据测试 Bank5 模型
results['Bank6.1']=Bank6.evaluate(Bank6.X_test,Bank6.Y_test)
results['Bank7.1']=Bank7.evaluate(Bank7.X_test,Bank7.Y_test)
results['Bank8.1']=Bank8.evaluate(Bank8.X_test,Bank8.Y_test)
results['Bank9.1']=Bank9.evaluate(Bank9.X_test,Bank9.Y_test)
results['Bank10.1']=Bank10.evaluate(Bank10.X_test,Bank10.Y_test)
results['BankG.2']=GlobalBank.evaluate(GlobalBank.X_test,GlobalBank.Y_test)  # 用 GlobalBank 服务器数据测试 GlobalBank 模型
GlobalBank.setData(Data_Test,0.9)  # 将服务器数据改为 Data_Test
results['BankG.3']=GlobalBank.evaluate(GlobalBank.X_test,GlobalBank.Y_test)  # 用 Data_Test 数据测试 GlobalBank 模型

results
ticks = time.time()
sys.stdout = Logger('%f.txt' %(ticks))              # 只打印11个矩阵
print(join_sum)
print(join_sum1)
print(join_sum2)
print(join_sum3)
print(join_sum4)
print(join_sum5)
print(join_sum6)
print(join_sum7)
print(join_sum8)
print(join_sum9)
print(join_sum10)
