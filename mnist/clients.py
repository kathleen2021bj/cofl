import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
import random


class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None


    def lastloss(self, localBatchSize, Net, lossFun, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        local_accu_chushi = 0
        local_num_chushi = 0
        for data, label in self.train_dl:
            data, label = data.to(self.dev), label.to(self.dev)
            preds = Net(data)
            loss = lossFun(preds, label)
            preds = torch.argmax(preds, dim=1)
            local_accu_chushi += (preds == label).float().mean()
            local_num_chushi += 1
        # lastacc = float(float(local_accu_chushi) / float(local_num_chushi))
        loss1 = float(loss)
        return loss1


    def lastacc(self, localBatchSize, Net, lossFun, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        local_accu_chushi = 0
        local_num_chushi = 0
        for data, label in self.train_dl:
            data, label = data.to(self.dev), label.to(self.dev)
            preds = Net(data)
            loss = lossFun(preds, label)
            preds = torch.argmax(preds, dim=1)
            local_accu_chushi += (preds == label).float().mean()
            local_num_chushi += 1
        lastacc = float(float(local_accu_chushi) / float(local_num_chushi))
        # loss1 = float(loss)
        return lastacc

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()

        return Net.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}

        self.test_data_loader = None

        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        index = [i for i in range(len(train_data))]
        random.shuffle(index)
        train_data = train_data[index]
        train_label = train_label[index]

        # train_data1 = mnistDataSet.train_data
        # train_label1 = mnistDataSet.train_label

        # shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
        # shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
        # for i in range(self.num_of_clients):
        #     shards_id1 = shards_id[i * 2]
        #     shards_id2 = shards_id[i * 2 + 1]
        #     data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
        #     data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
        #     label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
        #     label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
        #     local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
        #     local_label = np.argmax(local_label, axis=1)
        #     someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
        #     self.clients_set['client{}'.format(i)] = someone




        # for i in range(self.num_of_clients):
        #     # shards_id1 = shards_id[i * 2]
        #     # shards_id2 = shards_id[i * 2 + 1]
        #     # data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
        #     # data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
        #     # label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
        #     # label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
        #     # local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
        #     # local_label = np.argmax(local_label, axis=1)
        #     # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
        #     # self.clients_set['client{}'.format(i)] = someone
        #
        #     if i < 40:
        #         shard_size = 500
        #         data_shards = train_data[i*shard_size: i*shard_size + shard_size]
        #         label_shards = train_label[i*shard_size: i*shard_size + shard_size]
        #         local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
        #         local_label = np.argmax(local_label, axis=1)
        #         # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev,)
        #         someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
        #         self.clients_set['client{}'.format(i)] = someone
        #         print('第', i, '个节点:', len(data_shards))
        #     elif 39 < i < 60:
        #         shard_size = 100
        #         j = i - 40
        #         data_shards = train_data[20000 + j * shard_size: 20000 + j * shard_size + shard_size]
        #         label_shards = train_label[20000 + j * shard_size: 20000 + j * shard_size + shard_size]
        #         local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
        #         local_label = np.argmax(local_label, axis=1)
        #         # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev,)
        #         someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
        #         self.clients_set['client{}'.format(i)] = someone
        #         print('第', i, '个节点:', len(data_shards))
        #     elif 59 < i < 120:
        #         shard_size = 300
        #         j = i - 60
        #         data_shards = train_data[22000 + j * shard_size: 22000 + j * shard_size + shard_size]
        #         label_shards = train_label[22000 + j * shard_size: 22000 + j * shard_size + shard_size]
        #         local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
        #         local_label = np.argmax(local_label, axis=1)
        #         # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev,)
        #         someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
        #         self.clients_set['client{}'.format(i)] = someone
        #         print('第', i, '个节点:', len(data_shards))
        #     else:
        #         shard_size = 1000
        #         j = i - 120
        #         data_shards = train_data[40000 + j * shard_size: 40000 + j * shard_size + shard_size]
        #         label_shards = train_label[40000 + j * shard_size: 40000 + j * shard_size + shard_size]
        #         local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
        #         local_label = np.argmax(local_label, axis=1)
        #         # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev,)
        #         someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
        #         self.clients_set['client{}'.format(i)] = someone
        #         print('第', i, '个节点:', len(data_shards))

        for i in range(self.num_of_clients):
            # shards_id1 = shards_id[i * 2]
            # shards_id2 = shards_id[i * 2 + 1]
            # data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            # data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            # label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            # label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            # local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            # local_label = np.argmax(local_label, axis=1)
            # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            # self.clients_set['client{}'.format(i)] = someone

            if i < 2:                # 2 3000
                shard_size = 3000
                data_shards = train_data[i*shard_size: i*shard_size + shard_size]
                label_shards = train_label[i*shard_size: i*shard_size + shard_size]
                local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
                local_label = np.argmax(local_label, axis=1)
                # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev,)
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
                print('第', i, '个节点:', len(data_shards))
            elif 1 < i < 16:      # 14 1000
                shard_size = 1000
                j = i - 2
                data_shards = train_data[6000 + j * shard_size: 6000 + j * shard_size + shard_size]
                label_shards = train_label[6000 + j * shard_size: 6000 + j * shard_size + shard_size]
                local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
                local_label = np.argmax(local_label, axis=1)
                # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev,)
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
                print('第', i, '个节点:', len(data_shards))
            elif 15 < i < 21:    # 5 100 iid
                shard_size = 100
                j = i - 16
                data_shards = train_data[20000 + j * shard_size: 20000 + j * shard_size + shard_size]
                label_shards = train_label[20000 + j * shard_size: 20000 + j * shard_size + shard_size]
                local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
                local_label = np.argmax(local_label, axis=1)
                # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev,)
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
                print('第', i, '个节点:', len(data_shards))
            elif 20 < i < 36:      # 15 100 no iid
                shard_size = 100
                j = i - 21
                data_shards = train_data[20500 + j * shard_size: 20500 + j * shard_size + shard_size]
                label_shards = train_label[20500 + j * shard_size: 20500 + j * shard_size + shard_size]
                local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
                local_label = np.argmax(local_label, axis=1)
                # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev,)
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
                print('第', i, '个节点:', len(data_shards))

            elif 35 < i < 96:   # 60 300 no iid
                shard_size = 300
                j = i - 36
                data_shards = train_data[22000 + j * shard_size: 22000 + j * shard_size + shard_size]
                label_shards = train_label[22000 + j * shard_size: 22000 + j * shard_size + shard_size]
                local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
                local_label = np.argmax(local_label, axis=1)
                # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev,)
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
                print('第', i, '个节点:', len(data_shards))
            else:   # 40 500
                shard_size = 500
                j = i - 96
                data_shards = train_data[40000 + j * shard_size: 40000 + j * shard_size + shard_size]
                label_shards = train_label[40000 + j * shard_size: 40000 + j * shard_size + shard_size]
                local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
                local_label = np.argmax(local_label, axis=1)
                # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev,)
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
                print('第', i, '个节点:', len(data_shards))

if __name__=="__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])


