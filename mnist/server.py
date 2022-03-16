import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup, client
import sys
import time
import datetime


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=140, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=300, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
panduan = 10


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger('mnist_daluan_ceshi_309（加时间差）.txt')

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    last_acc_list = []
    init_list = [100.0, 100.0, 100.0, 100.0, 100.0]
    com_time = []
    time_list = []
    node0 = []
    node1 = []
    node2 = []
    node40 = []
    node41 = []
    node42 = []
    node60 = []
    node61 = []
    node120 = []
    node121 = []
    for i in range(140):
        last_acc_list.append(init_list)

    global_list = []
    global_loss_list = []
    for i in range(args['num_comm']):

        print("communicate round {}".format(i+1))

        # order = np.random.permutation(args['num_of_clients'])
        order = []
        for j in range(140):
            order.append(j)
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None
        jishuqi = 0
        duitou = 0
        for client in tqdm(clients_in_comm):
            last_acc = myClients.clients_set[client].lastacc(args['batchsize'], net, loss_func, global_parameters)
            print('上一次全局', last_acc)

            if last_acc < max(last_acc_list[duitou]):
                print('%d训练了' % jishuqi)

                if jishuqi == 0:
                    node0.append('1')
                elif jishuqi == 1:
                    node1.append('1')
                elif jishuqi == 2:
                    node2.append('1')
                elif jishuqi == 40:
                    node40.append('1')
                elif jishuqi == 41:
                    node41.append('1')
                elif jishuqi == 42:
                    node42.append('1')
                elif jishuqi == 60:
                    node60.append('1')
                elif jishuqi == 61:
                    node61.append('1')
                elif jishuqi == 120:
                    node120.append('1')
                elif jishuqi == 121:
                    node121.append('1')
                baocun_b = []
                for ij in range(5):
                    baocun_b.append(last_acc_list[0][ij])
                baocun_b.remove(max(baocun_b))
                baocun_b.append(float(last_acc))
                del last_acc_list[duitou]
                last_acc_list.append(baocun_b)
                ticks1 = time.time()
                local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                             loss_func, opti, global_parameters)
                ticks2 = time.time()
                time123 = ticks2 - ticks1
                time_list.append(time123)
                # 打印local_model 准确率情况
                net.load_state_dict(local_parameters, strict=True)
                torch.save(local_parameters, 'net_params%d.pth' % jishuqi)
                local_accu = 0
                local_num = 0
                # print('----------', local_accu, local_num)

                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    local_accu += (preds == label).float().mean()
                    local_num += 1
                print('local_accuracy: {}'.format(local_accu / local_num))
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in local_parameters.items():
                        sum_parameters[key] = var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] = sum_parameters[var] + local_parameters[var]
            else:
                baocun_c = []
                time_list.append(0)
                for ik in range(5):
                    baocun_c.append(last_acc_list[0][ik])
                # print(baocun_c)
                del last_acc_list[duitou]
                last_acc_list.append(baocun_c)
                print('%d没有训练' % jishuqi)
                if jishuqi == 0:
                    node0.append('0')
                elif jishuqi == 1:
                    node1.append('0')
                elif jishuqi == 2:
                    node2.append('0')
                elif jishuqi == 40:
                    node40.append('0')
                elif jishuqi == 41:
                    node41.append('0')
                elif jishuqi == 42:
                    node42.append('0')
                elif jishuqi == 60:
                    node60.append('0')
                elif jishuqi == 61:
                    node61.append('0')
                elif jishuqi == 120:
                    node120.append('0')
                elif jishuqi == 121:
                    node121.append('0')
                local_parameters = torch.load('net_params%d.pth' % jishuqi)
                local_accu = 0
                local_num = 0
                # print('----------', local_accu, local_num)

                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    local_accu += (preds == label).float().mean()
                    local_num += 1
                print('local_accuracy: {}'.format(local_accu / local_num))
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in local_parameters.items():
                        sum_parameters[key] = var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] = sum_parameters[var] + local_parameters[var]
            jishuqi = jishuqi + 1
        print('max(time_list):', max(time_list))
        print('time_list:', time_list)
        com_time.append(max(time_list))
        time_list.clear()

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    loss = F.cross_entropy(preds, label)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('global_accuracy: {}'.format(sum_accu / num))
                global_loss = float(loss)
                global_list.append(float(float(sum_accu) / float(num)))
                global_loss_list.append(global_loss)

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))
    print(global_list)
    print(global_loss_list)
    print(com_time)
    print('node0:', node0)
    print('node1:', node1)
    print('node2:', node2)
    print('node40:', node40)
    print('node41:', node41)
    print('node42:', node42)
    print('node60:', node60)
    print('node61:', node61)
    print('node120:', node120)
    print('node121:', node121)


