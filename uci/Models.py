import torch
import torch.nn as nn
import torch.nn.functional as F


# class Mnist_2NN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(561, 2048)
#         self.fc2 = nn.Linear(2048, 1024)
#         self.fc3 = nn.Linear(1024, 256)
#         self.fc4 = nn.Linear(256, 12)
#
#     def forward(self, inputs):
#         tensor = F.relu(self.fc1(inputs))
#         tensor = F.relu(self.fc2(tensor))
#         tensor = F.relu(self.fc3(tensor))
#         tensor = self.fc4(tensor)
#         return tensor


class Mnist_CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(561, 100, 1, batch_first=True)
        self.fc1 = nn.Linear(561, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 12)

    def forward(self, inputs):
        inputs = inputs.view(len(inputs), 1, -1)  # 把原有2维度[a,b]改为3维[a,1,b]
        h0 = torch.zeros(1, inputs.size(0), 100).requires_grad_().to(self.device)
        c0 = torch.zeros(1, inputs.size(0), 100).requires_grad_().to(self.device)
        out, (hn, cn) = self.lstm(inputs.to(self.device), (h0.detach().to(self.device), c0.detach().to(self.device)))
        # tensor = F.relu(self.fc1(inputs))
        # tensor = F.relu(self.fc2(tensor))
        # tensor = F.relu(self.fc3(tensor))
        # tensor = self.fc4(tensor)
        print(hn.size())
        print(cn.size())
        out = self.fc4(out[:, -1, :])
        # return tensor
        return out



class Mnist_2NN(nn.Module):
    def __init__(self, input_dim=561, hidden_dim=100, layer_dim=2, output_dim=12):
        super(Mnist_2NN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(len(x), 1, -1)  # 把原有2维度[a,b]改为3维[a,1,b]
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device=0)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device=0)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out




class Mnist_CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #self.fc1 = nn.Linear(7*7*64, 512)
        self.fc1 = nn.Linear(561, 512)
        self.fc2 = nn.Linear(512, 6)

    def forward(self, inputs):
        tensor = inputs.view(-1, 561)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 561)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor

