from torch import nn
import torch.nn.functional as F
import torch


class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size1=128, hidden_size2=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out



# class MLP(nn.Module):
#     def __init__(self, args):
#         super(MLP, self).__init__()
#
#         self.input_size = 784
#         self.hidden_sizes = [100]
#         self.output_size = 10
#
#         layers = []
#         sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
#         for i in range(len(sizes) - 1):
#             layers.append(nn.Linear(sizes[i], sizes[i+1]))
#             layers.append(nn.ReLU())
#
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.model(x)
#         return x
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes in CIFAR-10

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class MLR(nn.Module):
    def __init__(self, args):
        input_size = 784
        output_size = 10
        super(MLR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

class SVM(nn.Module):
    def __init__(self):
        input_size = 60
        output_size = 10
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

class Model1(nn.Module):
    def __init__(self, args):
        super(Model1, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d((2,2)),
        )
        self.linear = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)

        return x

class Model2(nn.Module):
    def __init__(self, args):
        super(Model2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d((2, 2)),
        )
        self.linear = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x


class CNN1(nn.Module):
    def __init__(self, in_channels=3, num_classes=6):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.lin1 = nn.Linear(3136, 64)
        self.lin2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x



class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, padding=3)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.out = nn.Linear(64 * 7 * 7, 62)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.flatten(1)
        # return self.dense2(self.act(self.dense1(x)))
        return self.out(x)
