import torch
import torch.nn as nn
import torchvision
from copy import deepcopy


class SimpleDecoder(nn.Module):
    def __init__(self, input_size, num_class):
        super(SimpleDecoder, self).__init__()
        self.fc = nn.Linear(input_size, num_class)

    def forward(self, x):
        x = self.fc(x)
        return x


class MiniNet(nn.Module):
    def __init__(self, class_num, args, num_decoder=1):
        super(MiniNet, self).__init__()
        self.num_decoder = num_decoder
        self.layers1 = nn.Sequential(  # input (batch_size, 3, 64, 64)
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (batch_size, 32, 32, 32)
        )
        self.layers2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (batch_size, 32, 16, 16)
        )
        self.layers3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (batch_size, 32, 8, 8)
        )
        self.layers4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (batch_size, 32, 4, 4)
        )
        self.layers = nn.Sequential(
            self.layers1,
            self.layers2,
            self.layers3,
            self.layers4
        )
        input_size = 32*4*4
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder):
            if i == 0:
                self.decoder.append(SimpleDecoder(input_size, class_num))
            else:
                if not args.unequal_classes_num:
                    self.decoder.append(SimpleDecoder(input_size, args.extra_classes_num))
                else:
                    if args.cifar100:
                        self.decoder.append(SimpleDecoder(input_size, 5 * i))
                    elif args.miniimagenet:
                        self.decoder.append(SimpleDecoder(input_size, 100+10 * i))
                    else:
                        self.decoder.append(SimpleDecoder(input_size, 2 if i == 1 else 5 * (i - 1)))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.layers(x)

        x = x.view(batch_size, -1)
        output = []
        for d in self.decoder:
            tmp = d(x)
            output.append(tmp)
        return output


class CifarNet(nn.Module):
    def __init__(self, class_num, args, num_decoder=1):
        super(CifarNet, self).__init__()
        self.num_decoder = num_decoder
        self.layer1 = nn.Sequential(  # input (batch_size, 3, 32, 32)
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # (batch_size, 64, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (batch_size, 64, 16, 16)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (batch_size, 128, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (batch_size, 128, 8, 8)
        )
        self.layers = nn.Sequential(
            self.layer1,
            self.layer2
        )
        input_size = 128 * 8 * 8
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder):
            if i == 0:
                self.decoder.append(SimpleDecoder(input_size, class_num))
            else:
                if not args.unequal_classes_num:
                    self.decoder.append(SimpleDecoder(input_size, args.extra_classes_num))
                else:
                    if args.cifar100:
                        self.decoder.append(SimpleDecoder(input_size, 5 * i))
                    else:
                        self.decoder.append(SimpleDecoder(input_size, 2 if i==1 else 5*(i-1)))


    def forward(self, x):
        batch_size = x.size(0)
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(batch_size, -1)
        output = []
        for d in self.decoder:
            tmp = d(x)
            output.append(tmp)
        return output
