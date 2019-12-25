import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
import time
import os
import torchvision
import torchvision.transforms as transforms
# from sklearn.cluster import KMeans
from sklearn.externals import joblib
import argparse
from model import *
import copy
from util import *


def replace_grad(parameter_gradients, parameter_name):

    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_


def train_sgd(model, device):
    max_acc = 0.0
    initepoch = 0
    path = 'weights.tar'

    model_tmp = copy.deepcopy(model)
    model_tmp = model_tmp.to(device)

    alpha = 0.001
    beta = 0.001
    if args.miniimagenet:
        nhop = 1
    else:
        nhop = 3

    # optimizer
    optimizer_alpha = torch.optim.SGD([
        {'params': model.layers.parameters(), 'lr': alpha},
    ], momentum=0.9, weight_decay=5e-4)

    optimizer_beta = torch.optim.SGD([
        {'params': model.layers.parameters(), 'lr': beta},
    ], momentum=0.9, weight_decay=5e-4)

    optimizer_shared_tmp = torch.optim.SGD([
        {'params': model_tmp.layers.parameters(), 'lr': alpha},
    ], momentum=0.9, weight_decay=5e-4)

    optimizer_decoders_tmp = torch.optim.SGD([
        {'params': model_tmp.decoder.parameters(), 'lr': alpha}
    ],momentum=0.9, weight_decay=5e-4)

    # loss
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(path) and args.use_checkpoint:
        print("############## load weights!!!! ##############")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initepoch = checkpoint['epoch']

    # start
    for epoch in range(initepoch, 99999):
        timestart = time.time()
        global patient
        if patient <= 0:
            print('early stop at epoch ', epoch)
            break
        running_loss = 0.0
        total = 0
        correct = 0

        for i, data in enumerate(trainloader):
            task = random.sample(range(n_tasks), 1)[0]

            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            for l in range(len(labels)):
                labels[l] = labels[l].to(device)

            model_tmp.load_state_dict(model.state_dict())
            model_tmp.train()

            # meta-train
            for hop in range(nhop):
                outputs = model_tmp(inputs)
                optimizer_shared_tmp.zero_grad()
                optimizer_decoders_tmp.zero_grad()
                loss1 = criterion(outputs[task], labels[task].long())
                loss1.backward()
                nn.utils.clip_grad_norm_(model_tmp.parameters(), 5)

                if hop == 0:
                    out_grad_layer1 = {name: param.grad for (name, param) in model_tmp.layers.named_parameters() if
                                           param.requires_grad}
                optimizer_shared_tmp.step()  # fast_weight<- shared layers
                optimizer_decoders_tmp.step()  # update decoder, then model_tmp decoder assign -> model

            # meta-test on main task
            model_tmp.eval()
            optimizer_shared_tmp.zero_grad()

            sampleloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
            sampleloader = enumerate(sampleloader)
            (inputs_other, labels_other) = next(sampleloader)[1]
            inputs_other = inputs_other.to(device)

            for l_o in range(len(labels_other)):
                labels_other[l_o] = labels_other[l_o].to(device)

            inner_task = 0
            outputs2 = model_tmp(inputs_other)
            loss2 = criterion(outputs2[inner_task], labels_other[inner_task].long())
            loss2.backward()
            nn.utils.clip_grad_norm_(model_tmp.parameters(), 5)

            # update grads
            hooks = []
            meta_grads = {name: out_grad_layer1[name] for (name, param) in
                          model_tmp.layers.named_parameters() if param.requires_grad}
            for name, param in model.layers.named_parameters():
                if param.requires_grad:
                    hooks.append(param.register_hook(replace_grad(meta_grads, name)))

            model.train()
            outputs2 = model(inputs_other)
            loss2 = criterion(outputs2[inner_task], labels_other[inner_task].long())
            loss2.backward(retain_graph=True)
            # Here the data (forwad, loss) doesn't matter at all, as the gradient will be replaced with meta_grads when "loss.backward()" is called.
            optimizer_alpha.step()
            optimizer_alpha.zero_grad()

            for hook in hooks:
                hook.remove()

            # update meta_grads
            hooks = []
            meta_grads = {name: param.grad for (name, param) in
                          model_tmp.layers.named_parameters() if param.requires_grad}
            for name, param in model.layers.named_parameters():
                if param.requires_grad:
                    hooks.append(param.register_hook(replace_grad(meta_grads, name)))

            loss2 = criterion(outputs2[inner_task], labels_other[inner_task].long())
            loss2.backward()
            # Here the data (forwad, loss) doesn't matter at all, as the gradient will be replaced with meta_grads when "loss.backward()" is called.
            optimizer_beta.step()

            optimizer_beta.zero_grad()
            for hook in hooks:
                hook.remove()

            # update decoder
            model.decoder.load_state_dict(model_tmp.decoder.state_dict())

            running_loss += loss2.item()

            if i % 500 == 499:  # print every 500 mini-batches
                print('[%d, %5d] loss: %.4f' %
                      (epoch, i, running_loss / 500))
                running_loss = 0.0
                _, predicted = torch.max(outputs2[0].data, 1)
                # print('predicted:', predicted)
                total += labels_other[0].size(0)
                correct += (predicted == labels_other[0]).sum().item()
                print('Accuracy of the network on the %d train images: %.3f %%' % (total,
                                                                                   100.0 * correct / total))
                total = 0
                correct = 0

        print('epoch %d cost %3f sec' % (epoch, time.time() - timestart))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs[0].data, 1)
                # print('predicted:', predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = float(correct / total)
        if acc >= max_acc:
            patient = 40
            max_acc = acc
            if args.save:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, path)
        else:
            patient -= 1
        print('now acc:', acc, 'best acc:', max_acc, 'patient:', patient)


"""
def test(model, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %.3f %%' % (
            100.0 * correct / total))
"""


def main(args):
    print('Python %s on %s' % (sys.version, sys.platform))
    print('Torch %s' % torch.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.miniimagenet:
        model = MiniNet(class_num, args, n_tasks).to(device)  # 4 cnn + 1 layer decoder
    else:
        model = CifarNet(class_num, args, n_tasks).to(device)  # 2 cnn + 1 layer decoder
       
    if args.train:
        train_sgd(model, device)


if __name__ == '__main__':
    # seed
    SEED = 20
    if SEED:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(SEED)

    patient = 40
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--extra_classes_num', default=10, type=int)
    parser.add_argument('--unequal_classes_num', action='store_true')
    parser.add_argument('--num_tasks', default=1, type=int)
    parser.add_argument('--cifar100', action='store_true')
    parser.add_argument('--cifar10', action='store_true')
    parser.add_argument('--miniimagenet', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--fine_labels', action='store_true')  # only valid when using cifar100
    parser.add_argument('--random_kmeans', action='store_true')
    args = parser.parse_args()

    n_tasks = args.num_tasks

    # miniimagenet
    transform_mini_train = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_mini_test = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # cifar100
    transform_cifar100_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_cifar100_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # cifar10
    transform_cifar10_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_cifar10_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # combine aux labels
    if args.miniimagenet:
        trainset = torchvision.datasets.ImageFolder('/home/lzqing/mlp/Data/data/mini-imagenet/train', transform_mini_train)
        testset = torchvision.datasets.ImageFolder('/home/lzqing/mlp/Data/data/mini-imagenet/test', transform_mini_test)
        class_num = 100

    if args.cifar100:
        trainset = MyCIFAR100(root='./data', train=True, download=True, 
                              transform=transform_cifar100_train, fine_labels=args.fine_labels)
        testset = MyCIFAR100(root='./data', train=False, download=True, 
                             transform=transform_cifar100_test, fine_labels=args.fine_labels)
        if args.fine_labels:
            class_num = 100
        else:
            class_num = 20

    if args.cifar10:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_cifar10_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_cifar10_test)
        class_num = 10


    # aux tasks
    kmeans_labels = []
    if not args.random_kmeans:
        for n in range(1, n_tasks):
            if not args.unequal_classes_num:
                if args.miniimagenet:
                    floader = "./aux_tasks/miniimagenet_100_cluster/half_dimen_100_cluster" if args.half else "./aux_tasks/miniimagenet_100_cluster/%d_cluster" % (args.extra_classes_num)
                    print("################## miniimagenet!!!! #####################")
                    kmeans = joblib.load('%s/miniimagenet_20_task_%dcluster_seed20_%d.pkl' % (floader, args.extra_classes_num, n))

                if args.cifar100:
                    kmeans = joblib.load("./aux_tasks/cifar100_100_cluster/cifar100_32_task_%dcluster_seed20_%d.pkl" % (args.extra_classes_num, n))
                    print("################## cifar100!!!! #####################")

                if args.cifar10:
                    kmeans = joblib.load("./aux_tasks/cifar10_15_cluster/cifar10_20_task_%dcluster_seed20_%d.pkl" % (args.extra_classes_num, n))
                    print("################## cifar10!!!! #####################")

            else:
                if args.cifar10:
                    print("################## cifar10!!!! ###################")
                    kmeans = joblib.load('kmeans/scale_kmeans/%d_cluster/cifar10_20_task_%dcluster_seed20_%d.pkl' % (
                    2 if n == 1 else 5 * (n - 1), 2 if n == 1 else 5 * (n - 1), 1))
                elif args.cifar100:
                    print("################## cifar100!!!! ###################")
                    kmeans = joblib.load(
                        'kmeans/100_scale_kmeans/%d_cluster/cifar100_32_task_%dcluster_seed20_%d.pkl' % (
                            5 * n, 5 * n, 1))
                else:
                    print("################## miniimagenet!!!! ###################")
                    floader = "../deep_miniimagenet_half" if args.half else "../mini_kmeans/scale_kmeans/%d_cluster/" % (
                                100 + 10 * n)
                    kmeans = joblib.load('%s/miniimagenet_20_task_%dcluster_seed20_%d.pkl' % (floader, 100 + 10 * n, 1))

            l_k = kmeans.labels_.tolist()
            print(len(l_k))
            kmeans_labels.append(kmeans.labels_)
    else:
        print("################### random label!!!! ######################")
        # combine n_task-1 random labels
        for n in range(1, n_tasks):
            if not args.unequal_classes_num:
                print("##################### aux decoder classes num fix!!!! #####################")
                kmeans_labels.append(np.random.randint(0, args.extra_classes_num, len(trainset.train_data)))
            else:
                print("##################### aux decoder grow!!!! #####################")
                if args.cifar10:
                    print("###################### cifar10!!!! ####################")
                    kmeans_labels.append(np.random.randint(0, 2 if n == 1 else 5 * (n - 1), len(trainset.train_data)))
                elif args.miniimagenet:
                    print("##################### miniimagenet!!!! #####################")
                    kmeans_labels.append(np.random.randint(0, 10 * n, len(trainset.train_data)))
                else:
                    print("#################### cifar100!!!! ######################")
                    kmeans_labels.append(np.random.randint(0, 5 * n, len(trainset.targets)))

    trainset = MyCIFAR(trainset, kmeans_labels)

    # load train set and test set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False)
    main(args)
