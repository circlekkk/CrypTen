#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import shutil
import tempfile
import time

import crypten
import crypten.communicator as comm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from examples.meters import AverageMeter
from examples.util import NoopContextManager
from torchvision import datasets, transforms

# 基于 MPC的 CIFAR-10 图像分类任务
def run_mpc_cifar(
    epochs=25,
    start_epoch=0,
    batch_size=1,
    lr=0.001,
    momentum=0.9,
    weight_decay=1e-6,
    print_freq=10,
    model_location="",
    resume=False,
    evaluate=True,
    seed=None,
    skip_plaintext=False,
    context_manager=None,
):
    if seed is not None:
        random.seed(seed)
        # 用于设置random 模块的随机数种子
        torch.manual_seed(seed)
    #     用于设置 PyTorch 库的随机数种子

    crypten.init()

    # create model
    model = LeNet()#创建一个 LeNet 模型的实例,手写数字识别,卷积神经网络

    criterion = nn.CrossEntropyLoss()#定义交叉熵损失函数
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )#定义随机梯度下降（SGD）优化器

    # optionally resume from a checkpoint
    best_prec1 = 0#初始化最好的准确率为0
    if resume:
        # 如果resume参数为True，则尝试从检查点文件中恢复模型和优化器。
        if os.path.isfile(model_location):
            # 检查给定的模型文件路径是否存在并且是一个文件
            logging.info("=> loading checkpoint '{}'".format(model_location))
            checkpoint = torch.load(model_location)
            start_epoch = checkpoint["epoch"]
            # 从模型检查点字典中获取保存的训练周期（epoch）数
            best_prec1 = checkpoint["best_prec1"]
            # 从模型检查点字典中获取保存的最佳准确率
            model.load_state_dict(checkpoint["state_dict"])
            # 将保存的模型参数加载到当前模型中。load_state_dict()函数用于加载模型的状态字典。
            optimizer.load_state_dict(checkpoint["optimizer"])
            # 将保存的优化器参数加载到当前优化器中。
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    model_location, checkpoint["epoch"]
                )
            )
        #     指示模型检查点成功加载。这条信息包含模型文件路径和加载的训练周期数。
        else:
            raise IOError("=> no checkpoint found at '{}'".format(model_location))
    #     如果模型文件不存在，则抛出一个IOError异常，并打印出不存在的模型文件路径。

    # Data loading code
    def preprocess_data(context_manager, data_dirname):
        # 对数据进行预处理
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        with context_manager:
            trainset = datasets.CIFAR10(
                data_dirname, train=True, download=True, transform=transform
            )
            testset = datasets.CIFAR10(
                data_dirname, train=False, download=True, transform=transform
            )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=4, shuffle=True, num_workers=2
        )#创建训练集数据加载器
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        return trainloader, testloader

    if context_manager is None:
        context_manager = NoopContextManager()
    # 如果context_manager参数为空，则使用NoopContextManager()作为默认的上下文管理器。他不做任何事情
    data_dir = tempfile.TemporaryDirectory()
    # 创建一个临时目录来存储数据。
    train_loader, val_loader = preprocess_data(context_manager, data_dir.name)
    # 调用preprocess_data函数，加载并预处理数据。

    if evaluate:
        # 如果evaluate参数为True，则执行模型的评估
        if not skip_plaintext:
            logging.info("===== Evaluating plaintext LeNet network =====")
            validate(val_loader, model, criterion, print_freq)
        logging.info("===== Evaluating Private LeNet network =====")
        input_size = get_input_size(val_loader, batch_size)
        # 获取输入数据的大小
        private_model = construct_private_model(input_size, model)
        #  构建使用 MPC 的 Private LeNet 模型???
        # construct_private_model 函数的作用是将给定的模型转换为一个私有模型。
        # 私有模型是使用差分隐私技术进行保护的模型，可以在不暴露原始数据的情况下进行训练和推理。
        #私有模型中的权重会被添加差分隐私的噪声，以保护模型的隐私。
        validate(val_loader, private_model, criterion, print_freq)
        # 调用validate函数评估 Private LeNet 模型在验证集上的性能。准确率作为评估指标
        #它接受验证数据加载器val_loader，私有模型private_model，损失函数criterion和打印频率print_freq作为参数
        # logging.info("===== Validating side-by-side ======")
        # validate_side_by_side(val_loader, model, private_model)
        return

    # define loss function (criterion) and optimizer
    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch, lr)#调整学习率

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, print_freq)
        # 调用train函数在训练集上训练模型。
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, print_freq)
        # 调用validate函数评估模型在验证集上的性能。
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        # 更新最好的准确率
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": "LeNet",
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )
    #     保存模型的检查点
    data_dir.cleanup()#清理临时目录


def train(train_loader, model, criterion, optimizer, epoch, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()#top-1准确率
    top5 = AverageMeter()#top-5准确率

    # switch to train mode
    model.train()#将模型设置为训练模式

    end = time.time()#记录当前时间

    for i, (input, target) in enumerate(train_loader):
        # 遍历训练数据加载器，获取每个批次的输入和目标。
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.add(loss.item(), input.size(0))#将损失值添加到损失对象中，并记录样本的数量
        top1.add(prec1[0], input.size(0))
        top5.add(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()#将优化器的梯度置零
        loss.backward()#计算损失相对于模型参数的梯度
        optimizer.step()#根据梯度更新模型参数

        # measure elapsed time
        current_batch_time = time.time() - end
        batch_time.add(current_batch_time)
        end = time.time()

        if i % print_freq == 0:
            logging.info(
                "Epoch: [{}][{}/{}]\t"
                "Time {:.3f} ({:.3f})\t"
                "Loss {:.4f} ({:.4f})\t"
                "Prec@1 {:.3f} ({:.3f})\t"
                "Prec@5 {:.3f} ({:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    current_batch_time,
                    batch_time.value(),
                    loss.item(),
                    losses.value(),
                    prec1[0],
                    top1.value(),
                    prec5[0],
                    top5.value(),
                )
            )


def validate_side_by_side(val_loader, plaintext_model, private_model):
    """Validate the plaintext and private models side-by-side on each example"""
    # 在每个样本上同时验证明文模型和私有模型
    # switch to evaluate mode
    plaintext_model.eval()
    private_model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # compute output for plaintext
            output_plaintext = plaintext_model(input)
            # encrypt input and compute output for private
            # assumes that private model is encrypted with src=0
            input_encr = encrypt_data_tensor_with_src(input)
            output_encr = private_model(input_encr)
            # log all info
            logging.info("==============================")
            logging.info("Example %d\t target = %d" % (i, target))
            logging.info("Plaintext:\n%s" % output_plaintext)
            logging.info("Encrypted:\n%s\n" % output_encr.get_plain_text())
            # only use the first 1000 examples
            if i > 1000:
                break


def get_input_size(val_loader, batch_size):
    input, target = next(iter(val_loader))
    return input.size()


def construct_private_model(input_size, model):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    dummy_input = torch.empty(input_size)

    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = model
    else:
        model_upd = LeNet()
    private_model = crypten.nn.from_pytorch(model_upd, dummy_input).encrypt(src=0)
    return private_model


def encrypt_data_tensor_with_src(input):
    """Encrypt data tensor for multi-party setting"""
    # get rank of current process
    rank = comm.get().get_rank()
    # get world size
    world_size = comm.get().get_world_size()

    if world_size > 1:
        # party 1 gets the actual tensor; remaining parties get dummy tensor
        src_id = 1
    else:
        # party 0 gets the actual tensor since world size is 1
        src_id = 0

    if rank == src_id:
        input_upd = input
    else:
        input_upd = torch.empty(input.size())
    private_input = crypten.cryptensor(input_upd, src=src_id)
    return private_input


def validate(val_loader, model, criterion, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if isinstance(model, crypten.nn.Module) and not crypten.is_encrypted_tensor(
                input
            ):
                input = encrypt_data_tensor_with_src(input)
            # compute output
            output = model(input)
            if crypten.is_encrypted_tensor(output):
                output = output.get_plain_text()
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.add(loss.item(), input.size(0))
            top1.add(prec1[0], input.size(0))
            top5.add(prec5[0], input.size(0))

            # measure elapsed time
            current_batch_time = time.time() - end
            batch_time.add(current_batch_time)
            end = time.time()

            if (i + 1) % print_freq == 0:
                logging.info(
                    "\nTest: [{}/{}]\t"
                    "Time {:.3f} ({:.3f})\t"
                    "Loss {:.4f} ({:.4f})\t"
                    "Prec@1 {:.3f} ({:.3f})   \t"
                    "Prec@5 {:.3f} ({:.3f})".format(
                        i + 1,
                        len(val_loader),
                        current_batch_time,
                        batch_time.value(),
                        loss.item(),
                        losses.value(),
                        prec1[0],
                        top1.value(),
                        prec5[0],
                        top5.value(),
                    )
                )

        logging.info(
            " * Prec@1 {:.3f} Prec@5 {:.3f}".format(top1.value(), top5.value())
        )
    return top1.value()


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """Saves checkpoint of plaintext model"""
    # only save from rank 0 process to avoid race condition
    rank = comm.get().get_rank()
    if rank == 0:
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, "model_best.pth.tar")


def adjust_learning_rate(optimizer, epoch, lr=0.01):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class LeNet(nn.Sequential):
    """
    Adaptation of LeNet that uses ReLU activations
    """

    # network architecture:
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
