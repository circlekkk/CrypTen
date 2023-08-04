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
                data_dirname, train=True, download=False, transform=transform
            )
            testset = datasets.CIFAR10(
                data_dirname, train=False, download=False, transform=transform
            )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=4, shuffle=True, num_workers=2
        )#创建训练集数据加载器
        # 将数据集封装成一个迭代器，以便在训练模型时进行批量的数据读取和处理
        # batch_size：每个批次的样本数量
        # num_workers：用于数据加载的子进程数，默认为2。可以加快数据加载的速度，特别是当数据集较大时。
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        return trainloader, testloader

    if context_manager is None:
        context_manager = NoopContextManager()
    # 如果context_manager参数为空，则使用NoopContextManager()作为默认的上下文管理器。他不做任何事情
    data_dir = tempfile.TemporaryDirectory()
    logging.info('#####data_dir######',data_dir)
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
    # 对比明文模型和密文模型的前1000条输出,用不到
    # 验证数据加载器val_loader、明文模型plaintext_model和私有模型private_model作为参数
    # switch to evaluate mode
    plaintext_model.eval()
    private_model.eval()
    # 将明文模型和私有模型切换到评估模式，这意味着模型的参数不会被更新。
    with torch.no_grad():
        # 用于在验证过程中禁用梯度计算，以减少内存消耗
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
    # 验证数据加载器val_loader和批次大小batch_size作为参数
    input, target = next(iter(val_loader))
    # 获取验证数据集中的一个批次，然后返回输入张量的大小
    logging.info('#####val_loader######',val_loader)
    logging.info('#####input.size()######', input.size())
    return input.size()


def construct_private_model(input_size, model):
    """Encrypt and validate trained model for multi-party setting."""
    # 接受输入大小input_size和模型model作为参数。该函数用于在多方设置中加密和验证训练好的模型
    # 模型加密
    # get rank of current process
    rank = comm.get().get_rank()
    # 获取当前进程的排名
    dummy_input = torch.empty(input_size)
    # 创建一个空的张量dummy_input，大小与输入大小相同

    # party 0 always gets the actual model; remaining parties get dummy model
    # 在多方设置中，第0方（即排名为0的进程）始终获得实际模型，而其他方则获得一个虚假模型。
    # 因此，如果当前进程的排名为0，则model_upd等于实际模型；否则，model_upd等于一个新的LeNet模型。
    # 其他方只能使用加密模型进行计算，但无法获取模型的明文表示
    if rank == 0:
        model_upd = model
    else:
        model_upd = LeNet()#必要性?没必要
    # 这行代码是进程为0的一方负责执行的
    logging.info('#####rank######', rank)
    private_model = crypten.nn.from_pytorch(model_upd, dummy_input).encrypt(src=0)
    logging.info('#####private_model######', private_model)
    # 将model_upd转换为Crypten模型,并使用dummy_input加密,加密的源方为第0方
    return private_model


def encrypt_data_tensor_with_src(input):
    """Encrypt data tensor for multi-party setting"""
    # 在多方计算场景中对数据张量进行加密
    # get rank of current process
    rank = comm.get().get_rank()
    # get world size
    world_size = comm.get().get_world_size()
    logging.info('#####world_size######', world_size)
    # 获取当前通信域中的进程数量

    if world_size > 1:
        # party 1 gets the actual tensor; remaining parties get dummy tensor
        # 将标识符src_id设置为1，表示进程1将获得实际的数据张量
        src_id = 1
    else:
        # party 0 gets the actual tensor since world size is 1
        src_id = 0

    if rank == src_id:
        input_upd = input
    #     其他参与方获得输入
    else:
        input_upd = torch.empty(input.size())
    #
    private_input = crypten.cryptensor(input_upd, src=src_id)
    logging.info('#####private_input######', private_input)
    # 将PyTorch张量 input_upd 加密为Crypten张量,其他参与方接收
    # 如果 world_size=1,进程为0的一方加密
    #如果 world_size>1,进程为1的一方加密
    return private_input


def validate(val_loader, model, criterion, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # 用于记录批次时间、损失和精度。
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if isinstance(model, crypten.nn.Module) and not crypten.is_encrypted_tensor(
                input
            ):
                input = encrypt_data_tensor_with_src(input)
            #     如果模型是crypten.nn.Module的实例，并且输入数据不是加密张量（crypten.is_encrypted_tensor(input)为False），则对输入数据进行加密
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
# top-1准确度

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """Saves checkpoint of plaintext model"""
    """
    state example:
    {
                "epoch": epoch + 1,
                "arch": "LeNet",
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
    """
    # only save from rank 0 process to avoid race condition
    # state表示当前的模型状态，is_best表示当前模型是否是最佳模型，filename表示保存检查点文件的路径和名称，默认为"checkpoint.pth.tar"。
    rank = comm.get().get_rank()
    if rank == 0:
        torch.save(state, filename)
        # 如果排名为0，则表示是主进程，可以保存检查点文件
        # 将对象state序列化为二进制格式，并将其写入指定的文件
        if is_best:
            shutil.copyfile(filename, "model_best.pth.tar")
#                             如果is_best为True，则将检查点文件复制为"model_best.pth.tar"，表示这是到目前为止的最佳模型。


def adjust_learning_rate(optimizer, epoch, lr=0.01):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #在训练过程中调整学习率
    # optimizer表示优化器对象，epoch表示当前的训练轮数，lr表示初始学习率，默认为0.01
    new_lr = lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # 计算模型在给定输出和目标值下的准确率
    # output表示模型的输出，target表示目标值，topk表示计算的准确率的前k个值，默认为1。
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        # 使用topk函数，找出输出output中前maxk个最大值的索引，并将索引保存到pred中。注意到_表示丢弃的值，这里没有使用。
        pred = pred.t()
        # 将pred进行转置，使得它的每一列包含了一个样本的预测结果
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # 将目标值target进行扩展，使其大小与pred相同，并与pred进行逐元素比较，得到一个布尔值矩阵correct，表示预测结果是否与目标值相等。

        res = []
        # res，用于保存计算得到的准确率
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            # 取correct的前k行，将其展平为一维向量，并将其转换为浮点型。然后对该向量进行求和操作，并保持维度为1，得到该k值下的正确预测数量。
            res.append(correct_k.mul_(100.0 / batch_size))
        #     将该k值下的正确预测数量乘以100除以批次大小，得到该k值下的准确率，并将其添加到res列表中
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
    #     定义网络的结构

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
if __name__ == "__main__":
    testmodel=LeNet()
    print('LeNet() type:',type(testmodel))
    print('isinstance',isinstance(testmodel, crypten.nn.Module))