#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time

import crypten
import torch
from examples.meters import AverageMeter


def train_linear_svm(features, labels, epochs, lr, print_time=False):
    # 这段代码的主要功能是训练一个线性SVM模型，并返回训练后的权重和偏置。
    # 在训练过程中，它会计算每个训练周期的准确率，并打印出来。
    # 如果设置了print_time参数为True，它还会计算每个迭代的时间，并打印出来。
    # Initialize random weights初始化随机权重
    w = features.new(torch.randn(1, features.size(0)))
    # features.size(0)是特征的维度
    b = features.new(torch.randn(1))

    if print_time:
        pt_time = AverageMeter()
        # 创建一个AverageMeter对象pt_time，用于计算和记录每个迭代的时间。
        end = time.time()

    for epoch in range(epochs):
        # Forward
        label_predictions = w.matmul(features).add(b).sign()

        # Compute accuracy
        correct = label_predictions.mul(labels)
        # 计算正确预测的样本，将预测标签label_predictions与真实标签labels进行逐元素相乘。
        accuracy = correct.add(1).div(2).mean()
        # 计算准确率，将正确预测的样本加1后除以2，再取平均值。
        if crypten.is_encrypted_tensor(accuracy):
            # 检查accuracy是否是加密的张量。
            accuracy = accuracy.get_plain_text()

        # Print Accuracy once
        if crypten.communicator.get().get_rank() == 0:
            # 如果当前进程的排名为0，表示是主进程。
            print(
                f"Epoch {epoch} --- Training Accuracy %.2f%%" % (accuracy.item() * 100)
            )
        #     打印当前迭代的训练准确率

        # Backward
        loss_grad = -labels * (1 - correct) * 0.5  # Hinge loss
        # 计算损失函数的梯度，使用Hinge Loss函数计算
        b_grad = loss_grad.mean()
        # 计算偏置b的梯度，将损失函数的梯度取平均值。
        w_grad = loss_grad.matmul(features.t()).div(loss_grad.size(1))
        # 计算权重w的梯度，将损失函数的梯度与特征的转置矩阵相乘后再除以样本数量。

        # Update
        w -= w_grad * lr
        b -= b_grad * lr

        if print_time:
            iter_time = time.time() - end
            pt_time.add(iter_time)
            logging.info("    Time %.6f (%.6f)" % (iter_time, pt_time.value()))
            # 打印当前迭代的时间和平均时间
            end = time.time()

    return w, b


def evaluate_linear_svm(features, labels, w, b):
    """Compute accuracy on a test set"""
    predictions = w.matmul(features).add(b).sign()
    correct = predictions.mul(labels)
    accuracy = correct.add(1).div(2).mean().get_plain_text()
    if crypten.communicator.get().get_rank() == 0:
        print("Test accuracy %.2f%%" % (accuracy.item() * 100))


# def run_mpc_linear_svm(
#     epochs=50, examples=50, features=100, lr=0.5, skip_plaintext=False
# ):
def run_mpc_linear_svm(
        epochs, examples, features, lr, skip_plaintext=False
):
    # 设置x,y输入训练模型
    crypten.init()

    # Set random seed for reproducibility
    torch.manual_seed(1)#设置随机数种子，以确保实验的可重复性。

    # Initialize x, y, w, b
    x = torch.randn(features, examples)#生成一个大小为(features, examples)的张量
    w_true = torch.randn(1, features)#生成一个大小为(1, features)的张量w_true
    b_true = torch.randn(1)#生成一个大小为(1)的张量
    y = w_true.matmul(x) + b_true
    y = y.sign()#将y中的所有元素替换为它们的符号
    # 用于将数组中的元素转换为其符号（即正号、负号或零）。在这里，.sign()函数将标签向量中的正数转换为+1，负数转换为-1，0保持不变。
    # 这是为了将标签向量转换为二分类问题中的预测标签，其中+1表示正类，-1表示负类。

    if not skip_plaintext:
        logging.info("==================")
        logging.info("PyTorch Training")
        logging.info("==================")
        w_torch, b_torch = train_linear_svm(x, y, lr=lr,epochs=epochs, print_time=True)
        #明文训练，使用PyTorch训练线性SVM模型，并将得到的权重和偏置保存在w_torch和b_torch中。

    # Encrypt features / labels
    x = crypten.cryptensor(x)
    y = crypten.cryptensor(y)

    logging.info("==================")
    logging.info("CrypTen Training")
    logging.info("==================")
    w, b = train_linear_svm(x, y, epochs=epochs,lr=lr, print_time=True)
    # 密文训练
    if not skip_plaintext:
        logging.info("PyTorch Weights  :")
        logging.info(w_torch)
    logging.info("CrypTen Weights:")
    logging.info(w.get_plain_text())

    if not skip_plaintext:
        logging.info("PyTorch Bias  :")
        logging.info(b_torch)
    logging.info("CrypTen Bias:")
    logging.info(b.get_plain_text())
