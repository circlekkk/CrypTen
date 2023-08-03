#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import tempfile

import crypten
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from examples.meters import AccuracyMeter
from examples.util import NoopContextManager


try:
    from crypten.nn.tensorboard import SummaryWriter

    # 用于将训练过程中的指标和可视化数据写入TensorBoard日志文件。它提供了一种方便的方法来跟踪和可视化模型的训练进度，例如损失函数的变化、准确率的变化等。通过使用SummaryWriter类，可以将这些信息记录下来，并在TensorBoard中进行可视化分析，以帮助用户更好地理解模型的性能和训练过程。
except ImportError:  # tensorboard not installed
    SummaryWriter = None


def run_experiment(
    model_name,
    imagenet_folder=None,
    tensorboard_folder="/tmp",
    num_samples=None,
    context_manager=None,
):
    """Runs inference using specified vision model on specified dataset.
    @param model_name:模型名称
    @param imagenet_folder:图像数据集文件夹,
    @param tensorboard_folder:执行记录文件夹,
    @param num_samples:样本数量
    @param context_manager:上下文管理器
    """

    crypten.init()
    # check inputs:
    assert hasattr(models, model_name), (
        "torchvision does not provide %s model" % model_name
    )
    if imagenet_folder is None:
        imagenet_folder = tempfile.gettempdir()
        # 如果imagenet_folder为None，则将imagenet_folder设置为临时文件夹
        # gettempdir()函数会返回操作系统默认的临时文件夹路径,在Linux系统上，临时文件夹通常是/tmp
        download = True
    #     设置一个标志变量download，用于指示是否需要下载数据集。
    else:
        download = False
    #     如果imagenet_folder不为None，不需要下载数据集
    if context_manager is None:
        context_manager = NoopContextManager()
    #     如果context_manager为None，则将其设置为NoopContextManager的实例。这个实例什么都不做

    # load dataset and model:#这里加载的是明文pytorch模型,且是集中式
    with context_manager:
        model = getattr(models, model_name)(pretrained=True)
        # 根据模型名称获取对应的预训练模型，并设置pretrained参数为True。
        model.eval()
        # 用于将模型设置为评估模式,将模型的参数设置为不可训练状态,
        # 在评估模式下，模型的参数不会被梯度更新，从而保持模型的参数不变。以便进行推理或评估任务
        # dataset = datasets.ImageNet(imagenet_folder, split="val", download=download)
        dataset=datasets.CIFAR10(imagenet_folder,train=True,download=download)
    #     加载ImageNet数据集，将其存储在dataset变量中。该路径应包含用于训练和验证的图像文件夹
    # split：指定要加载的数据集的划分。默认为"val"，即加载验证集。其他可选的值包括"train"（加载训练集）和"test"（加载测试集）
    # download：一个布尔值，指示是否自动下载ImageNet数据集。默认为False，表示不自动下载。如果数据集尚未下载，将引发一个异常。

    # define appropriate transforms:
    # 定义数据预处理的转换操作，包括调整大小、中心裁剪、转换为张量、归一化等。
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # 输入的图像的每个通道的像素值减去均值，然后再除以标准差
    to_tensor_transform = transforms.ToTensor()#将图像转换为张量

    # encrypt model:
    dummy_input = to_tensor_transform(dataset[0][0])#将第一个样本的图像转换为张量
    dummy_input.unsqueeze_(0)#在张量的第一个维度上添加一个维度，将其变成一个batch
    encrypted_model = crypten.nn.from_pytorch(model, dummy_input=dummy_input)
    # 将PyTorch模型转换为Crypten模型。dummy_input：一个输入张量，用于指定模型的输入大小
    encrypted_model.encrypt()#对加密模型进行加密

    # show encrypted model in tensorboard:
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=tensorboard_folder)
        writer.add_graph(encrypted_model)
        # 将加密模型的图形显示在TensorBoard中
        writer.close()

    # loop over dataset:
    meter = AccuracyMeter()
    for idx, sample in enumerate(dataset):

        # preprocess sample:
        image, target = sample#将样本拆分为图像和目标。
        image = transform(image)
        image.unsqueeze_(0)#将图像添加一个维度，将其变成一个batch
        target = torch.tensor([target], dtype=torch.long)#将目标转换为张量

        # perform inference using encrypted model on encrypted sample:
        encrypted_image = crypten.cryptensor(image)
        encrypted_output = encrypted_model(encrypted_image)#使用加密模型对加密图像进行推理

        # measure accuracy of prediction
        output = encrypted_output.get_plain_text()
        meter.add(output, target)

        # progress:
        logging.info(
            "[sample %d of %d] Accuracy: %f" % (idx + 1, len(dataset), meter.value()[1])
        )
        # 打印当前样本的准确率
        if num_samples is not None and idx == num_samples - 1:
            break

    # print final accuracy:
    logging.info("Accuracy on all %d samples: %f" % (len(dataset), meter.value()[1]))
# 打印最终的准确率