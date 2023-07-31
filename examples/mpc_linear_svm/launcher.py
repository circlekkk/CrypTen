#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
To run mpc_linear_svm example in multiprocess mode:

$ python3 examples/mpc_linear_svm/launcher.py --multiprocess

To run mpc_linear_svm example on AWS EC2 instances:

$ python3 scripts/aws_launcher.py \
      --ssh_key_file=$HOME/.aws/fair-$USER.pem \
      --instances=i-038dd14b9383b9d79,i-08f057b9c03d4a916 \
      --aux_files=examples/mpc_linear_svm/mpc_linear_svm.py \
      examples/mpc_linear_svm/launcher.py
"""

import argparse
import logging
import os

from examples.multiprocess_launcher import MultiProcessLauncher

# 定义了一个命令行参数解析器，并设置了一些参数的默认值和帮助信息
parser = argparse.ArgumentParser(description="CrypTen Linear SVM Training")
# 创建一个argparse.ArgumentParser对象parser，并设置了一个描述信息为"CrypTen Linear SVM Training"。
# 使用add_argument方法添加了一系列命令行参数。每个参数都有一个名字、类型、默认值、帮助信息等属性。
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)
#--world_size：参与方个数，表示要启动的进程数，每个参与方执行自己的进程。默认为2。
parser.add_argument(
    "--epochs", default=50, type=int, metavar="N", help="number of total epochs to run"
)
# --epochs：表示要运行的总的训练轮数，默认为50。metavar="N"用于指定在帮助信息中显示的参数值的格式
parser.add_argument(
    "--examples", default=50, type=int, metavar="N", help="number of examples per epoch"
)
# --examples：表示每轮训练中的样本数量，默认为50。
parser.add_argument(
    "--features",
    default=100,
    type=int,
    metavar="N",
    help="number of features per example",
)
# --features：表示每个样本的特征数量，默认为100。
parser.add_argument(
    "--lr", "--learning-rate", default=0.5, type=float, help="initial learning rate"
)
# --lr 或 --learning-rate：表示初始学习率，默认为0.5。
parser.add_argument(
    "--skip_plaintext",
    default=False,
    action="store_true",
    help="skip evaluation for plaintext svm",
)
# skip_plaintext：表示是否跳过明文 SVM 的评估，默认为False。
parser.add_argument(
    "--multiprocess",
    default=False,
    action="store_true",
    help="Run example in multiprocess mode",
)
# --multiprocess：表示是否以多进程模式运行，默认为False。

def _run_experiment(args):
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    )
    from mpc_linear_svm import run_mpc_linear_svm

    run_mpc_linear_svm(
        args.epochs, args.examples, args.features, args.lr, args.skip_plaintext
    )


def main(run_experiment):
    args = parser.parse_args()
    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main(_run_experiment)
