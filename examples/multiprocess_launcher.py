#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
import os
import uuid

import crypten


class MultiProcessLauncher:

    # run_process_fn will be run in subprocesses.
    def __init__(self, world_size, run_process_fn, fn_args=None):
        env = os.environ.copy()
        # 创建一个环境变量的副本
        #
        env["WORLD_SIZE"] = str(world_size)
        # 将WORLD_SIZE环境变量设置为world_size的字符串表示。WORLD_SIZE是一个环境变量，它通常用于指定并行计算的进程数量
        multiprocessing.set_start_method("spawn")
        # 设置子进程的启动方法为spawn
        # fork：这是最常见的默认启动方法，它使用fork()系统调用来创建子进程的副本。在UNIX和Linux系统上可用，可以快速有效地创建子进程。但是，在某些特殊情况下，比如在使用多线程的程序中，使用fork方法可能会导致意想不到的问题。
        # spawn：这是一个跨平台的启动方法，它使用spawn系统调用来创建子进程。它创建一个全新的Python解释器进程，并在其中导入主模块，并执行指定的子进程函数。这种方法的好处是它可以避免fork方法可能带来的一些问题，尤其是在多线程程序中。spawn方法在所有平台上都可用，但它的启动速度相对较慢。
        # forkserver：这是另一种跨平台的启动方法,与spawn方法类似，启动速度更快。它适用于某些特殊情况，例如在使用大型数据集时

        # Use random file so multiple jobs can be run simultaneously
        INIT_METHOD = "file:///tmp/crypten-rendezvous-{}".format(uuid.uuid1())
        # 根据一个唯一的UUID生成一个临时文件路径,这个临时文件路径会被用作一种共享的“rendezvous”点，即进程之间的会合点。
        # 通过共享这个文件路径，不同的进程可以在同一个文件中相互发现和连接，从而进行通信和同步操作。
        # 这种方法可以确保不同的进程可以找到彼此，并共享必要的信息，以便进行协作和数据交换。
        # 使用随机文件名创建初始化方法。

        # uuid.uuid1()函数基于系统时间和计算机的MAC地址生成UUID。它的生成方式包括时间戳、时钟序列和节点（通常是MAC地址）的组合。由于时间戳是其中的一部分，所以生成的UUID有一定的时间顺序性。
        # UUID是由标准算法根据时间戳、计算机的MAC地址和随机数生成的128位数字，通常以字符串的形式表示。
        env["RENDEZVOUS"] = INIT_METHOD
        #  将RENDEZVOUS环境变量设置为初始化方法。
        # RENDEZVOUS环境变量通常用于指定子进程之间进行通信和同步的方法。INIT_METHOD的值通常是一个URL或路径

        self.processes = []
        for rank in range(world_size):
            process_name = "process " + str(rank)
            process = multiprocessing.Process(
                target=self.__class__._run_process,
                name=process_name,
                args=(rank, world_size, env, run_process_fn, fn_args),
            )
            # 使用for循环创建多个子进程，每个进程执行相同的_run_process方法。
            # _run_process方法接受多个参数，包括进程的排名、总进程数、环境变量、要运行的函数(run_process_fn)和函数的参数(fn_args)
            self.processes.append(process)

        if crypten.mpc.ttp_required():
            # 是否需要创建一个TTP（Trusted Third Party）进程
            ttp_process = multiprocessing.Process(
                target=self.__class__._run_process,
                name="TTP",
                args=(
                    world_size,
                    world_size,
                    env,
                    crypten.mpc.provider.TTPServer,
                    None,
                ),
            )
            # crypten.mpc.provider.TTPServer是Crypten中的一个类，用于实现安全多方计算（Secure Multiparty Computation，MPC）中的可信第三方（Trusted Third Party，TTP）功能。TTP进程通常用于帮助多个参与方进行安全计算
            # 它负责协调和管理参与方之间的通信、密钥交换和计算结果的验证等任务。
            self.processes.append(ttp_process)

    @classmethod
    # 类方法可以在不创建类实例的情况下被调用。
    def _run_process(cls, rank, world_size, env, run_process_fn, fn_args):
        for env_key, env_value in env.items():
            os.environ[env_key] = env_value
        os.environ["RANK"] = str(rank)
        orig_logging_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO)
        # crypten.init()
        logging.getLogger().setLevel(orig_logging_level)
        if fn_args is None:
            run_process_fn()
        else:
            run_process_fn(fn_args)

    def start(self):
        for process in self.processes:
            process.start()

    def join(self):
        for process in self.processes:
            process.join()
            assert (
                process.exitcode == 0
            ), f"{process.name} has non-zero exit code {process.exitcode}"
    # 遍历self.processes列表中的每个进程，并调用它们的join方法，以等待这些进程完成。在等待过程中，还使用断言语句检查每个进程的退出码是否为0（表示正常退出），如果不是0，则会抛出一个断言错误。
    def terminate(self):
        for process in self.processes:
            process.terminate()
if __name__ == "__main__":
    launcher = MultiProcessLauncher(1,None)
    launcher.start()
    launcher.join()
    launcher.terminate()