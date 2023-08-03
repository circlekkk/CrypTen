1.安装
    pip install crypten
    pip install -r requirements.examples.txt
2.运行例子
/root/PycharmProjects/CrypTen
只有rank为0的worker才会显示日志记录信息
2.1 run mpc_linear_svm
    python3 examples/mpc_linear_svm/launcher.py --multiprocess
    指定参数：
    python3 examples/mpc_linear_svm/launcher.py --world_size 2 --epochs 6  --examples 200 --features 5 --skip_plaintext --multiprocess
2.2 run mpc_imagenet
    python3 examples/mpc_imagenet/launcher.py  --tensorboard_folder /root/PycharmProjects/CrypTen/tensorboard_log/ --num_samples 50 --multiprocess
2.3
python3 examples/mpc_cifar/launcher.py \
    --evaluate \
    --resume path-to-model/model.pth.tar \
    --batch-size 1 \
    --print-freq 1 \
    --skip-plaintext \
    --multiprocess
 这里有训练集\测试集数据加载器,保存模型的检查点
3.现有的算法
    1.linear_svm->mpc_linear_svm,超参：epochs,lr
    2.cnn->mpc_autograd_cnn
    3.LeNet->mpc_cifar
    4.mpc计算机视觉模型->mpc_imagenet
    5.基准测试->tfe_benchmarks
4.crypten使用教程
crypten.init()初始化CrypTen库，用于加密计算。初始化加密运算库：crypten使用基于PyTorch的加密运算库来执行加密计算。初始化随机数生成器
x = crypten.cryptensor(x)
b.get_plain_text()
a.matmul(b)
.mul()
a.add(b)
crypten.is_encrypted_tensor(a)检查a是否是加密的张量
        if crypten.communicator.get().get_rank() == 0:
            # 如果当前进程的排名为0，表示是主进程。
            print(
                f"Epoch {epoch} --- Training Accuracy %.2f%%" % (accuracy.item() * 100)
            )
多进程：
    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size参与方个数, run_experiment要运行的函数, args函数参数)
        launcher.start()
        launcher.join()
        launcher.terminate()
5.to do
    1.先求交集，尝试CSV入参转为crypten tensor
    2.pytorch模拟多方
    3.模型保存,推理
    4.模型评估指标少
