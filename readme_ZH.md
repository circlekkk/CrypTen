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
2.3 run mpc_cifar
python3 examples/mpc_cifar/launcher.py --epochs 3 --print-freq 1 --skip-plaintext --multiprocess
 这里有训练集\测试集数据加载器,保存模型的检查点
*临时目录保存了什么,为什么要最后清空
*保存检查点文件和保存模型一样吗
    保存模型通常是指将模型的参数（权重和偏置）保存到文件中，以便在以后的使用中加载和恢复模型。
    保存模型通常使用的函数是torch.save(model.state_dict(), filename)。

    保存检查点文件是在训练过程中定期保存模型的参数和优化器的状态，以便在训练中断或意外中止后能够从中断的地方继续训练。检查点文件除了模型的参数外，还包含了优化器的参数、当前的训练轮数、损失值等信息。
    保存检查点文件通常使用的函数是torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, filename)
LeNET()是isinstance(model, crypten.nn.Module)吗
    不是
3.现有的算法
    1.linear_svm->mpc_linear_svm,超参：epochs,lr
    2.cnn->mpc_autograd_cnn
    3.LeNet->mpc_cifar,手写数字识别
    4.mpc计算机视觉模型->mpc_imagenet,图像分类
    5.基准测试->tfe_benchmarks
4.crypten使用教程
初始化CrypTen库
    crypten.init()初始化CrypTen库，用于加密计算。初始化加密运算库：crypten使用基于PyTorch的加密运算库来执行加密计算。初始化随机数生成器
加解密
    x = crypten.cryptensor(x)
    b.get_plain_text()
算子
    a.matmul(b)
    .mul()
    a.add(b)
检查a是否是加密的张量
    crypten.is_encrypted_tensor(a)
获取进程排名:
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
pytorch模型转为crypten模型
    crypten.nn.from_pytorch(model_upd, dummy_input).encrypt(src=0)
    model_upd是在明文训练过程中更新的PyTorch模型，而dummy_input是一个具有适当形状的示例输入张量，用于确定模型的输入大小
    encrypt(src=0)是Crypten库中的一个方法，用于将模型加密。这个方法接受一个参数src，指定加密的源。
    在多方设置中，src指定发送加密模型的进程的排名。在这里，src=0表示进程0将加密模型发送给其他进程。
模型保存:
    torch.save(state, filename)
训练模式评估模式
model.train()#将模型设置为训练模式
model.eval()
5.to do
    1.先求交集，尝试CSV入参转为crypten tensor
    2.pytorch模拟多方
    3.模型保存,推理
    4.模型评估指标少
