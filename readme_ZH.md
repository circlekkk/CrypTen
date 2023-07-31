1.安装
pip install crypten
pip install -r requirements.examples.txt
2.运行例子
2.1 run mpc_linear_svm
python3 examples/mpc_linear_svm/launcher.py --multiprocess
3.现有的算法
1.linear_svm->mpc_linear_svm
2.cnn->mpc_autograd_cnn
3.LeNet->mpc_cifar
4.mpc图形化
5.基准测试->tfe_benchmarks