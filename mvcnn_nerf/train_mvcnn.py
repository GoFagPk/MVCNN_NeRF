import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN
# from nerf_pl.models.nerf import Embedding,NeRF


#23.12.6 分布式训练
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=8)# it will be *12 images in each batch for mvcnn , 原本是8
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000) #原本是1000 ， 但是没办法 看看能不能降一下显存要求
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5) #原来是5e-4  #
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
#原来的23.11.16
# parser.add_argument("-train_path", type=str, default="D:\\Python_Projects\\datasetforMVCNN\\ScanObjectNN_render\\*\\train")   #训练集
# parser.add_argument("-val_path", type=str, default="D:\\Python_Projects\\datasetforMVCNN\\ScanObjectNN_render\\*\\test")   #测试、验证集、预测及？

#local host 23.12.1
parser.add_argument("-train_path", type=str, default="D:\\Pytorch_thesis\\ScanObjectNN_render_selected_renamed\\*\\train")   #训练集
parser.add_argument("-val_path", type=str, default="D:\\Pytorch_thesis\\ScanObjectNN_render_selected_renamed\\*\\test")   #测试、验证集、预测及？

#remote host 23.12.1
# parser.add_argument("-train_path", type=str, default="/root/autodl-tmp/ScanObjectNN_render_selected/*/train")   #训练集
# parser.add_argument("-val_path", type=str, default="/root/autodl-tmp/ScanObjectNN_render_selected/*/test")

parser.set_defaults(train=False)

def create_folder(log_dir):   #创建log文件夹
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':
    args = parser.parse_args()

    pretraining = not args.no_pretraining
    log_dir = args.name  # 创造一个叫mvcnn的log文件夹
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    # STAGE 1
    log_dir = args.name+'_stage_1'   #所以这个stage 1 就是从这个物体的视图dataset里面提取各个视图的特征？
    create_folder(log_dir)
    cnet = SVCNN(args.name, nclasses=4, pretraining=pretraining, cnn_name=args.cnn_name)  #这里nclasses具体指什么？我现在也没有在用原来modelnet40的nclasses了？这里需要修改吗？而且注意这里用的是SVCNN做的cnet，不是MVCNN。难道也许就是说，按照论文的示意图，先是用SVCNN获取单独的视图，或许后面再提取单独视图的特征，再在viewpool合成特征？？023.6.18改了classes从40到4.
    optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)  #优化器上面weight_decay默认是0，权重衰减是一种正则化技术，在机器学习中用来降低模型的复杂性并防止过度拟合。它已被证明可以改善许多类型的机器学习模型的泛化性能，包括深度神经网络。
    n_models_train = args.num_models*args.num_views   #设置的是1000*12。 不过num_models是啥？

    train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)  #训练数据集，从tools里导的包
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0) #64太大了超出内存了，试一下32

    val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True)  #测试数据集，从tools里面导的包，用来测试从训练数据集里训练出来的模型是否适用于其他数据？
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)#64太大了超出内存了，试一下32
    print('num_train_files in svcnn: '+str(len(train_dataset.filepaths)))
    print('num_val_files in svcnn: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=1)  #注意这里最后的num_views只传了1，具体的还要看是为什么。
    # trainer.train(50)

    # STAGE 2
    log_dir = args.name+'_stage_2'   #所以第二个阶段就来到"mvcnn"了？
    create_folder(log_dir)

    cnet_2 = MVCNN(args.name, cnet, nclasses=4, cnn_name=args.cnn_name, num_views=args.num_views)  #注意这里修改了nclasses为4

    del cnet

    optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))   #betas？Adam里的betas参数- (Tuple[float, float], 可选) – 用于计算梯度运行平均值及其平方的系数（默认：0.9，    0.999）

    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views , is_train=True) #23.12.2 加多了个is_train来给MultiviewImgDataset判断是不是加载训练集
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer
    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views, is_train=False)#23.12.2 加多了个is_train来给MultiviewImgDataset判断是不是加载训练集
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
    print('num_train_files in mvcnn: '+str(len(train_dataset.filepaths)))
    print('num_val_files in mvcnn: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=args.num_views)  #这里cnet_2就是MVCNN了
    trainer.train(50)  #这里30是epoch了



