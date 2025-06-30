# import get_gpu_info as gpuinfo
# import GPUtil
# import torch
# import os


#获取gpu的信息方法一
# deviceinfo = gpuinfo.get_gpu_info()
# print(deviceinfo)

#获取gpu的信息方法二
# use_cuda = torch.cuda.is_available()
# GPUtil.getAvailable()
#
# if use_cuda:
#     print('__CUDNN VERSION:', torch.backends.cudnn.version())
#     print('__Number CUDA Devices:', torch.cuda.device_count())
#     print('__CUDA Device Name:',torch.cuda.get_device_name(0))
#     print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/nerf_pl
#%cd D:/Pytorch_thesis/nerf_pl


#第一步就是先输入原始照片进行渲染
#这个是原版，在python里跑不了，下面让ChatGPT帮我优化了一下。
# import os
# # set training configurations here
# os.environ['ROOT_DIR'] = "/content/drive/MyDrive/NeRF/cat"
#                          # directory containing the data
#
# #os.environ['ROOT_DIR'] = "D:/Pytorch_thesis/NeRF/cat"
# os.environ['IMG_W'] = "504" # image width (do not set too large)
# os.environ['IMG_H'] = "378" # image height (do not set too large)
# os.environ['NUM_EPOCHS'] = "30" # number of epochs to train (depending on how many images there are,
#                                 # 20~30 might be enough)
# os.environ['EXP'] = "cat" # name of the experience (arbitrary)
#
# python train.py \
#     --dataset_name llff \
#     --root_dir "$ROOT_DIR" \
#     --N_importance 64 --img_wh $IMG_W $IMG_H \
#     --num_epochs $NUM_EPOCHS --batch_size 1024 \
#     --optimizer adam --lr 5e-4 \
#    #--lr_scheduler cosine \
#     --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 \
#     --exp_name $EXP



# 这个是ChatGPT帮我写的，这个没有问题。不过跑之前要用colmap生成相应输入的位姿文件(poses_bounds.npy)
import subprocess


# Set the command-line arguments as a list
command = [
           # 原来的windows本地运行文件地址
           'python', 'D:\\Pytorch_thesis\\nerf_pl\\train.py',

           #autodl服务器运行文件地址
           # '/root/miniconda3/envs/thesis/bin/python', '/root/autodl-tmp2/Pytorch_thesis/nerf_pl/train.py',

           '--dataset_name', 'llff',

           #本地电脑的数据路径
           # '--root_dir', '/content/drive/MyDrive/NeRF/cat',  # replace with actual path to the data
           # '--root_dir', 'D:\\Python_Projects\\dataSetForNeRF\\renderpic_test_2',
           #现在用的本地的,#23.11.11这个是旧的了，下面选择了4个新的classes
           # '--root_dir', 'D:\\Python_Projects\\datasetforMVCNN\\ScanObjectNN_render\\bag',
           '--root_dir', 'D:\\Pytorch_thesis\\ScanObjectNN_render_selected\\toilet',
           #####

           #Autodl服务器的数据路径
           # '--root_dir', '/root/autodl-tmp2/Pytorch_thesis/ScanObjectNN_render_selected/sink',   #这里是pytorch_thesis 不同于第一个毕设代码，这里用的是原始的llff，没有添加类目，要手动添加类目
           ####
           '--N_importance', '32',  #rbg_fine里多添加的采样点（在coarse的基础上） #原来是64
           #'--img_wh', '504', '378', #所有nerf的原数据集的图片大小都是4032x3024，比率是0.75跟这里的378/504一样。现在用的数据集的图片大小都是256x256，这里修改一下。2023/6/21
           '--img_wh', '256', '256',
           '--num_epochs', '1', #原30
            #2023.11.2 加多一个gpu的条件，因为我现在租了两个gpu
           '--num_gpus', '1',
           '--batch_size', '32',
           '--optimizer', 'adam',
           '--lr', '5e-4',
           # '--lr','1e-2', #貌似不行，渲染出来没有东西
           '--lr_scheduler', 'steplr',
           #'--decay_step', '10', '20',  #渲染不出来 不知道是什么原因换一个decay_step
           '--decay_step', '2', '4', '8',
           '--decay_gamma', '0.5',
           #这里学习率类型lr_scheduler依照kwea123的代码，还可以选择'cosine',但是如果用了余弦退火类型，就不用decay_step跟gamma。
           #'--lr_scheduler', 'cosine',
           '--exp_name', 'sink']       #这里要更改哦 换成相应的images分类名称

#Execute the command
try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Command {e.cmd} returned non-zero exit status {e.returncode}")


#这一块是对checkpoint文件进行渲染的。

# """# Testing! (takes about 20~30 minutes)
#
# ### You can also download the pretrained model for `fern` [here](https://github.com/kwea123/nerf_pl/releases)
#
# ### The results are saved to `results/llff/$SCENE`
# """
#这是kwea123的代码

# os.environ['SCENE'] = 'fern'
# os.environ['CKPT_PATH'] = '/content/nerf_pl/ckpts/exp/epoch=29.ckpt'
#
# !python eval.py \
#    --root_dir "$ROOT_DIR" \
#    --dataset_name llff --scene_name $SCENE \
#    --img_wh $IMG_W $IMG_H --N_importance 64 --ckpt_path $CKPT_PATH






# 下面这里是Chatgpt写的（修改了一部分，很多删除了，改的跟上面gpt写的那个一样）
# Build command
# import subprocess
# command = [
#     'python',
#     'D:\\Python_Projects\\nerf_pl\\eval.py',
#     '--root_dir', 'D:\\Python_Projects\\dataSetForNeRF\\renderpic_test_2',
#     '--dataset_name', 'llff',
#     '--scene_name', 'renderpic_test_2',              #这个是生成gif文件的名称
#     '--img_wh', '256','256',
#     #'--img_wh', '504','378',  #这个新的Objectnn的size不是旧的size，上面是新的size
#     '--N_importance', '64',
#     '--ckpt_path', 'D:\\Python_Projects\\ckpts\\renderpic_test_2\\epoch=9.ckpt'
# ]
#
# # Run command
# subprocess.run(command, check= True)


#下面这个部分是MVCNN
#源代码：
#python train_mvcnn.py -name mvcnn -num_models 1000 -weight_decay 0.001 -num_views 12 -cnn_name vgg11

#这个什么multipleprocessing不知道有什么用
# import multiprocessing
#
# if __name__ == '__main__':
#     multiprocessing.freeze_support()

    # Your code that uses the multiprocessing module here
    #...






#学着chatgpt写子进程代码
# import subprocess
#
# data = 'D:\\Python_Projects\\datasetforMVCNN'
#
# command = [
#     'python',
#     'D:\\Python_Projects\\MVCNN-PyTorch\\controller.py',
#     data
#
# ]
# subprocess.run(command, check= True)



#出现了多进程问题，然而好像原来的代码或者是我的环境问题不支持多进程，下面用gpt的意见写的代码，看看行不行：


# import os
# import subprocess
#
# data = 'D:\\Python_Projects\\datasetforMVCNN'
#
# pid = os.fork()
#
# if pid == 0:
#     # Child process
#     command = ['python', 'D:\\Python_Projects\\MVCNN-PyTorch\\controller.py', data]
#     subprocess.run(command, check=True)
# else:
#     # Parent process
#     os.waitpid(pid, 0)
#不行！！！！！！！！！！！！！！！！！！！！！！！！！！！fork在linux才可以，windows没有。


#换这个windows的，gpt写的。
# import multiprocessing
# import subprocess
# import os
#
# data = 'D:\\Python_Projects\\datasetforMVCNN'
#
# def run_controller():
#     command = [
#         'python',
#         'D:\\Python_Projects\\MVCNN-PyTorch\\controller.py',
#         data
#     ]
#     subprocess.run(command, check=True)
#
# if __name__ == '__main__':
#     # Spawn a new process to run the controller
#     p = multiprocessing.Process(target=run_controller)
#     p.start()
#     p.join()

#放弃 上面这个都是Rberkiland的人的代码，换一个。换jongchyisu。

#下面这个没问题啊
# import subprocess
#
# command = [
#     'python',
#     'D:\\Python_Projects\\mvcnn_pytorch_fail\\mvcnn_pytorch\\train_mvcnn.py',
#     '-name', 'mvcnn',
#     '-num_models', '1000',
#     '-weight_decay', '0.001',
#     '-num_views', '12',
#     #'-cnn_name', 'vgg11' #原本是用vgg11
#     '-cnn_name','resnet50'
# ]

# Run the subprocess command
# subprocess.run(command, check=True)


#23.5.19注释完nerf代码，开始看mvcnn。
# 目前思路是nerf生成的120个渲染视图作为mvcnn的输入。去对比原来用polygrid物体的多视角视图作为输入有什么结果上的区别，但是这个很难，我用的输入不会有相应的polygrid。
# 所以就去对比用手机原图作为输入去比对用nerf生成的作为输入有什么区别。
# 那所以目前nerf的部分不用怎么动，如果后期考虑改进，如果输入方式要更换的话，那也是直接不用nerf了。



#2023.8.1
#instant_ngp
# import subprocess
#
# root_dir = "D:\\Users\\Administrator\\Desktop\\mifi_extreme"
# exp_name = "mifirabbit"
# batch_size = 64
# num_epoch = 30
# lr = 1e-2
# dataset_name = "colmap"  #没懂具体是干嘛的，但是有5种能选，'nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv'
#
# command = [
#     'python',
#     'D:\\Python_Projects\\ngp_pl\\train.py',
#     '--root_dir', root_dir,
#     '--exp_name', exp_name,
#     '--batch_size', str(batch_size),
#     '--num_epoch', str(num_epoch),
#     '--lr', str(lr),
#     '--dataset_name',  dataset_name
# ]
#
# # Run the subprocess command
# subprocess.run(command, check=True)


