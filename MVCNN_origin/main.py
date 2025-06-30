import subprocess

command = [
    'python',
    'D:\\Project_MVCNN_origin\\train_mvcnn.py',

    # remote host
    # '/root/miniconda3/envs/mythesis/bin/python','/root/autodl-tmp/Project_MVCNN_origin/train_mvcnn.py',
    ###
    '-name', 'mvcnn',
    '-num_models', '1000',
    '-weight_decay', '0.001',  # 原本是0.001  #24.1.10貌似0.0005效果不太好
    '-num_views', '2',
    # '-cnn_name', 'vgg11' #原本是用vgg11
    '-cnn_name', 'resnet18'  # 之前用的是这个 试一下18的
    # '-cnn_name','resnet18'
]

# Run the subprocess command
subprocess.run(command, check=True)