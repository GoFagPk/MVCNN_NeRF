import numpy as np
import glob
import torch.utils.data
import os
import math
from skimage import io, transform
from PIL import Image
import torch
import torchvision as vision
from torchvision import transforms, datasets
import random
import re

# import torch.nn.functional as F

class MultiviewImgDataset(torch.utils.data.Dataset):

    #23.12.29
    def sort_numerically(self, files):
        """
        Sorts the files list numerically based on the number in the filename.
        """

        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [atoi(c) for c in re.split(r'(\d+)', text)]

        files.sort(key=lambda file: natural_keys(file))
        return files

    #24.1.7新的 用于计算数据集的std跟mean，用于归一化
    def calculate_mean_std(self):
        # Initialize lists to store all features
        all_features = []

        # Iterate through all NeRF feature files and collect features
        for feature_path in glob.glob(os.path.join(self.nerf_feature_dir, '**/*.pt'), recursive=True):
            features = torch.load(feature_path)
            all_features.append(features)

        # Concatenate all features and calculate mean and std
        all_features_tensor = torch.cat(all_features, dim=0)
        mean = all_features_tensor.mean()
        std = all_features_tensor.std()

        return mean, std
    ###########


    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12, shuffle=True,is_train=False):

        #23.12.2 加多了个is_train来给MultiviewImgDataset判断是不是加载训练集
        # self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
        #                  'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
        #                  'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
        #                  'person','piano','plant','radio','range_hood','sink','sofa','stairs',
        #                  'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.classnames=['bag', 'pillow', 'sink','toilet'] #23.11.16自己准备的4个类目
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views

        # set_ = root_dir.split('/')[-1]  #这是原本的

        #local 23.12.1
        set_ = root_dir.split('\\')[-1]  # 就比如说train_path，就会取倒数第一个的文件夹名字：train。

        #remote host version 23.12.1
        # set_ = root_dir.split('/')[-1]

        # parent_dir = root_dir.rsplit('/',2)[0] #这个就是path的开头两个文件夹路径名字。这是原本的

        # local 23.12.1
        parent_dir = root_dir.rsplit('\\', 2)[0]

        # remote host version 23.12.1
        # parent_dir = root_dir.rsplit('/', 2)[0]

        self.filepaths = []

        #23.11.29
        self.nerf_features_paths = []

        #local 23.12.1
        # self.nerf_feature_dir = "D:\\Pytorch_thesis\\eighth_layer_output_reshaped" #128
        self.nerf_feature_dir = "D:\\Pytorch_thesis\\eighth_layer_output_reshaped_256"
        # self.nerf_feature_dir = "D:\\Pytorch_thesis\\eighth_layer_output_reshaped_512"

        # 24.1.7 用来计算数据集的std跟mean的
        # Calculate and store mean and std of NeRF features
        self.mean, self.std = self.calculate_mean_std()

        #remote host
        # self.nerf_feature_dir = "/root/autodl-tmp/eighth_layer_output_reshaped"
        #####

        for i in range(len(self.classnames)):
            #all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*shaded*.png')) #这个是原来的

            #local 23.12.1
            all_files = sorted(glob.glob(parent_dir + '\\' + self.classnames[i] + '\\' + set_ + '\\*.jpg')) #这理后缀记得要改，原本是png现在是jpg
            # print("the image files are:",all_files)
            # print("and the length of image files is:",len(all_files))
            #remote host 23.12.1
            # all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/*.jpg'))

            ## Select subset for different number of views
            stride = int(12/self.num_views) # 12 6 4 3 2 1
            all_files = all_files[::stride]

            #23.11.29
            # Load corresponding NeRF features
            nerf_files = sorted(glob.glob(os.path.join(self.nerf_feature_dir, self.classnames[i], '*.pt')))

            # 23.12.29
            # Sort the NeRF feature files numerically
            nerf_files = self.sort_numerically(nerf_files)

            # print("the nerf features files are:", nerf_files)
            # print("the length of the nerf files is :",len(nerf_files))
            #24.1.17nerf好像用不到stride
            # nerf_files = nerf_files[::stride]
            ##
            #这个行不通，会直接内存爆炸，四个classes features总共18gb。######23.11.30
            # nerf_features = [torch.load(f) for f in nerf_files]

            #23.11.29 原来的
            # if num_models == 0:
            #     # Use the whole dataset
            #     self.filepaths.extend(all_files)
            # else:
            #     self.filepaths.extend(all_files[:min(num_models,len(all_files))])

            # if shuffle == True:
            #     # permute 置换的意思
            #     rand_idx = np.random.permutation(int(len(self.filepaths) / num_views))
            #     filepaths_new = []
            #     for i in range(len(rand_idx)):
            #         filepaths_new.extend(self.filepaths[rand_idx[i] * num_views:(rand_idx[i] + 1) * num_views])
            #     self.filepaths = filepaths_new
            ##########################23.11.29



            #23.11.29 现在的
            if num_models == 0 : #for val_set
                self.filepaths.extend(all_files)
                self.nerf_features_paths.extend(nerf_files)
            else :
                self.filepaths.extend(all_files[:min(num_models * self.num_views, len(all_files))]) #这里是选取all_files从第一个到all files的总文件数目，num_models是1000，用的数据集没有比1000*num_views多的
                self.nerf_features_paths.extend(nerf_files[:min(num_models, len(nerf_files))]) #
                # print("this is filepaths in train_set:", len(self.filepaths))  #是在这个循环里依次叠加各个class的image，768,1788,2952,3732
                # print("this is nerf_file_paths in val_set:", len(self.nerf_features_paths)) #64,149,246,311

        if shuffle == True:
            if is_train:

                #23.12.14更新一下shuffle 之前的image无法跟object file对应上
                # # Shuffle for training dataset including NeRF features
                # print("now the trainloader is in ")
                # combined = list(zip(self.filepaths, self.nerf_features_paths))
                # np.random.shuffle(combined)
                # self.filepaths, self.nerf_features_paths = zip(*combined)
                # self.filepaths = list(self.filepaths)
                # self.nerf_features_paths = list(self.nerf_features_paths)
                # Group file paths by object


                #23.12.14 新的shuffle
                num_objects = len(self.filepaths) // self.num_views
                object_groups = [self.filepaths[i * self.num_views:(i + 1) * self.num_views] for i in
                                 range(num_objects)]
                # print("num_objects in shuffle is ",num_objects)
                # print("object_groups are:",object_groups)

                # Shuffle object groups
                rand_idx = np.random.permutation(len(object_groups))
                shuffled_object_groups = [object_groups[i] for i in rand_idx]
                # print("after shuffle the rand_idx of object_groups is:",rand_idx)
                # print("and the shuffled_object_groups are:",shuffled_object_groups)

                # Reconstruct the filepaths and nerf_features_paths
                self.filepaths = [path for group in shuffled_object_groups for path in group]
                self.nerf_features_paths = [self.nerf_features_paths[i] for i in rand_idx]
                # print("the number of filepaths is",len(self.filepaths),"and the number of nerf_features files is ",len(self.nerf_features_paths))
                # print("image filepaths after shuffled:",self.filepaths)
                # print("the correspond nerf features paths:",self.nerf_features_paths)
                #24.1.3 目前查看了filepaths跟nerf_features_paths的情况，都是没问题的，shuffled后能够对应上。例如filepaths开头是bag的74-84.jpg,而nerf第一个文件就是6.pt（即第7个bag物体的特征文件）

            else:
            # Shuffle for validation dataset (only filepaths) 23.12.2
                rand_idx = np.random.permutation(int(len(self.filepaths) / num_views))
                filepaths_new = []
                for i in range(len(rand_idx)):
                    filepaths_new.extend(self.filepaths[rand_idx[i] * num_views:(rand_idx[i] + 1) * num_views])
                self.filepaths = filepaths_new
                # print("the length is ",len(self.filepaths)) #864
                # print("the content is :",self.filepaths)
            ####################23.11.29

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])    
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return int(len(self.filepaths)/self.num_views)

    #24.1.5用来给nerf的tensor里的value做归一，不然太大了，基本都5000以上，img转rgb的value基本都是2-4
    # def normalize_features(self,features):
    #     min_val = features.min()
    #     max_val = features.max()
    #     return (features - min_val) / (max_val - min_val)

    #24.1.7 新的归一化，基于用的数据集计算的
    def normalize_features(self, features):
        # Normalize using the calculated mean and std
        normalized = (features - self.mean) / self.std
        return normalized
    #################



    def __getitem__(self, idx):
        # print("the idx now is ",idx)  #idx跟batch_size有关，目前bs是8，idx就每次从0-7,8-15,16-23这样跳
        path = self.filepaths[idx*self.num_views]
        # print("the path content is ", path) #path是选中的idx对应的物体的第一张view
        #class_name = path.split('/')[-3] #这个是原来的。

        #23.12.1 local host
        class_name = path.split('\\')[-3]

        #23.12.1 remote host
        # class_name = path.split('/')[-3]

        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
                # print("the imgs are:", im)
            imgs.append(im)


        #23.11.29
        # Load corresponding NeRF features
        if idx < len(self.nerf_features_paths):
            nerf_feature_path = self.nerf_features_paths[idx]
            nerf_feature_list = torch.load(nerf_feature_path)
            # Normalize NeRF features
            nerf_feature_list = self.normalize_features(nerf_feature_list)
            # print("the nerf_feature_list is:",nerf_feature_list.shape)
            # print("the nerf_feature_list is:",nerf_feature_list)#这个nerf_feature_path是在init里面加载好了路径，而且shuffle过了，能够对应上相应的filepath，都是具体的特征文件路径，上面又直接根据idx load了相应的文件，所以输出是具体的
        else:
            print(f"Index {idx} is out of range for NeRF features.")
        # Handle the error or raise an exception
            raise IndexError("Index out of range for NeRF features.")

        # Convert list of tensors to a single tensor
        # nerf_feature = torch.stack(nerf_feature_list) #23.12.7 报错说好像已经是tensors不用stack了

        # Reshape the NeRF features
        # nerf_feature_reshaped = nerf_feature.view(12, 24576, 32)
        #23.12.6 直接在外面把nerf features文件全部
        #23.12.4
        # Apply max pooling along the feature dimension
        # nerf_feature_pooled = F.max_pool1d(nerf_feature_reshaped.view(1, 12, -1), kernel_size=6144)
        # nerf_feature_reshaped = nerf_feature_pooled.view(12, 128)


        ##########

        #23.11.29原来的
        # return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])
        #####
        # print("the files using now are:", self.filepaths[idx * self.num_views:(idx + 1) * self.num_views])
        #23.11.29新的，相较原来多了个nerf_feature
        return class_id, torch.stack(imgs), nerf_feature_list, self.filepaths[idx * self.num_views:(idx + 1) * self.num_views]

class SingleImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12):
        #有没有可能是他这里写死了class呢，我下面自己打一下我自己准备的类目
        # self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
        #                  'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
        #                  'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
        #                  'person','piano','plant','radio','range_hood','sink','sofa','stairs',
        #                  'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.classnames = ['bag', 'pillow', 'sink','toilet']   #23.11.16自己准备的
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode

        ####
        # set_ = root_dir.split('/')[-1]  #这是原本的

        #local host 23.12.1
        set_ = root_dir.split('\\')[-1] #就比如说train_path，就会取倒数第一个的文件夹名字：train。

        #remote host 23.12.1
        # set_ = root_dir.split('/')[-1]

        ###
        # parent_dir = root_dir.rsplit('/',2)[0] #这个就是path的开头两个文件夹路径名字。这是原本的

        #local host 23.12.1
        parent_dir = root_dir.rsplit('\\', 2)[0]

        #remote host 23.12.1
        # parent_dir = root_dir.rsplit('/', 2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):

            ###
            #all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*shaded*.png')) #我草 这个是个shaded是个什么东西，我的图片名字只有数字，而原本它那个modelnet40的图片都带有shaded名字。

            #local host 23.12.1
            all_files = sorted(glob.glob(parent_dir + '\\' + self.classnames[i] + '\\' + set_ + '\\*.jpg')) #注意这里原本是png现在改成了jpg

            #remote host
            # all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/*.jpg'))
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        self.transform = transforms.Compose([
            # transforms.Resize((224, 224)),# gpt推荐的 统一输入的大小 2023.6.18暂时先注释掉看看
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):
        path = self.filepaths[idx]

        ###
        #class_name = path.split('/')[-3] #这是原本的，filepaths在上面的循环里被赋值了allpath，就是图片的全路径

        #local host 23.12.1
        class_name = path.split('\\')[-3]

        #remote host 23.12.1
        # class_name = path.split('/')[-3]

        class_id = self.classnames.index(class_name)

        # Use PIL instead
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)

        return (class_id, im, path)

