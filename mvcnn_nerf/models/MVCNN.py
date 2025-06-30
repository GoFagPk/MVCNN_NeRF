import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model
# from nerf_pl.models.nerf import NeRF,Embedding

mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(Model):
    '''
    以及这里的nclasses参数也要改，那我这里的nclasses是15，cnn也打算用resnet，上面改一下

    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11'):
    '''
    #这个是自己的
    def __init__(self, name, nclasses=4, pretraining=True, cnn_name='resnet18'):        #23.9.26 现在只用4个classes
        super(SVCNN, self).__init__(name)

        '''
        这里按照gpt说的 更改成自己的classes类的数目及内容,
        下面这个是原来的40个类目
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']    #这个地方不知道要不要改咯，弄成自己的classname。问了4.0，是需要需改的。
        #这个是自己的，不过这个也是旧的了，现在换了ScanObjectnn的数据集
        '''
        self.classnames = ['bag', 'pillow', 'sink','toilet'] #23.11.16
        self.nclasses = nclasses  #40？能干嘛？已经换成现在的7了。
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
              #  self.net.fc = nn.Linear(512,40)  #按照gpt说的，线性全连接层输出数要跟类的数目一样（fc层是整个网络的最后一层？）
                self.net.fc = nn.Linear(512,4)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
              #  self.net.fc = nn.Linear(512,40)  #作相应更改
                self.net.fc = nn.Linear(512,4)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048,4)  #原本是40改成4

        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
            
            # self.net_2._modules['6'] = nn.Linear(4096,40)  #原来的，输出是40
            self.net_2._modules['6'] = nn.Linear(4096,4)  # 新的SCANOBJECTNN数据集15个类目

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0],-1))


class MVCNN(Model):

    '''原来的
    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12):
        super(MVCNN, self).__init__(name)

    '''

    def __init__(self, name, model, nclasses=4, cnn_name='resnet18', num_views=12):    #现在只用4个classes
        super(MVCNN, self).__init__(name)
        '''
        ModelNet40的类目
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        '''

        self.classnames = ['bag', 'pillow', 'sink','toilet'] #23.11.16
        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        self.use_resnet = cnn_name.startswith('resnet')

        # Dimensionality reduction for NeRF features 23.12.3
        # self.nerf_feature_reduction = nn.Linear(32, 25088 // 32)  # Adjust 25088 based on your actual MVCNN feature size


        #23.12.26 下面这个是为了迎合原来mvcnn中的fc层的输入输出(2048,4)，在结合特征进入net2之前需要reshape，现在考虑更改fc层的输入，从而不使用下面的linear层
        # self.nerf_feature_reduction = nn.Linear(2176, 2048)  #resnet50是(2176, 2048),resnet18是(640,512)


        # self.additional_fc = nn.Linear(6293504, 2048)  # Additional fully connected layer for reduction#23.12.3
        ###########

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])  #就是把使用的resnet各层赋予net_1(全部层开头的中间的，除了结尾的fc)

            #23.12.26这里用来分类的fc应该就是沿用svcnn中属于resnet50的fc，是fc = nn.Linear(2048,4),但是为了配合加入128个nerf的特征 这里直接赋值改一下


            '''num_views 12'''
            # self.net_2 = model.net.fc #原来的
            # self.net_2 = nn.Linear(2176, 4) #这个是池化之前合并的方法用的fc，如果是resnet18就是(640,4)
            # self.net_2 = nn.Linear(640, 4) #resnet18,nerf128
            # self.net_2 = nn.Linear(2560, 4) #resnet50,nerf512
            # self.net_2 = nn.Linear(1024,4) #renset18,nerf512
            self.net_2 = nn.Linear(768, 4)  # renset18,nerf256
            # self.net_2 = nn.Linear(2304, 4) #resnet50, nerf256

            '''num_views 6'''
            # self.net_2 = nn.Linear(1536, 4) #resnet18,nerf512 #for combination after&before viewpooling
            # self.net_2 = nn.Linear(1024, 4) #resnet18,nerf256
            '''num_views 4'''
            # self.net_2 = nn.Linear(2048, 4) #resnet18,nerf512
            # self.net_2 = nn.Linear(1280, 4)  # resnet18,nerf256
            '''num_views 3'''
            # self.net_2 = nn.Linear(2560,4) #resnet18,nerf512
            # self.net_2 = nn.Linear(1536, 4)  # resnet18,nerf256
            '''num_views 2'''
            # self.net_2 = nn.Linear(3584, 4) #resnet18,nerf512
            # self.net_2 = nn.Linear(2048, 4)  # resnet18,nerf256
            '''num_views 1'''
            # self.net_2 = nn.Linear(6656, 4) #resnet18,nerf512
            # self.net_2 = nn.Linear(3584, 4)  # resnet18,nerf256
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

    # def forward(self, x):
    #     y = self.net_1(x)
    #     y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))#(8,12,512,7,7)  #7x7是features map，之后用tile变成同7x7shape
    #     return self.net_2(torch.max(y,1)[0].view(y.shape[0],-1))   #这里如果使用restnet，那这里的net2就是model.net.fc,然后self.net = models.resnet50.第二个阶段这里用resnet，那么net1跟net2都跟上面原来的resnet用相同的网络

    #23.12.2 旧的
    # def forward(self, x, nerf_features):
    #     # Extract features for each view using MVCNN's CNN (net_1)
    #     cnn_features = self.net_1(x)
    #     cnn_features = cnn_features.view(
    #         (int(x.shape[0] / self.num_views), self.num_views, -1))  # Reshape for view pooling
    #
    #     # Reshape NeRF features: Assuming nerf_features is a list of shape [294912, 32]
    #     # Split into 12 segments, each representing a view
    #     nerf_features_viewwise = nerf_features.view(-1, self.num_views, 32)
    #
    #     # Expand dimensions of NeRF features to match MVCNN features
    #     nerf_features_expanded = nerf_features_viewwise.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 12, 32, 1, 1]
    #
    #     # Concatenate NeRF features with MVCNN features along the feature dimension
    #     combined_features = torch.cat((cnn_features, nerf_features_expanded), dim=2)  # Adjust dim if needed
    #
    #     # View pooling
    #     pooled_features = torch.max(combined_features, 1)[0]
    #
    #     # Final fully connected layers
    #     output = self.net_2(pooled_features.view(pooled_features.shape[0], -1))
    #     return output

    # def forward(self, x, nerf_features):
    #     # Extract features for each view using MVCNN's CNN (net_1)
    #     cnn_features = self.net_1(x)
    #     cnn_features = cnn_features.view(
    #         (int(x.shape[0] / self.num_views), self.num_views, -1))  # Reshape for view pooling [8, 12, 2048]
    #
    #     print("Shape of cnn_features after CNN1:", cnn_features.shape)
    #
    #     # Process NeRF features: Assuming nerf_features is a list with total elements 294912 for 12 views
    #     # Apply dimensionality reduction to NeRF features
    #     nerf_features_reduced = [self.nerf_feature_reduction(feature.view(-1,32)) for feature in nerf_features]
    #     print("Shape of nerf_features after reduced:", nerf_features_reduced.shape)
    #     # 23.12.4
    #     nerf_features_reduced = torch.stack(nerf_features_reduced)  # Stack the list of reduced tensors
    #     print("Shape of nerf_features after stack:", nerf_features_reduced.shape)#23.12.4
    #     nerf_features_tensor = nerf_features_reduced.view(int(x.shape[0]/self.num_views), 12, -1)
    #     print("Shape of nerf_features after view operation:", nerf_features_tensor.shape)
    #     # nerf_features_tensor = torch.stack(nerf_features_reduced, dim=1)  # [batch_size, 12, 24576, 32] [12, 8, 24576, 784]
    #
    #     # print("Shape of nerf_features_tensor:", nerf_features_tensor.shape)
    #
    #     # Ensure the batch size and view count dimensions align with cnn_features #23.12.3
    #     # nerf_features_tensor = nerf_features_tensor.transpose(0, 1)  # Swap batch size and view count dimensions
    #
    #     # Reshape nerf_features_tensor #23.12.3 因为cnnfeatures是dim3 而nerf是dim4 故重新reshape
    #     # nerf_features_tensor = nerf_features_tensor.view(nerf_features_tensor.shape[0], nerf_features_tensor.shape[1],-1)
    #
    #
    #     # Concatenate NeRF features with MVCNN features along the feature dimension
    #     combined_features = torch.cat((cnn_features, nerf_features_tensor), dim=2)  # Adjust dim if needed
    #     print("Shape of combined_features:", combined_features.shape)
    #     # View pooling
    #     pooled_features = torch.max(combined_features, 1)[0]
    #     print("Shape of pooled_features:", pooled_features.shape)
    #
    #     # Additional reduction step before the final fully connected layers  #23.12.3 新加的linear改一下shape
    #     # reduced_features = self.additional_fc(pooled_features.view(pooled_features.shape[0], -1))
    #
    #     # Final fully connected layers
    #     # output = self.net_2(pooled_features.view(pooled_features.shape[0], -1)) #23.12.3
    #
    #     # Final fully connected layers #新的 23.12.3
    #     output = self.net_2(pooled_features)
    #
    #     return output


    #23.12.7 新的
    # def forward(self, x, nerf_features):
    #     # Extract features for each view using MVCNN's CNN (net_1)
    #     cnn_features = self.net_1(x)
    #     print("Shape of cnn_features before reshaped:", cnn_features.shape)
    #     cnn_features = cnn_features.view(
    #         (int(x.shape[0] / self.num_views), self.num_views,
    #          -1))  # Reshape for view pooling [batch_size, num_views, cnn_feature_size]
    #     print("Shape of cnn_features after reshaped:", cnn_features.shape)
    #
    #
    #     # Since nerf_features is already preprocessed and in the correct shape [batch_size, num_views, nerf_feature_size],
    #     # no need for additional reshaping or reduction
    #     print("Shape of nerf_features:", nerf_features.shape)
    #
    #     # Concatenate NeRF features with MVCNN features along the feature dimension
    #     combined_features = torch.cat((cnn_features, nerf_features), dim=2)  #应该整成dim=1 channel tile整成同hxw.
    #     # print("Shape of combined_features:", combined_features.shape)
    #
    #     # View pooling
    #     pooled_features = torch.max(combined_features, 1)[0]
    #     # print("Shape of pooled_features:", pooled_features.shape)
    #
    #     resized_features = self.nerf_feature_reduction(pooled_features)
    #
    #     # Final fully connected layers
    #     output = self.net_2(resized_features.view(resized_features.shape[0], -1))
    #
    #     return output

    #


    #23.12.25新的，这里是的方式是在池化前结合两种features
    #方法一
    def forward(self, x, nerf_features):
        # Extract features for each view using MVCNN's CNN (net_1)
        cnn_features = self.net_1(x)
        # print("Shape of cnn_features before reshaped:", cnn_features.shape) #([96, 2048, 1, 1])
        '''
        24.1.18
        上面这里是views为12的时候，才是[96,2048...]
        现在views是3，缩小4倍，那么shape前面两者均缩小4倍，现在打印出来是[24,512,1,1]
        所以现在结合的就有问题，但是nerf的shape就主要还是跟选择的nerf特征数量直接相关，现在选择的是512的，那接受过来的nerf的shape就是[8,12,512].
        还是得根据具体的numviews跟使用nerf特征数量更改上面net_2的nnLinear。
        '''
        cnn_features = cnn_features.view(
            (int(x.shape[0] / self.num_views), self.num_views,cnn_features.shape[-3],cnn_features.shape[-2],cnn_features .shape[-1]))  # Reshape for view pooling [batch_size, num_views, cnn_feature_size]
        # print("Shape of cnn_features after reshaped:", cnn_features.shape) #([8, 12, 2048, 1, 1])   #value是tensor([[[[[0.1940]],[[0.7365]],......
        # Since nerf_features is already preprocessed and in the correct shape [batch_size, num_views, nerf_feature_size],
        # no need for additional reshaping or reduction
        # print("Shape of nerf_features:", nerf_features.shape)#最初始传进来的nerf的shape([8, 12, 128])
        # Reshape NeRF features to match MVCNN feature format
        nerf_features = nerf_features.view(
            (int(x.shape[0] / self.num_views), self.num_views, -1, cnn_features.shape[-2],cnn_features .shape[-1]))
        # print("shape of nerf_features after reshape:",nerf_features.shape) #([8, 12, 128, 1, 1])
        #24.1.11 给nerf_features在结合前用Relu，看看有没有效果
        nerf_features = F.relu(nerf_features)
        #下面这个是用来第二种方法：在view-pooling 之后结合用的，两种features都先在结合前用torch.max提取单独的每个物体（？）features，然后再结合。
        #但是下面这个就不在这个第一种方法里面使用了，也没有删掉，就注释掉放在这看看吧
        # nerf_features_pooled = torch.max(nerf_features, 1)[0]
        # Concatenate NeRF features with MVCNN features along the feature dimension
        combined_features = torch.cat((cnn_features, nerf_features), dim=2)  #dim2是没有问题的.
        # print("Shape of combined_features:", combined_features.shape)  #([8, 12, 2176, 1, 1])
        # View pooling,后面的view操作是依照原代码上的shape进行操作
        pooled_features = torch.max(combined_features, 1)[0].view(cnn_features.shape[0],-1)
        # print("Shape of pooled_features:", pooled_features.shape)  #([8, 2176])
        #这里我不想再用linear去reshape池化后的组合特征成2048，直接就用2176的，上面mvcnn的net2改一下。
        # resized_features = self.nerf_feature_reduction(pooled_features)
        # print("shape after resized is:", resized_features.shape) #([8, 2048])
        # Final fully connected layers
        output = self.net_2(pooled_features.view(pooled_features.shape[0], -1))
        # print("the output is",output.shape) #([8, 4])
        return output
    ##########


    ########24.1.11
    #下面这个是第二种结合方法，两种features在池化后结合，那么做法就是两者都先用torch.max提取出单一features再结合。
    #也考虑用relu，给nerf用，但是就是在自身使用max之前用，不然提取完单一features再用好像也没必要
    # def forward(self, x, nerf_features):
    #     # Extract features for each view using MVCNN's CNN (net_1)
    #     cnn_features = self.net_1(x)
    #     # print("Shape of cnn_features before reshaped:", cnn_features.shape) #([96, 2048, 1, 1])
    #     cnn_features = cnn_features.view(
    #         (int(x.shape[0] / self.num_views), self.num_views,cnn_features.shape[-3],cnn_features.shape[-2],cnn_features .shape[-1]))  # Reshape for view pooling [batch_size, num_views, cnn_feature_size]
    #     # print("Shape of cnn_features after reshaped:", cnn_features.shape) #([8, 12, 2048, 1, 1])   #value是tensor([[[[[0.1940]],[[0.7365]],......
    #     #24.1.12这里如果用resnet18就是[8,1,512,1,1]
    #
    #
    #     # Since nerf_features is already preprocessed and in the correct shape [batch_size, num_views, nerf_feature_size],
    #     # no need for additional reshaping or reduction
    #     # print("Shape of nerf_features:", nerf_features.shape)#最初始传进来的nerf的shape([8, 12, 128])
    #
    #     # Reshape NeRF features to match MVCNN feature format
    #     nerf_features = nerf_features.view(
    #         (int(x.shape[0] / self.num_views), self.num_views, -1, cnn_features.shape[-2],cnn_features .shape[-1]))
    #     # print("shape of nerf_features after reshape:",nerf_features.shape) #([8, 12, 128, 1, 1])
    #
    #
    #     #24.1.11 给nerf_features在结合前用Relu，看看有没有效果
    #     nerf_features = F.relu(nerf_features)
    #
    #     #下面这个是用来第二种方法：在view-pooling 之后结合用的，两种features都先在结合前用torch.max提取单独的每个物体（？）features，然后再结合。
    #     #24.1.11
    #     nerf_features_pooled = torch.max(nerf_features, 1)[0].view(nerf_features.shape[0],-1)
    #     # print("the nerf features' shape after vp:",nerf_features_pooled.shape) #应该是[8,128]
    #
    #     #24.1.11下面这个是给cnnfeatures池化
    #     cnn_features_pooled = torch.max(cnn_features, 1)[0].view(cnn_features.shape[0],-1)
    #     # print("the nerf features' shape after vp:", cnn_features_pooled.shape) #应该是[8,2048]  #!!!不是2048，但是估计是跟resnet18有关系，现在用的18的shape是[8,512]，如果是resnet50的话应该是2048
    #
    #     # Concatenate NeRF features with MVCNN features along the feature dimension
    #     combined_features = torch.cat((cnn_features_pooled, nerf_features_pooled), dim=1)  #沿着bs=8之后的dim1 featuresmap结合
    #     # print("Shape of combined_features:", combined_features.shape)  #24.1.11 #既然是两者池化后结合就是沿着Dim1结合([8,2176])
    #
    #     # View pooling,后面的view操作是依照原代码上的shape进行操作
    #     #24.1.11这个是没用的，这是第一种方法的操作了
    #     # pooled_features = torch.max(combined_features, 1)[0].view(cnn_features.shape[0],-1)
    #     # print("Shape of pooled_features:", pooled_features.shape)  #([8, 2176])
    #
    #     #这里我不想再用linear去reshape池化后的组合特征成2048，直接就用2176的，上面mvcnn的net2改一下。
    #     # resized_features = self.nerf_feature_reduction(pooled_features)
    #     # print("shape after resized is:", resized_features.shape) #([8, 2048])
    #
    #     # Final fully connected layers
    #     output = self.net_2(combined_features)
    #     # print("the output is",output.shape) #([8, 4])
    #
    #     return output




