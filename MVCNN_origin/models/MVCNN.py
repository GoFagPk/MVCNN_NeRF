import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model

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

    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11'):  #本来cnn_name是vgg11
        super(SVCNN, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        # self.classnames = ['bag', 'pillow', 'sink', 'toilet']
        # self.classnames = ['bag', 'bed', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display', 'door', 'pillow', 'shelf',
        #                    'sink', 'sofa',
        #                    'table', 'toilet']
        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()
        #24.1.9
        self.reduction_for_vgg11 = nn.Linear(512 * 8 * 8, 25088)

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,40)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,15)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048,15)
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
            
            self.net_2._modules['6'] = nn.Linear(4096,4)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            # print("y's shape is ",y.shape) for vgg11 y's shape is  torch.Size([64, 512, 8, 8]).
            #这里直接用vgg11于我的数据集报错了：mat1 and mat2 shapes cannot be multiplied (64x32768 and 25088x4096)
            #那么我的数据集的rgb转出来就是64x32768，而vgg11的classifier头个linear是这样的self.classifier = nn.Sequential(
            #nn.Linear(512 * 7 * 7, 4096)，所以是(25088x4096)，就不匹配了，那么我这里直接修改vgg11的第一层网络。
            #
            # y = y.view(y.shape[0],-1)
            # y = self.reduction_for_vgg11(y)
            #
            # 原来的
            return self.net_2(y.view(y.shape[0],-1))

            # return self.net_2(y)




class MVCNN(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12):  #这里也把vgg11改成resnet50
        super(MVCNN, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        # self.classnames = ['bag', 'pillow', 'sink', 'toilet']
        # self.classnames = ['bag', 'bed', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display', 'door', 'pillow', 'shelf',
        #                    'sink', 'sofa',
        #                    'table', 'toilet']
        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()
        self.reduction_for_vgg11 = nn.Linear(512 * 8 * 8, 25088)

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            print("net_1 are:",self.net_1)
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

    def forward(self, x):
        y = self.net_1(x)
        # print("y's shape is :",y.shape) #([96, 2048, 1, 1])
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))#(8,12,512,7,7)
        # print("y's shape after reshape is :",y.shape) #([8, 12, 2048, 1, 1])

        #这个专门来看池化的效果，当然也可以用作vgg11进net_2之前的reshape的中间变量
        # y1 = torch.max(y,1)[0].view(y.shape[0],-1)
        # print("y's shape after torch.max is :", y1.shape) #([8, 2048])
        # y1 = self.reduction_for_vgg11(y1)

        #原来的 24.1.14
        return self.net_2(torch.max(y,1)[0].view(y.shape[0],-1))

        #用于vgg11的
        # return self.net_2(y1)

