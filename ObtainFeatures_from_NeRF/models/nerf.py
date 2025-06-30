import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

class CBAM(nn.Module):
    def __init__(self, channel_count):
        super(CBAM, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channel_count, channel_count // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel_count // 16, channel_count, 1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        channel_att = self.sigmoid_channel(out)
        x = x * channel_att

        # Clean up intermediate tensors 清理临时的张量
        del avg_out, max_out, out, channel_att

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(x))
        x = x * spatial_att

        '''
        修正后的代码确保空间注意力图（spatial_att）是使用合并的 avg_out 和 max_out 计算的，
        但会将该注意力图与原始输入张量 x 相乘，以产生最终输出。这样可以确保输入张量的通道尺寸在通过 CBAM 模块后保持不变。
        '''

        # # Spatial Attention
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # spatial_att_map = torch.cat([avg_out, max_out], dim=1)
        # spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_att_map))
        # x = x * spatial_att
        #
        # # Clean up intermediate tensors 清理临时的张量
        # del avg_out, max_out, spatial_att_map, spatial_att

        return x


class NeRF(nn.Module):

    #2023.10.26原来的
    # def __init__(self,
    #              D=8, W=256,
    #              in_channels_xyz=63, in_channels_dir=27,
    #              skips=[4]):

    #2023.10.26 新增一个变量控制在nerf_fine的时候获取第八层输出
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27,
                 skips=[4],
                 record_outputs=False):




        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips
        self.eighth_layer_outputs = []
        # self.tentative_output = None
        # Add CBAM after the 8th layer 2023.9.26
        self.cbam = CBAM(channel_count=256)  # assuming 256 channels after the 8th layer

        # Add a convolutional layer for dimensionality reduction 2023.9.26
        self.conv_dim_reduce = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)

        #23.11.7 本来kernel跟stride都是2 为了后续读取效益升到4
        self.max_pool = nn.MaxPool2d(kernel_size=4, stride=4)
        ##################2023.9.26

        #2023.10.27 flag用作只获取nerf_fine时的第八层输出
        self.record_outputs = record_outputs

        # 23.11.12用来控制val_step里面不进去获取第八层输出
        self.is_training_mode = True


        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, x, sigma_only=False,system= None):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)


            #2023.10.26
            # if self.record_outputs: #只要nerf_fine的NeRF() calling，原本是没有这个，直接if i==7
            #23.11.12新的
            if self.is_training_mode and self.record_outputs:
                if i == 7:
                    B, C = xyz_.shape

                    #原本的23.11.11
                    # H, W = self.find_factors(B)

                    #新的
                    new_B = B // 12
                    H, W = self.find_factors(new_B)
                # H = W = int((B / C) ** 0.5)
                #     print('B is ',B) #4096 23.11.11 BC依然是一样的
                #     print('C is ', C) #256
                #     print('H is ', H)
                #     print('W is ', W)

                    # print("Shape of xyz_ before reshaping:", xyz_.shape)
                    # tentative_output = xyz_.view(-1, C, H, W)
                    # print("tentative_output.shape before the cbam:",tentative_output.shape)  #[1,256,64,64]
                    # cbam_output = self.cbam(tentative_output)
                    # print("cbam_output.shape after cbam:",cbam_output.shape)  # self.tentative_output.shape after cbam: torch.Size([1, 256, 64, 64]) at bs as 32
                    # #23.11.6缩减output大小,用最大汇聚层
                    # reduced_spatially = self.max_pool(cbam_output)
                    # B2, C2, H2, W2 = reduced_spatially.shape
                    # print("output shape after maxpool is :",reduced_spatially.shape) #[1,2,16,16] when max_pool's kernel and stride are 4 #23.11.11
                    # # 23.10.31用下面这个
                    # after_reduced = reduced_spatially.view(B2,-1)
                    # print("after_reduced shape is :",after_reduced.shape) #[1,512]
                    # # Detach the tensor from the graph, and move it to CPU memory.
                    # detached_output = after_reduced.detach().to('cpu')
                    # self.eighth_layer_outputs.append(detached_output)#23.11.4 new

                    #23.11.11 新的
                    # print("xyz's content is ", xyz_)
                    tentative_output = xyz_.view(12, C, H, W)
                    print("tentative_output's content before the cbam:", tentative_output.shape) #[12,256,16,16]
                    cbam_output = self.cbam(tentative_output)
                    print("cbam_output's content after cbam:", cbam_output.shape)#[12,2,16,16]
                    reduced_spatially = self.max_pool(cbam_output)
                    print("output content after maxpool is :", reduced_spatially.shape)#[12,2,4,4]
                    after_reduced = reduced_spatially.view(12, -1)
                    print("after_reduced's content is :", after_reduced.shape)#[12,32]
                    detached_output = after_reduced.detach().to('cpu')
                    for view_output in detached_output:#12x[32]
                        self.eighth_layer_outputs.append(view_output)

                ###########


        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out


    def find_factors(self,n):
        """Find factors of n that are close to each other."""
        for i in range(int(n ** 0.5), 0, -1):
            if n % i == 0:
                return i, n // i
        return 1, n

    #23.11.10 这里是手动清除存贮第八层输出的变量
    def reset_outputs(self):
        self.eighth_layer_outputs = []

    # 23.11.12用来控制val_step里面不进去获取第八层输出
    def set_mode(self, is_training_mode: bool):
        self.is_training_mode = is_training_mode



