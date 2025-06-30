# 2023.9.13 在这里获取第八层输出
if i == 7:
    B, C = xyz_.shape
    H, W = self.find_factors(B)
    # H = W = int((B / C) ** 0.5)
    # print('B is ',B)
    # print('C is ', C)
    # print('H is ', H)
    # print('W is ', W)

    print("Shape of xyz_ before reshaping:", xyz_.shape)  # [32768,256]for bs as 1
    self.tentative_output = xyz_.view(-1, C, H, W)
    print("self.tentative_output.shape before the cbam:",
          self.tentative_output.shape)  # [768, 256, 1, 1] for bs as 1
    self.tentative_output = self.cbam(self.tentative_output)

    print("self.tentative_output.shape after cbam:",
          self.tentative_output.shape)  # self.tentative_output.shape after cbam: torch.Size([1, 256, 128, 256]) at bs as 1

    self.tentative_output = self.conv_dim_reduce(self.tentative_output)  # 缩小tensor
    self.eighth_layer_outputs.append(self.tentative_output)  # 2023.9.20 new
    # print('now, the shape of 8th output is:',self.eighth_layer_outputs[-1].shape)
    # print('Length of eighth_layer_outputs:', len(self.eighth_layer_outputs))

    # 2023.9.26 应用上CBAM和一个卷积层来缩小tensor
##########################