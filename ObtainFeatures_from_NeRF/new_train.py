import os, sys
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.logging import TestTubeLogger

class SaveOutputsCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        object_idx = pl_module.current_object_idx
        epoch = trainer.current_epoch
        pl_module.save_8th_layer_outputs(object_idx, epoch)

if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(f'ckpts/{hparams.exp_name}'),
        filename='{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        verbose=True,
        mode='min',
        save_top_k=5,
        save_last=True
    )

    #23.11.7 callbacks里面添加调用
    save_outputs_callback = SaveOutputsCallback()

    logger = TensorBoardLogger(
        save_dir='logs',
        name = hparams.exp_name
    )


    early_stop_callback = EarlyStopping(
        monitor='val_loss',  # Make sure this matches exactly what you log in `validation_epoch_end`
        patience=3,
        strict=False,
        verbose=True,
        mode='min'
    )

    for object_idx in range(64):
        system.current_object_idx = object_idx
        print('this is the object_idx in trianing_loop:',object_idx)

        trainer = Trainer(max_epochs=hparams.num_epochs,
                        callbacks=[early_stop_callback,checkpoint_callback,save_outputs_callback],
                        resume_from_checkpoint=hparams.ckpt_path,
                        logger=logger,
                        enable_model_summary = False,
                        enable_progress_bar = True,  #用来展示训练后的过程内容
                        gpus=hparams.num_gpus,
                        accelerator = 'gpu',
                        devices = 1 ,
                        # strategy='dp',  # Previously distributed_backend
                        num_sanity_val_steps=0,
                        benchmark=True,
                        profiler=SimpleProfiler() if hparams.num_gpus > 0 else None)

        trainer.fit(system)





class NeRF(nn.Module):

    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27,
                 skips=[4],
                 record_outputs=False):

        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips
        self.eighth_layer_outputs = []
        self.cbam = CBAM(channel_count=256)  # assuming 256 channels after the 8th layer

        # Add a convolutional layer for dimensionality reduction 2023.9.26
        self.conv_dim_reduce = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)

        self.max_pool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.record_outputs = record_outputs

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

            if self.is_training_mode and self.record_outputs:
                if i == 7:
                    B, C = xyz_.shape
                    new_B = B // 12
                    H, W = self.find_factors(new_B)
                    tentative_output = xyz_.view(12, C, H, W)#[12,256,16,16]
                    cbam_output = self.cbam(tentative_output)
                    reduced_spatially = self.max_pool(cbam_output)
                    after_reduced = reduced_spatially.view(12, -1)
                    detached_output = after_reduced.detach().to('cpu')
                    for view_output in detached_output:
                        self.eighth_layer_outputs.append(view_output)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out