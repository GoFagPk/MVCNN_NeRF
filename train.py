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
# from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

#23.11.2新需要的包
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint , Callback
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.strategies import strategy
#23.11.3#用来分布数据加载在多个gpus上
from torch.utils.data.distributed import DistributedSampler
import gc

#23.11.7为了解决trainer.fit是一个闭环call导致后面的保存代码无法正常执行，多弄一个callback放进trainer
#并且关于只在nerf——fine里提取空间信息的设置移到了这里面 因为这个SaveOutputCallback在Trainer参数里靠前 他会先执行 所以放这里也可以在正式mlp训练前执行到这个record_outputs.
class SaveOutputsCallback(Callback):

    #23.11.11
    def on_train_epoch_start(self, trainer, pl_module):
        # Enable recording only in the last epoch
        if trainer.current_epoch == trainer.max_epochs - 1:
            pl_module.nerf_fine.record_outputs = True
    #原本的
    # def on_train_epoch_end(self, trainer, pl_module):
    #     # Only save outputs in the last epoch
    #     if trainer.current_epoch == trainer.max_epochs - 1:
    #         object_idx = pl_module.current_object_idx
    #         epoch = trainer.current_epoch
    #         pl_module.save_8th_layer_outputs(object_idx, epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        object_idx = pl_module.current_object_idx
        epoch = trainer.current_epoch
        # Save and clear outputs only in the last epoch
        if epoch == trainer.max_epochs - 1:
            pl_module.save_8th_layer_outputs(object_idx, epoch)
            pl_module.nerf_fine.record_outputs = False



class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()

        # 23.11.2 自从更新了pytorch lightning为1.8.1之后 这个传参用法被取消了
        # self.hparams = hparams
        # 下面这个是新的
        self.save_hyperparameters(hparams)

        self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF(record_outputs=False)
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:

            #2023.10.27这个是原本的，没有指定flag去只获得fine的nerf
            # self.nerf_fine = NeRF()
            #现在的
            self.nerf_fine = NeRF(record_outputs=False)
            ######

            self.models += [self.nerf_fine]

        #23.10.31用来保存输出的序号变量
        self.current_object_idx = 0
        #23.11.3 训练数据集和验证数据集
        self.datasets = []  # train_datasets
        self.val_datasets = []

        #23.11.10用来控制最后一轮epoch才保存输出
        self.current_train_epoch = 0



    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs

    def forward(self, rays,white_back):   #原本是forward(self,rays) 多了一个white_back
        """Do batched inference on rays using chunk."""
        # print("shape of rays in NeRF's forward():",rays.shape)
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            #self.train_dataset.white_back  #这是原本的23.10.31
                            white_back)  #重新写成这个

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    # 原来的代码
    # def prepare_data(self):
    #     dataset = dataset_dict[self.hparams.dataset_name]
    #     kwargs = {'root_dir': self.hparams.root_dir,
    #               'img_wh': tuple(self.hparams.img_wh)}
    #     if self.hparams.dataset_name == 'llff':
    #         kwargs['spheric_poses'] = self.hparams.spheric_poses
    #         kwargs['val_num'] = self.hparams.num_gpus
    #     self.train_dataset = dataset(split='train', **kwargs)
    #     self.val_dataset = dataset(split='val', **kwargs)

    # 2023.10.19 新更改的
    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus

        num_objects = 780//12  #这里记得要改！！！！！！！！！！！！！！！
        for object_idx in range(num_objects):   #num_objects=85的话 range(85)是从0-84
            train_dataset = dataset(split='train', object_idx=object_idx, **kwargs)
            val_dataset = dataset(split='val',object_idx=object_idx,**kwargs)
            self.datasets.append(train_dataset) #现在datasets是list，64
            # print("datasets' length:",len(self.datasets))
            self.val_datasets.append(val_dataset)
            # print("val datasets' length:", len(self.val_datasets))


        # for idx, dataset in enumerate(self.datasets):
        #     print(f"Dataset at index {idx}:")
        #     print(f"  Type: {type(dataset)}")
        #     # If the datasets are of a type that has a length (like a list or a PyTorch Dataset), you can print that.
        #     try:
        #         print(f"  Length: {len(dataset)}")
        #     except TypeError:
        #         print(f"  Length: Not applicable (no len())")
        #
        #     # If it's safe to print the dataset directly (e.g., it's a small dictionary of parameters), do so here.
        #     # If it's a large dataset or contains large tensors, consider printing just the first item or a summary.
        #     print(f"  Content (summary or first item): {str(dataset)[:200]}")  # Truncate for safety



    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    #23.10.31原来的
    # def train_dataloader(self):
    #     return DataLoader(self.train_dataset,
    #                       shuffle=True,
    #                       num_workers=4,
    #                       batch_size=self.hparams.batch_size,
    #                       pin_memory=True)

    #23.10.31现在的，没有strategy用这个 有的话 如果是dp 也用这个
    def train_dataloader(self):
        print(f"Current object index: {self.current_object_idx}")
        # print(f"Number of datasets: {len(self.datasets)}")
        current_dataset = self.datasets[self.current_object_idx]
        return DataLoader(current_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    #23.11.3又有一个新的
    # def train_dataloader(self):
    #     print(f"Current object index: {self.current_object_idx}")
    #     print(f"Number of datasets: {len(self.datasets)}")
    #     current_dataset = self.datasets[self.current_object_idx]
    #
    #     # Replace 'shuffle=True' with a distributed sampler here.
    #     sampler = DistributedSampler(current_dataset)
    #
    #     return DataLoader(current_dataset,
    #                       sampler=sampler,
    #                       num_workers=4,
    #                       batch_size=self.hparams.batch_size,
    #                       pin_memory=True)


    #23.10.31旧的
    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset,
    #                       shuffle=False,
    #                       num_workers=4,
    #                       batch_size=1, # validate one image (H*W rays) at a time
    #                       pin_memory=True)

    #23.10.31新的
    def val_dataloader(self):
        current_dataset = self.val_datasets[self.current_object_idx]
        return DataLoader(current_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    ##########
    
    def training_step(self, batch, batch_nb):
        #23.11.12设置是否进控制
        self.nerf_fine.set_mode(True)

        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs = self.decode_batch(batch)
        white_back = self.datasets[object_idx].white_back
        results = self(rays,white_back)
        # print("the shape of rgb_coarse in trianing_step is:",results['rgb_coarse'].shape)
        log['train/loss'] = loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
               }

    def validation_step(self, batch, batch_nb):

        # 23.11.12设置是否进控制
        self.nerf_fine.set_mode(False)

        rays, rgbs = self.decode_batch(batch)
        white_back = self.val_datasets[object_idx].white_back
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays,white_back)
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        return log

    #23.11.5 为了配合新的tensorboardlogger，不用下面这个了
    # def validation_epoch_end(self, outputs):
    #     mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
    #
    #     return {'progress_bar': {'val_loss': mean_loss,
    #                              'val_psnr': mean_psnr},
    #             'log': {'val/loss': mean_loss,
    #                     'val/psnr': mean_psnr}
    #            }

    #23.11.5 用新的val——epoch——end
    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val_loss', mean_loss, prog_bar=True)
        self.log('val_psnr', mean_psnr, prog_bar=True)

    # #2023.10.19新的trian函数
    # def custom_train(self):
    #     eighth_outputs_list = []
    #
    #     for train_dataset in self.datasets:
    #         # update train dataset
    #         self.train_dataset = train_dataset
    #
    #         dataloader = self.train_dataloader()
    #         for epoch in range(self.hparams.num_epochs):
    #             for batch in dataloader:
    #                 output = self.training_step(batch, epoch)
    #                 loss = output['loss']
    #
    #                 # Here, you can add additional logic such as printing or saving the model
    #                 print(f"Epoch: {epoch}, Loss: {loss.item()}")
    #
    #             # Handle validation (if you want to do validation after each epoch)
    #             val_dataloader = self.val_dataloader()
    #             for val_batch in val_dataloader:
    #                 self.validation_step(val_batch, epoch)
    #
    #             val_results = self.validation_epoch_end(
    #                 [])  # Assuming you want to do some aggregated logic after all val batches
    #             print(
    #                 f"Validation Loss: {val_results['progress_bar']['val_loss']}, PSNR: {val_results['progress_bar']['val_psnr']}")


    #2023.10.31 用来在main里调用的保存输出的函数
    #本地跟remote保存地址
    #/root/autodl-tmp/8thOutput
    #D:\\Pytorch_thesis\\eighth_layer_output\\pillow_features
    #D:\\Pytorch_thesis\\eighth_layer_output
    #/root/autodl-tmp2/Pytorch_thesis/eighth_layer_output
    def save_8th_layer_outputs(self, object_idx,epoch,output_dir='D:\\Pytorch_thesis\\eighth_layer_output\\toilet_features'):
        if epoch == self.hparams.num_epochs - 1:
            outputs_8th_layer = self.nerf_fine.eighth_layer_outputs
            save_path = os.path.join(output_dir, f'outputs_8th_layer_object_{object_idx}.pt')
            torch.save(outputs_8th_layer, save_path)

            #23.11.5
            # Check if the file was saved successfully
            if os.path.exists(save_path):
                print(f'File saved successfully: {save_path}')
            else:
                print(f'Failed to save file: {save_path}')

            #23.11.5新加的去除第八层输出占用内存的
            del self.nerf_fine.eighth_layer_outputs
            gc.collect()
            ####23.11.5
            # Clear the list to free up memory and prepare for the next object
            self.nerf_fine.eighth_layer_outputs = []


            print(f'8th layer outputs saved to {save_path}')

    #23.11.11做出修改，暂时不用
    # def on_train_epoch_end(self, unused=None):
    #     print("Now coming into on_train_epoch_end")
    #     if self.current_epoch == self.hparams.num_epochs - 1:
    #         # Enable recording in the last epoch
    #         self.nerf_fine.record_outputs = True
    #         print("now turn the record_outputs to True.")
    #     else:
    #         # Disable recording and clear any recorded outputs
    #         self.nerf_fine.record_outputs = False
    #         # self.nerf_fine.reset_outputs()


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)

    #23.11.2 新的pl做了新的更改
    # checkpoint_callback = ModelCheckpoint(filepath=os.path.join(f'ckpts/{hparams.exp_name}',
    #                                                             '{epoch:d}'),
    #                                       monitor='val/loss',
    #                                       mode='min',
    #                                       save_top_k=5,)

    #1.8.1的pl的新写法
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(f'ckpts/{hparams.exp_name}'),
        filename='{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        verbose=True,
        mode='min',
        save_top_k=2,
        save_last=True
    )

    #23.11.7 callbacks里面添加调用
    save_outputs_callback = SaveOutputsCallback()

    #23.11.2做相应修改，原来的也用不了了
    # logger = TestTubeLogger(
    #     save_dir="logs",
    #     name=hparams.exp_name,
    #     debug=False,
    #     create_git_tag=False
    # )

    #新的
    # logger = CSVLogger(
    #     save_dir='logs',
    #     name=hparams.exp_name
    # )

    #还是新的23.11.5
    logger = TensorBoardLogger(
        save_dir='logs',
        name = hparams.exp_name
    )
    #############

    # 23.10.20 这个是原来的，下面我在弄个新的，用来跑新写的train代码。
    # trainer = Trainer(max_epochs=hparams.num_epochs,
    #                   checkpoint_callback=checkpoint_callback,
    #                   resume_from_checkpoint=hparams.ckpt_path,
    #                   logger=logger,
    #                   early_stop_callback=None,
    #                   weights_summary=None,
    #                   progress_bar_refresh_rate=1,
    #                   gpus=hparams.num_gpus,
    #                   distributed_backend='ddp' if hparams.num_gpus>1 else None,
    #                   num_sanity_val_steps=0,
    #                   benchmark=True,
    #                   profiler=hparams.num_gpus==1)

    # trainer.fit(system)
    #############


    #23.11.2 新需要的参数函数 /// 11.5 有新的
    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',  # or another metric that you wish to monitor
    #     patience=3,  # number of epochs with no improvement after which training will be stopped
    #     verbose=False,
    #     mode='min'  # or 'max' depending on what you are monitoring
    # )

    #23.11.5新的
    early_stop_callback = EarlyStopping(
        monitor='val_loss',  # Make sure this matches exactly what you log in `validation_epoch_end`
        patience=3,
        strict=False,
        verbose=True,
        mode='min'
    )


    #######23.10.31 新的
    for object_idx in range(61,65):
        system.current_object_idx = object_idx
        # print('this is the object_idx in trianing_loop:',object_idx)
        trainer = Trainer(max_epochs=hparams.num_epochs,
                        #1.8.1的pl不再接受下面这个参数
                        # checkpoint_callback=checkpoint_callback,
                         #换成这个新的 23.11.2
                        callbacks=[checkpoint_callback, save_outputs_callback],
                        #####
                        resume_from_checkpoint=hparams.ckpt_path,
                        logger=logger,
                        #23.11.3下面这个weights_summary 在pl 1.8.1里面已经没了
                        # weights_summary=None,
                        #换成下面这个 23.11.3
                        enable_model_summary = False,
                        ######
                        #23.11.3下面这个也是一样的情况
                        # progress_bar_refresh_rate=1,
                        enable_progress_bar = True,  #用来展示训练后的过程内容
                        ###########
                        #23.11.3 现在gpu写法也不一样了
                        gpus=hparams.num_gpus,
                        accelerator = 'gpu',
                        devices = 1 ,
                        ########
                        #23.11.3一样的情况
                        # distributed_backend='ddp' if hparams.num_gpus > 1 else None,
                        #现在是这样的，有ddp还有dp能选，自己看区别
                        # strategy='dp',  # Previously distributed_backend
                        ######
                        num_sanity_val_steps=0,
                        benchmark=True,
                        profiler=SimpleProfiler() if hparams.num_gpus > 0 else None)

        trainer.fit(system)

        #23.11.7 下面暂时用不到，挪用到callback里面去调用
        # current_epoch = trainer.current_epoch
        # system.save_8th_layer_outputs(object_idx,current_epoch)

    #########


