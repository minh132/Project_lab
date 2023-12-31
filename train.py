import numpy as np
# from dataset.brats_data_utils_multi_label import get_loader_brats
from dataset.od_dataloader import AquariumDetection,get_transforms
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
import argparse
from monai.losses.dice import DiceLoss
import yaml
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
set_determinism(123)
import os

data_dir = "/home/minh/Documents/DiffUnet-main/data"
logdir = ""

model_save_path = os.path.join(logdir, "model")

env = "pytorch" # or env = "pytorch" if you only have one gpu.

# max_epoch = 300
max_epoch = 200
batch_size = 8
val_every = 1
num_gpus = 2


device = "cpu"

number_modality = 3
number_targets = 1 ## WT, TC, ET

class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        ##### modify ##########
        #######################
        self.embed_model = BasicUNetEncoder(2, number_modality, number_targets, [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(2, number_modality + number_targets, number_targets, [64, 64, 128, 256, 512, 64], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
        
        #######################
        #######################
   
        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)


    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, number_targets, 512, 512), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out

class ISICTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[512, 512],
                                                sw_batch_size=batch_size,
                                                overlap=0.25)
        self.model = DiffUNet()

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                  warmup_epochs=30,
                                                  max_epochs=max_epochs)

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

    def training_step(self, batch):
        image, mask = self.get_input(batch)
        x_start = mask

        x_start = (x_start) * 2 - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")

        loss_dice = self.dice_loss(pred_xstart, mask)
        loss_bce = self.bce(pred_xstart, mask)

        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = self.mse(pred_xstart, mask)

        loss = loss_dice + loss_bce + loss_mse

        self.log("train_loss", loss, step=self.global_step)

        return loss 
 
    def get_input(self, batch):
        image = batch["image"]
        mask = batch["heatmap"]
       
   
        mask = mask.float()
        return image, mask

    def validation_step(self, batch):
        image, mask = self.get_input(batch)    
        
        output = self.window_infer(image, self.model, pred_type="ddim_sample")

        output = torch.sigmoid(output)

        output = (output > 0.5).float().cpu().numpy()

        target = mask.cpu().numpy()

        o = output[:, 0]
        t = target[:, 0]
        tc = dice(o, t)
        
        return [tc]

    def validation_end(self, mean_val_outputs):
        tc = mean_val_outputs[0]

        self.log("tc", tc, step=self.epoch)

        self.log("mean_dice", tc, step=self.epoch)

        mean_dice = tc
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{mean_dice:.4f}.pt"), 
                                        delete_symbol="final_model")

        print(f"tc is {tc}, mean_dice is {mean_dice}")

if __name__ == "__main__":
    
    dataset_path='/home/minh/Documents/DiffUnet-main/data'
    train_ds = AquariumDetection(root=dataset_path, split="train",transforms=get_transforms(True))
    val_ds = AquariumDetection(root=dataset_path, split="valid", transforms=get_transforms(False))

    trainer = ISICTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17750,
                            training_script=__file__)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)


