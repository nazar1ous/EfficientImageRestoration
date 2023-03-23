import lightning as pl
from basicsr.data.GoPro_deblur_dataset import PairedImageDataset
from torch.utils.data import DataLoader
import torch

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from omegaconf import DictConfig
import importlib


def get_class(class_route):
    values = class_route.split('.')
    module_path = '.'.join(values[:-1])
    class_name = values[-1]

    module = importlib.import_module(module_path)
    class_ = getattr(module, class_name)
    return class_


def get_params_from_dict(dct, names):
    return {key: dct[key] for key in dct if key not in names}


def get_optimizer(name, params, model_weights):
    return torch.optim.AdamW(model_weights, **params)


def get_scheduler(name, params, optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)


class DeblurModel(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        model_class = get_class(self.config.generator.class_name)
        self.model = model_class(**self.config.generator.params)
        if self.config.preloading.generator_path:
            self.model.load_state_dict(torch.load(self.config.preloading.generator_path))

        loss_class = get_class(self.config.loss.class_name)
        self.loss = loss_class(**self.config.loss.params)
        self.psnr_func = peak_signal_noise_ratio
        self.ssim_func = structural_similarity_index_measure

        self.optimizer = None
        self.scheduler = None

        self.batch_size = self.config.dataloader.batch_size
        self.num_workers = self.config.dataloader.num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        print(self.hparams)

    def setup(self, stage=None):
        if stage == "fit":
            # print(self.config.dataset.train.params)
            self.train_dataset = PairedImageDataset("train", self.config.dataset.train.params)
            self.val_dataset = PairedImageDataset("val", self.config.dataset.val.params)
        else:
            self.test_dataset = PairedImageDataset("test", self.config.dataset.test.params)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        blur, gt = batch['lq'], batch['gt']
        restored = self(blur)
        loss = self.loss(restored, gt)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        blur, gt = batch['lq'], batch['gt']
        restored = self(blur)
        restored = torch.clamp(restored, min=0, max=1)
        gt = torch.clamp(gt, min=0, max=1)
        psnr, ssim = self.psnr_func(restored, gt, data_range=1), self.ssim_func(restored, gt, data_range=1)
        loss = self.loss(restored, gt)

        self.log("val_loss", loss.detach().cpu(), on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr, on_epoch=True, prog_bar=True)
        self.log("val_ssim", ssim, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        blur, gt = batch['lq'], batch['gt']
        restored = self(blur)
        restored = torch.clamp(restored, min=0, max=1)
        gt = torch.clamp(gt, min=0, max=1)
        psnr, ssim = self.psnr_func(restored, gt, data_range=1), self.ssim_func(restored, gt, data_range=1)

        self.log("test_psnr", psnr, on_epoch=True, prog_bar=True)
        self.log("test_ssim", ssim, on_epoch=True, prog_bar=True)
        

    def configure_optimizers(self):
        optimizer_class = get_class(self.config.optimizer.class_name)

        self.optimizer = optimizer_class(self.model.parameters(), **self.config.optimizer.params)
        scheduler_class = get_class(self.config.scheduler.class_name)
        self.scheduler = scheduler_class(self.optimizer, **self.config.scheduler.params)
        return [self.optimizer], [self.scheduler]



