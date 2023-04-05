import torch
from basicsr.models.lightning.deblur_model import DeblurModel, get_class
from lightning import Trainer
import random
import os
import numpy as np
import hydra
from omegaconf import DictConfig


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)
CONFIG_PATH = "config"


@hydra.main(config_path=CONFIG_PATH, config_name="config")
def run_train(cfg : DictConfig) -> None:

    trainer_callbacks = []
    for _, callback_dict in cfg.callbacks.items():
        if callback_dict['use']:
            class_ = get_class(callback_dict['class_name'])
            trainer_callbacks.append(class_(**callback_dict['params']))
    
    logger_class = get_class(cfg.logger.class_name)
    logger = logger_class(**cfg.logger.params)

    ckpt_path = cfg.preloading.pl_path
    torch.set_float32_matmul_precision('high')

    trainer = Trainer(
        **cfg.trainer.params,
        logger=logger,
        callbacks=trainer_callbacks,

    )
    if ckpt_path:
        model = DeblurModel.load_from_checkpoint(ckpt_path)
    else:
        model = DeblurModel(cfg)

    model = torch.compile(model, mode="default")
    trainer.fit(model, ckpt_path=ckpt_path)
    # model = torch.compile(model, disable=True, dynamic=True)
    trainer.test(model)


if __name__ == "__main__":
    run_train()

    