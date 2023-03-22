import torch
from lightning_model import DeblurModel, get_class
from lightning import Trainer
import random
import os
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import get_original_cwd, to_absolute_path


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CONFIG_PATH = "config"

# hydra.run.dir = '.'

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
    # os.chdir(get_original_cwd())
    # print(os.getcwd())
    if ckpt_path:
        model = DeblurModel.load_from_checkpoint(ckpt_path)
        trainer = Trainer()
        trainer.fit(model, ckpt_path=ckpt_path)
    else:
        model = DeblurModel(cfg)
        # model_compiled = torch.compile(model, mode="reduce-overhead")

        trainer = Trainer(
            **cfg.trainer.params,
            logger=logger,
            callbacks=trainer_callbacks,

        )
        trainer.fit(model)

    trainer.test(model)




if __name__ == "__main__":
    seed_everything(42)
    run_train()

    