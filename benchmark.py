import torch
from basicsr.models.lightning.deblur_model import DeblurModel, get_class
from lightning import Trainer
import numpy as np
import hydra
from omegaconf import DictConfig
from train import seed_everything
from ptflops import get_model_complexity_info
import torch.nn as nn
from typing import Union
# from torchsummary import summary
# from torch.profiler import profile, record_function, ProfilerActivity


CONFIG_PATH = "config"


def measure_inference_time(model: nn.Module, inp_shape: tuple, device_str="cuda",
                           num_iter=1000, num_warmup_iter=100) -> Union[float, float]:
    device = torch.device(device_str)
    model.to(device)
    dummy_input = torch.randn(*inp_shape, dtype=torch.float).to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((num_iter, 1))
    with torch.no_grad():
        # GPU-WARM-UP
        for _ in range(num_warmup_iter):
            _ = model(dummy_input)

        for rep in range(num_iter):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / num_iter
    std_syn = np.std(timings)
    return mean_syn, std_syn


def get_peak_memory_usage(model, input_shape):
    torch.cuda.reset_peak_memory_stats()
    data_tensor = torch.randn(*input_shape).cuda()
    model.cuda()
    with torch.no_grad():
        output = model(data_tensor)
        # convert to MB
        memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return memory_usage


@hydra.main(config_path=CONFIG_PATH, config_name="config")
def run_test(cfg: DictConfig):
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
    torch.save(model.model.state_dict(), f"save_models/{model.config.generator.name}.pth")

    trainer.test(model)

    torch_model = model.model

    inp_shape = (3, 256, 256)
    tensor_shape = (1, *inp_shape)

    macs, params = get_model_complexity_info(torch_model, inp_shape, verbose=False, print_per_layer_stat=False)
    peak_memory_usage = get_peak_memory_usage(torch_model, tensor_shape)
    mean, std = measure_inference_time(torch_model, tensor_shape, device_str="cpu")

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(f'MACs: {macs}; params: {params}\n'
          f'Inference time: Mean: {mean}; std: {std}\n'
          f'Peak memory usage (MB): {peak_memory_usage}')


if __name__ == "__main__":
    seed_everything(42)
    torch_model = run_test()




