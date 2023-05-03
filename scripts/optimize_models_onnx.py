import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization import CalibrationDataReader
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from pathlib import Path

from train import seed_everything
from basicsr.models.lightning.deblur_model import DeblurModel


CONFIG_PATH = "../config"
SAVE_DIR = "optimized_models_onnx"


def assert_equal_conversion(model_onnx_path, input_tensor, output_tensor_torch):
    ort_session = onnxruntime.InferenceSession(model_onnx_path, providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
    ort_outs = ort_session.run(None, ort_inputs)
    # compare ONNX Runtime and PyTorch results
    print(np.testing.assert_allclose(to_numpy(output_tensor_torch), ort_outs[0], rtol=1e-03, atol=1e-05))
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def export_onnx(torch_model, out_model_onnx_path):
    device = torch.device("cpu")
    batch_size = 1
    x = torch.randn(batch_size, 3, 256, 256, requires_grad=False).to(device)
    torch_model.to(device)
    torch_model.eval()
    torch_out = torch_model(x)
    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      out_model_onnx_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})
    assert_equal_conversion(out_model_onnx_path, x, torch_out)


class ResNet50DataReader(CalibrationDataReader):
    def __init__(self, dataset, model_path):
        self.enum_data = None
        self.dataset = dataset
        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.dataset)

    @staticmethod
    def iter_dataset(dataset, inp_name):
        for i in range(len(dataset)):
            tnsr = np.expand_dims(dataset.__getitem__(i)['lq'], axis=0)
            yield {inp_name: tnsr}

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = self.iter_dataset(self.dataset, self.input_name)
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def quantize_torch_model(model_onnx_path, out_model_quantized_onnx_path, torch_train_dataset):
    quantize_static(
        model_onnx_path,
        out_model_quantized_onnx_path,
        ResNet50DataReader(torch_train_dataset, model_onnx_path),
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
        optimize_model=False,
    )


@hydra.main(config_path=CONFIG_PATH, config_name="config")
def get_optimized_models(cfg: DictConfig):
    ckpt_path = cfg.preloading.pl_path
    torch.set_float32_matmul_precision('high')
    if ckpt_path:
        model = DeblurModel.load_from_checkpoint(ckpt_path)
    else:
        model = DeblurModel(cfg)
    model.setup('fit')
    save_dir = Path(SAVE_DIR)
    train_dataset = model.train_dataset
    model_name = cfg.generator.name
    ext_name = '.onnx'
    torch_model = model.model
    out_model_onnx_path = save_dir.joinpath(model_name + '-fp=32' + ext_name)
    out_model_quantized_onnx_path = save_dir.joinpath(model_name + '-fp=int8' + ext_name)
    export_onnx(torch_model, str(out_model_onnx_path))

    quantize_torch_model(str(out_model_onnx_path), str(out_model_quantized_onnx_path), train_dataset)


if __name__ == "__main__":
    seed_everything(42)
    torch_model = get_optimized_models()

