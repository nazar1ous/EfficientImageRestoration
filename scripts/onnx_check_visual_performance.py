import torch
import torch.onnx
import numpy as np
import onnxruntime
import cv2
from basicsr.utils import img2tensor
import matplotlib.pyplot as plt
import argparse


def get_image_tensor(image_rgb):
    x = img2tensor(image_rgb,
                   bgr2rgb=False,
                   float32=True) / 255.
    x = torch.unsqueeze(x, 0)
    return x


def get_image_deblurred(image_tensor, ort_session):
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image_tensor)}
    ort_outs = ort_session.run(None, ort_inputs)

    restored = torch.from_numpy(ort_outs[0])
    restored = torch.clamp(restored, min=0, max=1) * 255
    restored = torch.squeeze(restored, 0)
    restored = to_numpy(restored).transpose(1, 2, 0)
    return restored


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", required=False, type=str,
                        default="datasets/GoPro/train/blur_crops/GOPR0372_07_00-000047_s001.png",
                        help="Path to an input image.")
    parser.add_argument("-o", "--model_onnx_path", required=False, type=str,
                        default="optimized_models_onnx/NAFNetLocal-width16-fp=int8.onnx",
                        help="Path to an optimized onnx model")
    args = parser.parse_args()
    image = cv2.imread(args.image_path)
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ort_session = onnxruntime.InferenceSession(args.model_onnx_path, providers=['CPUExecutionProvider'])
    img_tensor = get_image_tensor(image)
    deblurred_image = get_image_deblurred(img_tensor, ort_session)

    fig, axes = plt.subplots(1, 2, figsize=(15, 15))

    axes[0].imshow(np.uint(image))
    axes[1].imshow(np.uint(deblurred_image))
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[0].set_title('Input')
    axes[1].set_title('Deblurred')

    plt.show()
