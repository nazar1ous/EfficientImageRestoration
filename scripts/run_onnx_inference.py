import onnxruntime
import time
import numpy as np
import argparse


def benchmark(model_path):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.zeros((1, 3, 256, 256), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--model_onnx_path", required=False, type=str,
                        default="optimized_models_onnx/NAFNetLocal-width16-fp=int8.onnx",
                        help="Path to an optimized onnx model")
    args = parser.parse_args()
    benchmark(args.model_onnx_path)
