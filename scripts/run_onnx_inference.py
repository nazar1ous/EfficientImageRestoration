import onnxruntime
import time
import numpy as np
import argparse
from memory_profiler import memory_usage


import matplotlib.pyplot as plt
from pandas import DataFrame, to_datetime


def mem_profile_plot(mem, title):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    df = DataFrame(mem, columns=["memory", "timestamp"])
    df["timestamp"] = to_datetime(df.timestamp)
    df["timestamp"] -= df.timestamp.min()
    df.set_index("timestamp").plot(ax=ax)
    ax.set_title(title + "\nmemory usage")
    return ax


def benchmark(model_path):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.zeros((1, 3, 256, 256), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    max_memory_usage_mb = None
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        if i == 0:
            memprof_onx2 = memory_usage((session.run, ([], {input_name: input_data},)),
                                        timestamps=True, interval=0.01)
            max_memory_usage_mb = np.max([element[0] for element in memprof_onx2])
            # mem_profile_plot(memprof_onx2, "session.run time")
            # plt.show()

        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Max Memory Usage, (MB): {max_memory_usage_mb}")
    print(f"Avg: {total:.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--model_onnx_path", required=False, type=str,
                        default="optimized_models_onnx/NAFNetLocal-width16-fp=int8.onnx",
                        help="Path to an optimized onnx model")
    args = parser.parse_args()
    benchmark(args.model_onnx_path)
    # memprof_skl = memory_usage((clr.predict, (X_test,)), timestamps=True, interval=0.01)

