python train.py hydra.job.chdir=False
# python -m scripts.optimize_models_onnx hydra.job.chdir=False
# python benchmark.py hydra.job.chdir=False
# python -m scripts.run_onnx_inference -o optimized_models_onnx/NAFNetLocal-optimized-fp=int8.onnx