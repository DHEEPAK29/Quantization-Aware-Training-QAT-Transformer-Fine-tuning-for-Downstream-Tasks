# Quantization-Aware Training (QAT): Transformer Fine-tuning for Downstream Tasks

This project provides a complete pipeline to perform **Quantization-Aware Training (QAT)** on the **Microsoft Phi-2** model. QAT simulates low-precision hardware constraints (INT8/INT4) during training, forcing the model's weights to cluster around "safe" grid values. This process minimizes accuracy loss when the model is deployed on edge hardware.

---

## 1. Project Structure

```text
qat_project/
├── model.py         # Phi-2 configuration and QAT preparation
├── train.py         # Training loop with layer-wise weight logging
├── visualize.py     # Script to visualize weight clustering (histogram evolution)
└── run.sh           # Bash script for automated execution
```

---

## 2. Implementation Components

### `model.py`
Uses `torchao` to inject `FakeQuantize` modules into Phi-2. These modules use the **Straight-Through Estimator (STE)**, which rounds values during the forward pass but preserves gradients during the backward pass. 

### `train.py`
Executes the fine-tuning phase. As the model trains, it logs the distribution of weights from the first transformer layer (`model.model.layers[0]`). You will observe these weights "parking" on the valid integer values supported by the quantization grid.

### `viz.py`
Generates histograms comparing the initial FP32 weight distribution against the post-training clustered distribution, showing the transition from smooth Gaussian curves to sharp, grid-aligned spikes. 

---

## 3. Pipeline Checkpoints

* **The STE "Detour":** The **Straight-Through Estimator (STE)** allows gradients to bypass the "rounding" during backpropagation, effectively enabling the optimizer to treat the rounding as an identity function while the forward pass still experiences the error.
* **Weight Clustering:** QAT effectively "fits" the high-precision weights into a discrete, hardware-friendly format. The visualization will confirm that your model has successfully adapted to the constraints of INT8/INT4 hardware.
* **Deployment:** After training, use `torchao.quantization.convert(model)` to replace the simulation modules with native integer-arithmetic kernels, ready for export to `ExecuTorch` or `vLLM`.

---

## 4. Industrial Context
This pipeline is based on production-grade standards used by silicon vendors to prepare LLMs for on-device inference. By using QAT instead of Post-Training Quantization (PTQ), you recover the accuracy lost to precision reduction, ensuring high-performance inference on mobile NPUs and edge accelerators.