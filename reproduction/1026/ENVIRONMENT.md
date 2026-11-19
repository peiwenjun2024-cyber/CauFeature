# Verified environment snapshot

Queried on 2026-10-13. This records the current environment; historical run environments were not independently verified.

| Component | Configuration |
|---|---|
| OS | Ubuntu 22.04.3 LTS, x86_64 |
| Host CPU | 2 × Intel Xeon Gold 6430; 64 physical cores, 128 logical threads |
| Container CPU quota | 16 vCPUs (cpu.max: 1600000 100000) |
| Container memory limit | 120 GiB (128849018880 bytes) |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB reported device memory |
| NVIDIA driver | 560.35.03 |
| Python | 3.12.3 |
| TensorFlow | 2.20.0 |
| Keras | 3.12.0 |
| TensorFlow CUDA build | 12.5.1 |
| TensorFlow cuDNN build | 9 |
| System CUDA compiler (nvcc) | 12.1, V12.1.105 |
| NumPy | 1.26.4 |
| SciPy | 1.11.4 |
| pandas | 2.2.3 |
| scikit-learn | 1.4.2 |

TensorFlow build versions and the system CUDA compiler version are distinct. Host CPU topology does not represent the container's CPU quota.
