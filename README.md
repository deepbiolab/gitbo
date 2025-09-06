# GIT-BO: High-Dimensional Bayesian Optimization with Tabular Foundation Models

# Requirements

## Hardware Requirement
1. GPU: To run high-dimensional (D>100) problems with `GIT-BO`, please ensure you have access to a GPU with at least 20 GB of GPU memory. Otherwise, it is possible that you will run into the CUDA OUT_OF_MEMORY error, and we cannot ensure the algorithm's success.
2. System: GIT-BO is set to be run on x86_64 architecture, Linux Ubuntu 22.04


## Installations
1. To run the main `GIT-BO` algorithm with `TURBO` and `SAASBO` baseline algorithms, make a conda/mamba environment and install the requirements:
```
pip install -r GITBO_requirements.txt
```

## Download open-source model and executables for GIT-BO
1. **GIT-BO Essential**: GIT-BO requires download the tabular foundation model TabPFNv2.
As we change some of the tabpfn code for `GIT-BO`, please use the provided tabpfn code here from this repo instead of directly downloading the tabpfn official repo. Additionally, `GIT-BO` uses the [`TabPFN-v2-reg model`](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/tree/main) (the `tabpfn-v2-regressor.ckpt`) as its model. Please download this checkpoint and put it under `tabpfn/model/` if it is not already there. 

With all the executables and data properly installed, the final directory structure should look like this:
```
GITBO/
├─ tabpfn/
│   ├─ model/
│   │   ├─ tabpfn-v2-regressor.ckpt
│   │   └─ ......
│   └─ ......
└─ ......
```
# Running experiment

## GIT-BO

### Main algorithm
The Python script for running `GIT-BO` is `run_GITBO.py`. The algorithm can be run as follows:
```
python run_GITBO.py --ITER 10 --FUNC_NAME Ackley --DIM 100
```
- The `ITER` flag determines the total number of iterations the algorithm runs. For the full experiment, we set it at 200 (but this will require more GPU memory).
- The `DIM` flag determines the problem dimension.


