# GIT-BO: High-Dimensional Bayesian Optimization with Tabular Foundation Models

This repository is the official implementation of GIT-BO: High-Dimensional Bayesian Optimization with Tabular Foundation Models

# Requirements

## Hardware Requirement
1. GPU: To run high-dimensional (D>100) problems with `GIT-BO`, please ensure you have access to a GPU with at least 20 GB of GPU memory. Otherwise, it is possible that you will run into the CUDA OUT_OF_MEMORY error, and we cannot ensure the algorithm's success.
2. System: GIT-BO is set to be run on x86_64 architecture, Linux Ubuntu 22.04


## Installations
1. To run the main `GIT-BO` algorithm with `TURBO` and `SAASBO` baseline algorithms, make a conda/mamba environment and install the requirements:
```
pip install -r GITBO_requirements.txt
```

**Note**: If running on a shared cluster, the installation of LassoBench might face issues. Make sure that you have connection to internet and have module such as openblas loaded. We advise skipping the installation of LassoBench if needed.

2. To run the `HESBO` and `ALEBO` baseline algorithms, make a SEPARATE conda/mamba environment and install the requirements as they require a specific [Ax](https://ax.dev/) version:
```
pip install -r ALEBO_HESBO_requirements.txt
```

## Download open-source model and executables for GIT-BO
1. **GIT-BO Essential**: GIT-BO requires download the tabular foundation model TabPFNv2.
As we change some of the tabpfn code for `GIT-BO`, please use the provided tabpfn code here from this repo instead of directly downloading the tabpfn official repo. Additionally, `GIT-BO` uses the [`TabPFN-v2-reg model`](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/tree/main) (the `tabpfn-v2-regressor.ckpt`) as its model. Please download this checkpoint and put it under `tabpfn/model/` if it is not already there. 

2. **Mazda benchmark problem**: Download the Mazda problem executables from https://ladse.eng.isas.jaxa.jp/benchmark/ and move the bin executables in `Mazda_CdMOBP/bin/Linux/` under `TestProblems_Utils/Mazda_Data/bin/` if it is not already there. 
3. **MOPTA08 Car benchmark problem**: Download the MOPTA08 problem executables from https://leonard.papenmeier.io/2023/02/09/mopta08-executables.html and put them under `TestProblems_Utils/Mopta_Data/` if it is not already there. 
4. **SVM benchmark problem**: Download the `slice_localization_data.csv` from https://github.com/XZT008/Standard-GP-is-all-you-need-for-HDBO/tree/main/benchmark/data and put it under `TestProblems_Utils/` if it is not already there. 

With all the executables and data properly installed, the final directory structure should look like this:
```
GITBO/
├─ HesBO/
├─ tabpfn/
│   ├─ model/
│   │   ├─ tabpfn-v2-regressor.ckpt
│   │   └─ ......
│   └─ ......
├─ TestProblems_Utils/
│   ├─ Mazda_Data/
│   │   ├─ bin/
│   │   │  ├─ mazda_mop
│   │   │  └─ mazda_mop_sca
│   │   └─ ......
│   ├─ Mopta_Data/
│   │   ├─ mopta08_armhf.bin
│   │   ├─ mopta08_elf32.bin
│   │   ├─ mopta08_elf64.bin
│   │   └─ ......
│   ├─ slice_localization_data.csv
│   └─ ......
└─ ......
```
# Running experiment

## GIT-BO

### Main algorithm
The Python script for running `GIT-BO` is `run_GITBO.py`, and the bash script to run the full experiment is `GITBO_script.sh`. The algorithm can be run as follows:
```
python run_GITBO.py --ITER 10 --FUNC_NAME Ackley --DIM 100
```
- The `ITER` flag determines the total number of iterations the algorithm runs. For the full experiment, we set it at 200 (but this will require more GPU memory).
- The `FUNC_NAME` flag determines the problem we are optimizing. See the table in Section **Benchmark Problems** to see all benchmark problems that are available.
- The `DIM` flag determines the problem dimension.

### Ablation study
To replicate our ablation study of vanilla TabPFN v2 + EI or TabPFN v2 + TS. The algorithm can be run as follows:
```
python run_GITBO.py --ITER 10 --FUNC_NAME Ackley --DIM 100 --ACQ EI --GI_SUBSPACE False
```
- The `ACQ` flag determines the acquisition function. There are three options: `EI` as expected improvement and `ThompsonSampling` as Thompson Sampling.
- The `GI_SUBSPACE` flag determines whether the gradient-informed subspace is employed or not. The default for GIT-BO is `True`

### Parameter Sweep
To replicate our ablation study of parameter sweep, use the `RANK_R` flag to change the dimension of subspace (subspace rank). The algorithm can be run as follows:
```
python run_GITBO.py --ITER 10 --FUNC_NAME Ackley --DIM 100 --RANK_R 5
```

## Baseline Algorithms
### TURBO: 
The Python script for running `TURBO` is `run_TURBO.py`. The code is taken from: https://botorch.org/docs/tutorials/turbo_1/. The algorithm can be run as follows:
```
python run_TURBO.py --ITER 5 --FUNC_NAME Ackley --DIM 100
```

### SAASBO: 
The Python script for running `SAASBO` is `run_SAASBO.py`. The code is taken from: https://botorch.org/docs/tutorials/saasbo/. The algorithm can be run as follows:
```
python run_SAASBO.py --ITER 5 --FUNC_NAME Ackley --DIM 100
```

### HESBO
We use the implementation of the original HESBO GitHub repo: https://github.com/aminnayebi/HesBO with its settings to run experiment jobs, while interfacing with our benchmark test functions. The algorithm can be run as follows:
```
python HesBO/experiments.py HeSBO [first_experiment_trial_id] [last_experiment_trial_id] GITBO [num_of_iterations] [low_dim] [high_dim] [num_of_initial_sample] [noise_variance] [REMBO_variant]
```
Here is an example of running HESBO on 100 dim noise-free Ackley:
```
python HesBO/experiments.py HeSBO 1 1 PFN 200 10 100 100 0 Ackley
```


### ALEBO:
We use the implementation of the original ALEBO GitHub repo: https://github.com/facebookresearch/alebo/blob/main/quickstart.ipynb, while interfacing with our benchmark test functions. The algorithm can be run as follows:
```
python _ALEBO.py --ITER 3 --DIM 100 --FUNC_NAME Ackley --d 10 
```
- The `ITER` flag determines the total number of iterations the algorithm runs. For the full experiment, we set it at 200 (but this will require more GPU memory).
- The `FUNC_NAME` flag determines the problem we are optimizing. See the table in Section **Benchmark Problems** to see all benchmark problems that are available.
- The `DIM` flag determines the problem dimension.
- The `d` flag determines the latent dimension. The options are 10 and 20.

## Benchmark Problems
The 23 benchmark problems collected are placed in `TestProblems_Utils` folder. The objective evaluation can be used as shown:

```
from TestProblems_Utils import *

# Create the search variable X
X = torch.rand(10,100)

# Define the function using these arguments: Problem Name, Problem Dimension, and indicating if there are constraints values (no for all the GIT-BO experiments)
Function = Function_selector("Ackley", 100, False)

# Evaluation outputs the constraints and objective (Y)
_, Y = Function.evaluate(X)

print(f"X: {X}, f(X): {Y}")
```

The following table shows the problem name supported by the GIT-BO code, its dimension, type, and source. 
| Problem Name | Problem Dimension | Problem Type | Source Reference |
| ------------- | ------------- | ------------- | ------------- |
| Ackley | N (scalable)  | Synthetic  | [Botorch](https://github.com/pytorch/botorch/blob/main/botorch/test_functions/synthetic.py) |
| Levy | N (scalable)  | Synthetic  | [Botorch](https://github.com/pytorch/botorch/blob/main/botorch/test_functions/synthetic.py) |
| Powell | N (scalable)  | Synthetic  | [Botorch](https://github.com/pytorch/botorch/blob/main/botorch/test_functions/synthetic.py) |
| Griewank | N (scalable)  | Synthetic  | [Botorch](https://github.com/pytorch/botorch/blob/main/botorch/test_functions/synthetic.py) |
| DixonPrice | N (scalable)  | Synthetic  | [Botorch](https://github.com/pytorch/botorch/blob/main/botorch/test_functions/synthetic.py) |
| Rosenbrock | N (scalable)  | Synthetic  | [Botorch](https://github.com/pytorch/botorch/blob/main/botorch/test_functions/synthetic.py) |
| Michalewicz | N (scalable)  | Synthetic  | [Botorch](https://github.com/pytorch/botorch/blob/main/botorch/test_functions/synthetic.py) |
| Rastrigin | N (scalable)  | Synthetic  | [Botorch](https://github.com/pytorch/botorch/blob/main/botorch/test_functions/synthetic.py) |
| StyblinskiTang | N (scalable)  | Synthetic  | [Botorch](https://github.com/pytorch/botorch/blob/main/botorch/test_functions/synthetic.py) |
| Rover | N (scalable)  | Real-World  | [Ensemble-Bayesian-Optimization](https://github.com/zi-w/Ensemble-Bayesian-Optimization/tree/master) |
| CEC2020_p34 | 118  | Real-World  | [CEC2020](https://github.com/P-N-Suganthan/2020-Bound-Constrained-Opt-Benchmark) |
| CEC2020_p35 | 153  | Real-World  | [CEC2020](https://github.com/P-N-Suganthan/2020-Bound-Constrained-Opt-Benchmark) |
| CEC2020_p36 | 158  | Real-World  | [CEC2020](https://github.com/P-N-Suganthan/2020-Bound-Constrained-Opt-Benchmark) |
| CEC2020_p37 | 126  | Real-World  | [CEC2020](https://github.com/P-N-Suganthan/2020-Bound-Constrained-Opt-Benchmark) |
| CEC2020_p38 | 126  | Real-World  | [CEC2020](https://github.com/P-N-Suganthan/2020-Bound-Constrained-Opt-Benchmark) |
| CEC2020_p39 | 126  | Real-World  | [CEC2020](https://github.com/P-N-Suganthan/2020-Bound-Constrained-Opt-Benchmark) |
| LassoSyntMedium | 100  | Synthetic  | [LassoBench](https://github.com/ksehic/LassoBench) |
| LassoSyntHigh | 300  | Synthetic  | [LassoBench](https://github.com/ksehic/LassoBench) |
| LassoSyntDNA | 180  | Real-World  | [LassoBench](https://github.com/ksehic/LassoBench) |
| MOPTA08Car | 124  | Real-World  | [Mopta08 Executables](https://leonard.papenmeier.io/2023/02/09/mopta08-executables.html) |
| Mazda_SCA | 148  | Real-World  | [Mazda Bechmark Problem](https://ladse.eng.isas.jaxa.jp/benchmark/) |
| Mazda | 222  | Real-World  | [Mazda Bechmark Problem](https://ladse.eng.isas.jaxa.jp/benchmark/) |
| SVM | 388  | Real-World  | [Standard-GP-is-all-you-need-for-HDBO](https://github.com/XZT008/Standard-GP-is-all-you-need-for-HDBO) |





# License
The GIT-BO code will be MIT licensed.
