# Cryptocurrency Price Forecasting using Variational AutoEncoder with Versatile Quantile Modeling

This repository is the official implementation of Multivariate Cryptocurrency Price Forecasting using CryptoVAE and benchmark models with pytorch. 

> **_NOTE:_** This repository supports [WandB](https://wandb.ai/site) MLOps platform!

## Training & Evaluation 

### 1. Training Proposed Method
```
python main.py --model <model>
```   
- `<model>` options: `GLD_finite`, `GLD_infinite`, `LSQF`, `ExpLog`
- detailed configuration files can be found in `configs` folder

### 2. Training Benchmark Methods
```
python main.py --model 'TLAE'
python main.py --model 'ProTran'
python benchmarks/deepar.py 
python benchmarks/gp_copula.py 
python benchmarks/mqrnn.py 
python benchmarks/sqf_rnn.py 
python benchmarks/tft.py
python benchmarks/benchmark_eval.py --model 'TFT' --tau 1
```
- detailed configuration files can be found in `configs` folder for `TLAE` and `ProTran`
- pretrained weights for `DeepAR`, `GP-copula`, `MQRNN`, `SQF-RNN`, `TFT` can be found in `/assets` folder

### codes for evaluation
- step-by-step evaluation for our proposed method: `infernce.py`
- step-by-step evaluation for benchmark methods: `benchmarks/benchmark_eval.py`   

## Directory and Codes
```
.
+-- assets (includes visualization results and pretrained weights of benchmark methods)
+-- benchmarks (includes codes for training benchmark methods)
+-- config (includes detailed configuration files)
+-- module 
+-- inference.py
+-- main.py
+-- LICENSE
+-- README.md
```

## Citation
```

```