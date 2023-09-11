# Multivariate Cryptocurrency Price Forecasting using Distributional Variational AutoEncoder

This repository is the official implementation of Multivariate Cryptocurrency Price Forecasting using Distributional Variational AutoEncoder (DMFVAE) and benchmark models with pytorch. 

> **_NOTE:_** This repository supports [WandB](https://wandb.ai/site) MLOps platform!

## Training & Evaluation 

### Proposed Method
```
python main.py --model <model>
```   
- `<model>` options: `GLD_finite`, `GLD_infinite`, `LSQF`, `ExpLog`
- detailed configuration files can be found in `configs` folder

### Benchmark Methods
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

## Citation
