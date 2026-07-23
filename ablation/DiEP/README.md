# DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning
![DiEP](framework.png)

Official Pytorch implementation of the differentiable expert pruning methods as presented in:

**DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning (Neurips 2025)**</br>
Sikai Bai, Haoxi Li, Jie ZHANG, Zicong Hong, Song Guo

[Paper](https://arxiv.org/abs/2509.16105)

## Installation
Step 1: Create a new conda environment:
```bash
conda create -n moe_pruning python=3.10
conda activate moe_pruning
```
Step 2: Install relevant packages

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install transformers==4.38.1 accelerate datasets fire tqdm pytablewriter
```

## Dataset Preparation
1. C4: Please download first part of the C4 training data `c4-train.00000-of-01024.json` from [allenai/c4](https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz).
2. MATH: You can use our pre-built calibration set in `./data/math_pretrain_style.json`. To reproduce our construction, please download the training set of [MATH](https://github.com/hendrycks/math) and use our [script](data/math_calib_construction.py).
3. Alpaca: In our project, generation speed is benchmarked using Alpaca dataset. Please download `alpaca_data_cleaned.json` from [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned).
4. Finally, please organize the calibration datasets as follows.
```
./data
|-- __init__.py
|-- alpaca_data_cleaned.json
|-- build.py
|-- c4-train.00000-of-01024.json
|-- dataset.py
|-- math_calib_construction.py
`-- math_pretrain_style.json
```

## Pruning
Usage:
```
python main.py [-h] --method {nas_pruning,load_pruning,layerwise_pruning,progressive_pruning,dynamic_skipping} [--r R] --calib_set {c4,math} --model_path MODEL_PATH [--output_path OUTPUT_PATH] [--max_block_size MAX_BLOCK_SIZE] [--n_blocks_for_stat N_BLOCKS_FOR_STAT] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--seed SEED] [--use_flash_attention_2]
```
Options:
- `-h, --help`: show this help message and exit
- `--method {nas_pruning,load_pruning,layerwise_pruning,progressive_pruning}`: Supported pruning methods:Nas_pruning, Load_pruning, Layerwise Pruning, Progressive Pruning, Dynamic Skipping
- `--r R`: Number of experts to preserve
- `--calib_set {c4,math}`: Supported calibration datasets: C4, MATH
- `--model_path MODEL_PATH`: Path to model to prune
- `--output_path OUTPUT_PATH`: Output path (pruned model, pruning results, etc.)
- `--max_block_size MAX_BLOCK_SIZE`: Maximal sequence length of each sample in calibration set
- `--n_blocks_for_stat N_BLOCKS_FOR_STAT`: Number of sequences in calibration set. If set to 0 or negative, the whole dataset will be used
- `--batch_size BATCH_SIZE`: Batch size for model inference
- `--num_workers NUM_WORKERS`: Number of workers in dataloader
- `--seed SEED`: Random seed for reproduction
- `--use_flash_attention_2`: If set, Flash Attention 2 will be used
- `--lambda_coeff`: the coefficient weight between cross-entropy loss and reconstruction regularization term
- `--lr`: the learning rate for alpha and beta updates
- `--num_train_epochs`: the number of epochs for alpha and beta updates
- `--eval_tasks`: the evaluation tasks that are splited with comma, e.g.,"mmlu,rte"
### One Example for Expert Pruning
You can perform differentiable expert pruning by running:
```
python main.py --method nas_pruning --r 4 --calib_set c4 --model_path Mixtral-8x7B-v0.1 --output_path ./output/
```
You can load cached pruning results and directly perform evaluation by running:
```
python main.py --method load_pruning --r 4 --calib_set c4 --model_path Mixtral-8x7B-v0.1 --output_path ./output/ --cache_pruned_file /path/to/NAS_32_4.txt
```

## Evaluation

### LM Harness Evaluation
We use the [EleutherAI LM Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/2a47159caff00135b026f724ace2a2011f3c7621) framework to evaluate the performance of pruned LLMs, and it has been integrated into our code. 

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.


## Citation

If you find our paper and code useful in your research, please cite

```
@article{bai2025diep,
  title={DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning},
  author={Bai, Sikai and Li, Haoxi and Zhang, Jie and Hong, Zicong and Guo, Song},
  journal={Advances in neural information processing systems},
  year={2025}
}
```

## Contact me
If you have any questions about this code or paper, please contact me at
[Sikai Bai](whitesk1973@gmail.com).

## Acknowledgement:

This code is mainly based on [NAEE](https://github.com/Lucky-Lance/Expert_Sparsity) and [LM Harness Evaluation](https://github.com/EleutherAI/lm-evaluation-harness).
