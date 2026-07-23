import os
import os.path as osp
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import pdb
import logging
import argparse
from argparse import Namespace
from datetime import datetime
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable, make_table
import lm_eval
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from model.modeling_mixtral import MixtralForCausalLM
import json
from method import METHODS
from data import DATASETS, build_calib_loader


logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default="nas_pruning", #required=True, load_pruning
                        choices=list(METHODS.keys()),
                        help=' '.join(['Supported pruning methods:'] + list(METHODS.keys())))
    parser.add_argument('--r', type=int, default=4,
                        help='Number of experts to preserve')
    parser.add_argument('--calib_set', type=str, default='c4', #required=True,
                        choices=list(DATASETS.keys()),
                        help=' '.join(['Supported calibration datasets:'] + list(DATASETS.keys())))
    parser.add_argument('--model_path', type=str, default="Mixtral-8x7B-v0.1", #required=True,
                        help='Path to model to prune')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='Output path (pruned model, pruning results, etc.)')
    parser.add_argument('--max_block_size', type=int, default=2048,
                        help='Maximal sequence length of each sample in calibration set')
    parser.add_argument('--n_blocks_for_stat', type=int, default=64,
                        help='Number of sequences in calibration set. If set to 0 or negative, the whole dataset will be used')
    parser.add_argument('--num_train_epochs', type=int, default=11, 
                        help='Number of trining epochs for alpha and beta updates')
    parser.add_argument('--lambda_coeff', type=float, default=0.12, 
                        help='Learning rate for alpha and beta updates')
    parser.add_argument('--lr', type=float, default=0.005, 
                        help='Learning rate for alpha and beta updates')
    parser.add_argument('--eval_tasks', type=str, default="mmlu,rte", #0.00005, #0.1
                        help='the evaluation tasks that are splited with comma, e.g.,mmlu,rte')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for model inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers in dataloader')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproduction')
    parser.add_argument('--cache_pruned_file', type=str, default=None,
                        help='Optional path to a user-provided pruning cache')
    parser.add_argument('--use_flash_attention_2', action='store_true',
                        help='If set, Flash Attention 2 will be used')

    return parser.parse_args()


def main(args: Namespace):
    logger.info(f'Arguments: {args}')
    
    if args.model_path.endswith('/'):
        args.model_path = args.model_path[:-1]
    model_name = args.model_path.split('/')[-1]

    if args.method.endswith('_pruning'): #eg: layerwise_pruning
        assert args.r is not None, 'Using pruning methods, argument `r` is required'
        save_path = osp.join(
            args.output_path, f'{model_name}_{args.method}_r{args.r}_{args.calib_set}_{args.n_blocks_for_stat}_{args.lambda_coeff}_{args.num_train_epochs}_{"fatt2_" if args.use_flash_attention_2 else ""}{datetime.now().strftime("%Y%m%d-%H%M%S")}') #args.n_blocks_for_stat:128, args.calib_set:c4
    else:
        if args.r is not None:
            logger.warn(f'Not using pruning methods, argument `r` is not used')
        save_path = osp.join(
            args.output_path, f'{model_name}_{args.method}_{args.calib_set}_{args.n_blocks_for_stat}_{"fatt2_" if args.use_flash_attention_2 else ""}{datetime.now().strftime("%Y%m%d-%H%M%S")}') 
    
    logger.info(f'Save path: {save_path}')
    os.makedirs(save_path, exist_ok=False)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    #tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_path,
    #     #load_in_8bit=True,
    #     device_map='auto',
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2" if args.use_flash_attention_2 else None
    # )
    model = MixtralForCausalLM.from_pretrained(
        args.model_path,
        #load_in_8bit=True,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.use_flash_attention_2 else None
    )
    
    calib_loader, test_loader = build_calib_loader(args.calib_set, tokenizer, args.max_block_size,
                                      args.n_blocks_for_stat, args.batch_size, args.num_workers, args.seed)
    
    
    print('method:', args.method)
    
    start_time = datetime.now()
    if args.method == "nas_pruning":
        
        model = METHODS[args.method](model, test_loader, test_loader, args, pruned_layers=range(0,32), num_train_epochs=args.num_train_epochs, lambda_coeff=args.lambda_coeff, lr=args.lr) #calib_loader
    elif args.method == "load_pruning":
        model = METHODS[args.method](model, args, test_loader, args, pruned_layers=range(0,32)) #calib_loader
    else:
        model, info = METHODS[args.method](model, test_loader, args)
        
    end_time = datetime.now()


    
    elapsed_time = end_time - start_time
    elapsed_time_in_hours = elapsed_time.total_seconds() / 3600

    print(f"模型运行时间: {elapsed_time_in_hours:.4f} 小时")
    
    
    model.experts_frequency = {}
    
    lm = HFLM(pretrained=model, tokenizer=tokenizer, dtype=torch.bfloat16, max_length=tokenizer.model_max_length,
              batch_size=args.batch_size, trust_remote_code=True)
    
    eval_tasks=args.eval_tasks.split(',')
    results = lm_eval.simple_evaluate(model=lm, tasks=eval_tasks,  
            task_manager=lm_eval.tasks.TaskManager(),) #, 'hellaswag', "arc_challenge", "arc_easy", "mmlu","boolq", "rte", "openbookqa", #, 'winogrande', 'hellaswag',"arc_challenge", "arc_easy" ,"boolq", "rte", "openbookqa"  ##"mmlu","boolq", "rte", "openbookqa", "gsm8k", "wikitext"

    #############
    if results is not None:
        dumped = json.dumps(
            results, indent=2, default=handle_non_serializable, ensure_ascii=False
        )
        eval_result=make_table(results)
        print(eval_result)
        
        log_file = osp.join(save_path, 'log.txt')
        with open(log_file, 'a') as file:
            file.write(str(model))
            file.write(eval_result)
            
        if "groups" in results:
            print(make_table(results, "groups"))
            with open(log_file, 'a') as file:
               file.write(make_table(results, "groups"))
            
    # model.save_pretrained(save_path)
    # tokenizer.save_pretrained(save_path)
    # torch.save((args, info), osp.join(save_path, 'pruning_info.pt'))


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    args = parse_args()
    main(args)
