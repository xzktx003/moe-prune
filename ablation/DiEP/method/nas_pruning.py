from tqdm import tqdm
from argparse import Namespace
import logging
import pdb
import torch
import math
import re
from torch.utils.data import DataLoader
from transformers.models.mixtral import modeling_mixtral # MixtralForCausalLM#
#import transformers
from model.modeling_mixtral import MixtralForCausalLM
import torch.nn as nn
from model import PrunableMixtralSparseMoeBlockWrapper, NASMixtralSparseMoeBlockWrapper
import torch.optim as optim
import torch.nn.functional as F
from transformers import (
    get_scheduler,
    SchedulerType,
)
from data import CacheDataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
logger = logging.getLogger(__name__)

class NAS_PrunableMixtralSparseMoe:
    def __init__(self, model, train_loader, test_loader, prev=6,layers_threshold=[0,1], lr_scheduler_type="linear", lr=0.005, weight_decay=5e-2, num_train_epochs=9, lambda_coeff=0.6): #30 45,
        if isinstance(model, MixtralForCausalLM) or isinstance(model, modeling_mixtral.MixtralForCausalLM): #lr=0.005
            self.model = model
        else: 
            self.model = None
        
        self.lr=lr
        print("Initializing NAS Pruning: lr: {}".format(lr))
        self.lr_scheduler_type=lr_scheduler_type
        self.weight_decay = weight_decay

        self.train_loader = test_loader
        self.test_loader = test_loader
        
        gradient_accumulation_steps=1
        
        self.num_update_steps_per_epoch = math.ceil(len(self.train_loader) / gradient_accumulation_steps)
        self.num_train_epochs=num_train_epochs
        
        self.layers_threshold=layers_threshold
        self.layers_nodes = [torch.tensor(prev) for _ in self.layers_threshold] #(self.model.model.layers)]
        
        self.layers_threshold_index ={layer: index for index, layer in enumerate(self.layers_threshold)}
        
        self.total_nodes = int(sum([node.item() for node in self.layers_nodes]))
        self.num_experts=8
        self.prev = prev
        self.lambda_coeff=lambda_coeff
        self.epsilon=1e-5
        
    def normalize(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    def init_model(self):
        self.model.eval()
        for l, layer in enumerate(self.model.model.layers):
            layer.block_sparse_moe = NASMixtralSparseMoeBlockWrapper(layer.block_sparse_moe)
            
    def get_alpha_value(self, epoch=0):
        if epoch==0:
            epoch=self.num_train_epochs
        save_path = './outputs/concatenated_alpha_layer0_9_epoch{}.pth'.format(epoch)
        
        concatenated_alpha_gate = torch.cat([torch.abs(layer.block_sparse_moe.alpha_gate)*(layer.block_sparse_moe.layer_weight) for l, layer in list(enumerate(self.model.model.layers)) if l in self.layers_threshold])
        torch.save(concatenated_alpha_gate, save_path)
        return concatenated_alpha_gate
    
    def prune_experts(self, concatenated_alpha_gate):
        values, indices = torch.sort(concatenated_alpha_gate)
        topk=(self.model.num_experts-self.prev)*len(self.layers_threshold)

        count=0
        for index in indices:
            if count>= topk:
                break
            layer=self.layers_threshold[index//self.model.num_experts]
            expert=index%self.model.num_experts
            
            if self.model.model.layers[layer].block_sparse_moe.experts_to_drop is None:
                self.model.model.layers[layer].block_sparse_moe.experts_to_drop =[expert.item()]
                
            elif len(self.model.model.layers[layer].block_sparse_moe.experts_to_drop) >= self.model.num_experts-self.model.model.layers[layer].block_sparse_moe.top_k:
                continue
            else:
                self.model.model.layers[layer].block_sparse_moe.experts_to_drop.append(expert.item())
            count+=1
       
        for l, layer in list(enumerate(self.model.model.layers)):
            if l not in self.layers_threshold or layer.block_sparse_moe.experts_to_drop is None:
                continue
            
            block = layer.block_sparse_moe
            block.prune()
            #layer.block_sparse_moe =  block.model  #layer.block_sparse_moe.model
            print('layer:', l, block.experts_to_drop)
            
            #print('layer:', l, block)
        
        for l, layer in enumerate(self.model.model.layers):
            layer.block_sparse_moe = layer.block_sparse_moe.model
            
    def train(self):
        
        self.model.eval()
        for l, layer in enumerate(self.model.model.layers):
            
            if l<=16: 
                weight=2.0
            else:
                weight=0.2
            if l in self.layers_threshold:
                layer.block_sparse_moe = NASMixtralSparseMoeBlockWrapper(layer.block_sparse_moe, weight=weight, search=True)
            else:
                layer.block_sparse_moe = NASMixtralSparseMoeBlockWrapper(layer.block_sparse_moe)
            layer.block_sparse_moe.cache_X = True
            layer.block_sparse_moe.cache_Z = True
            layer.block_sparse_moe.Cache=True

        cache_input=CacheDataset()
        for l, layer in enumerate(self.model.model.layers):
            if  l in self.layers_threshold:
                print('layer id:', l, layer.block_sparse_moe.layer_weight)
        with torch.inference_mode():
            for i, batch in enumerate(tqdm(self.train_loader, desc='Model forwarding on sample set...')):
                
                model_inputs = self.model.prepare_inputs_for_generation(**batch)
                cache_input.append(input_ids=model_inputs)
                
                outputs = self.model(**model_inputs)
                assert outputs is not None
        
        params_update = []
        for name, param in self.model.model.named_parameters():
            if "alpha_gate" in name:
                params_update.append(param) 
        
        
        alpha_optimizer = torch.optim.SGD(
            params=params_update,
            lr=self.lr,
            momentum=0.8,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            alpha_optimizer, 
            step_size=40, 
            gamma=0.5)
        del params_update
        
        weight_params_update = []
        

        for name, param in self.model.model.named_parameters():
            
            if "layer_weight" in name:   
                weight_params_update.append(param)
     
        weight_optimizer= optim.SGD(weight_params_update, lr=self.lr) # 
        
        weight_lr_scheduler = get_scheduler(
            name = self.lr_scheduler_type,
            optimizer=weight_optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_train_epochs * self.num_update_steps_per_epoch,
        )
        del weight_params_update
        

        starting_epoch=0
        losses= {}
        mse_loss = nn.MSELoss(reduction='mean')
        l1_loss = nn.L1Loss()
        for l, layer in list(enumerate(self.model.model.layers)):
            if l in self.layers_threshold:
                layer.block_sparse_moe.ce_loss = True
            
            
        
        for epoch in range(starting_epoch, self.num_train_epochs):
            #loss = 0
            loss_ce=None
            for name, param in self.model.model.named_parameters():
                if epoch%3<=1 and "alpha_gate" in name:   
                    param.requires_grad_(True)
                elif epoch%3>1 and "layer_weight" in name: 
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            
            cache_size = len(cache_input.input_ids)
            num_batches=0
            total_loss, total_loss_ce, total_loss_div = 0, 0, 0
            for index in range(cache_size):
                input_data = cache_input.input_ids[index]
                input_data['labels'] = input_data['input_ids']
                
                output = self.model(**input_data)
                
                loss_ce = output.loss
                loss_div=0
                for l, layer in list(enumerate(self.model.model.layers)):
                    
                    block = layer.block_sparse_moe #= NASMixtralSparseMoeBlockWrapper(layer.block_sparse_moe, layer=l)
                    block.cache_logits = False
                    block.cache_X = False
                    block.cache_Z = False
                    block.Cache=False
                    if not hasattr(block, 'cache_space'):
                        continue
                
                    if l not in self.layers_threshold:
                        continue
                
                    
                    hidden_states, final_hidden_states = block.cache_space.Xs[index], block.cache_space.Zs[index]
                    

                    hidden_states = hidden_states.to(
                        device=block.gate.weight.data.device, non_blocking=True)
                    final_hidden_states = final_hidden_states.to(
                        dtype=torch.float64, device=block.gate.weight.data.device, non_blocking=True)
                    
                    final_hidden_states_nas = block(hidden_states.unsqueeze(0), traintime=True)
                    
                    if final_hidden_states_nas is not None: 
                        
                        layer_loss = torch.norm(final_hidden_states -
                                      final_hidden_states_nas.squeeze(0).to(torch.float64))
                        try:
                            loss_div += layer_loss
                        except:
                            loss_div += layer_loss.to(loss_div.device)
                        if epoch == self.num_train_epochs-1:    
                            if l not in losses.keys():
                                losses[l] = 0
                            losses[l] += layer_loss.item()
                            
                        
             
                num_batches +=1
                if epoch <1:
                    loss = loss_ce+self.lambda_coeff*loss_div
                
                loss = (1.0/(epoch+1))*loss_ce+self.lambda_coeff*loss_div

                #loss = 0.2*loss_ce+self.lambda_coeff*loss_div
                if epoch%3<=1:
                    alpha_optimizer.zero_grad()
                    loss.backward()
                    alpha_optimizer.step()
                    lr_scheduler.step()
                    with torch.no_grad():
                        for name, param in self.model.model.named_parameters():
                            if "alpha_gate" in name:
                                param.clamp_(self.epsilon, 5.0)
                else:
                    weight_optimizer.zero_grad()
                    loss.backward()
                    weight_optimizer.step()
                    weight_lr_scheduler.step()
                    with torch.no_grad():
                        for name, param in self.model.model.named_parameters():
                            if "layer_weight" in name:
                                param.clamp_(self.epsilon, 5.0)
                #print(f"epoch: {epoch}, loss_ce: {loss_ce}, loss_div: {loss_div}, loss: {loss}")
                total_loss += loss.item()
                total_loss_ce += loss_ce.item()
                total_loss_div += loss_div.item()
                
                if epoch !=0 and epoch%9==0:
         
                    self.get_alpha_value(epoch=epoch)
            
            print(f"epoch: {epoch}, loss_ce: {total_loss_ce/num_batches}, loss_div: {total_loss_div/num_batches}, loss: {total_loss/num_batches}")
        
        for l, layer in list(enumerate(self.model.model.layers)):
            block = layer.block_sparse_moe
            block.ce_loss == False
       
       
        for l, layer in list(enumerate(self.model.model.layers)):
            if l in self.layers_threshold:    
                print('layer', l, 'alpha gate', layer.block_sparse_moe.alpha_gate,'layer weight', layer.block_sparse_moe.layer_weight.item(), 'final loss', losses[l])
                
        
        concatenated_alpha_gate = self.get_alpha_value()
        self.prune_experts(concatenated_alpha_gate)

def nas_pruning(model: MixtralForCausalLM, train_loader: DataLoader, test_loader: DataLoader, args: Namespace, pruned_layers: list[int], num_train_epochs=9, lambda_coeff=0.1, lr=0.005):
    assert isinstance(
        model, MixtralForCausalLM) or isinstance(model, modeling_mixtral.MixtralForCausalLM), 'Currently only `Mixtral` is supported'
    
    nas_pruning = NAS_PrunableMixtralSparseMoe(model, train_loader, test_loader, prev=args.r, layers_threshold=pruned_layers, num_train_epochs=num_train_epochs, lambda_coeff=lambda_coeff, lr=lr)
    
    nas_pruning.train()    
    
    model = nas_pruning.model
    
    model.num_experts = args.r #unified
    model.config.num_local_experts = args.r

    return model

def pruning_model_from_load(model: MixtralForCausalLM,train_loader: DataLoader, test_loader: DataLoader, args: Namespace, pruned_layers: list[int]):
    pruned_experts=[]
    
    
    with open(args.cache_pruned_file, 'r') as file:
        for line in file:
            line = line.strip()
            line= re.findall(r'\[(.*?)\]', line)[0]
            
            if len(line)>0:
                pruned_experts.append(list(map(int, line.split(','))))
            else:
                pruned_experts.append(None)
            

   
    nas_pruning = NAS_PrunableMixtralSparseMoe(model, train_loader, test_loader, prev=args.r, layers_threshold=pruned_layers)
    nas_pruning.init_model()
    model = nas_pruning.model
    index=0
  
    for l, layer in list(enumerate(model.model.layers)):
        
        if l not in pruned_layers:
            continue
        elif pruned_experts[index] is None:
            index+=1
            continue
       
        
        layer.block_sparse_moe.experts_to_drop=pruned_experts[index]
        
        
        block = layer.block_sparse_moe
        try:
            block.prune()
        except:
            pdb.set_trace()
        
        print('layer:', l+1, block.experts_to_drop)
        index+=1
    
    
    
    for l, layer in enumerate(model.model.layers):    
        layer.block_sparse_moe = layer.block_sparse_moe.model

    return model