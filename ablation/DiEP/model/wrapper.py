import itertools as I
import logging
from typing import Optional
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import copy
from transformers.models.mixtral import modeling_mixtral 

from model.modeling_mixtral import (
    MixtralForCausalLM,
    MixtralSparseMoeBlock,
    MixtralBLockSparseTop2MLP
)
from data import CacheDataset


logger = logging.getLogger(__name__)


class PrunableMixtralSparseMoeBlockWrapper(torch.nn.Module):
    def __init__(self, model,
                 r: Optional[int] = None,
                 ):
        super().__init__()
        if isinstance(model, MixtralSparseMoeBlock) or isinstance(model, modeling_mixtral.MixtralSparseMoeBlock):
            self.model = model
        else:   
            self.model = model.model
        self.r = r

        self.experts_to_drop = None
        self.cache_space = CacheDataset()
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False
        self.experts_frequency = torch.zeros(8).float()
        self.prune_file='./logs/prune_experts.txt'

    # Forward uses topk
    def forward(self, hidden_states: torch.Tensor, Cache=True) -> torch.Tensor:
        """ """

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        # try:
            #with torch.cuda.amp.autocast():

        router_logits = self.model.gate(hidden_states)



        if self.experts_to_drop is not None:
            for e in self.experts_to_drop:
                router_logits[:, e] = -float('inf')


        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.model.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.model.num_experts).permute(2, 1, 0) #selected_experts:[16384,2], expert_mask:[8,2,16384]

        
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.model.num_experts):
            try:
                expert_layer = self.model.experts[expert_idx]
            except:
                pdb.set_trace()

            idx, top_x = torch.where(expert_mask[expert_idx]) #output index that the element is not zero

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None,
                                          top_x_list].reshape(-1, hidden_dim)


            #current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None]) #bsk modify
            current_hidden_states = expert_layer(current_state)*routing_weights[top_x_list, idx_list, None]



            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype))



        # if self.experts_to_drop is not None and (self.cache_logits or self.cache_X or self.cache_Z):
        #     logger.warn(
        #         f'Already dropped {self.experts_to_drop} but still storing activations.')

        
        if self.experts_to_drop is None and Cache:
            self.cache_space.append(alpha=(router_logits if self.cache_logits else None), X=(hidden_states if self.cache_X else None), Z=(
            final_hidden_states if self.cache_Z else None))

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits, selected_experts

    def enumerate(self):
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False
        loss_history = dict()

        with torch.inference_mode():

            for dropped in I.combinations(range(self.model.num_experts), self.model.num_experts - self.r):
                self.experts_to_drop = dropped

                loss = 0


                for (hidden_states, final_hidden_states) in zip(self.cache_space.Xs, self.cache_space.Zs):

                    hidden_states = hidden_states.to(
                        device=self.model.gate.weight.data.device, non_blocking=True)
                    final_hidden_states = final_hidden_states.to(
                        dtype=torch.float64, device=self.model.gate.weight.data.device, non_blocking=True)
              
                    final_hidden_states_e, _,_ = self.forward(
                        hidden_states.unsqueeze(0))
                    
                    loss += torch.norm(final_hidden_states -
                                       final_hidden_states_e.squeeze(0).to(torch.float64)).item()

                #print('current: ', self.experts_to_drop, 'loss:', loss)
                loss_history[dropped] = loss

        
        self.experts_to_drop = min(loss_history, key=loss_history.get)



class DynamicSkippingMixtralSparseMoeBlockWrapper(nn.Module):
    def __init__(self, model: MixtralSparseMoeBlock, beta: float, differences: list):
        super().__init__()
        assert isinstance(model, MixtralSparseMoeBlock) or isinstance(model, modeling_mixtral.MixtralSparseMoeBlock)
        assert model.top_k == 2
        self.hidden_dim = model.hidden_dim
        self.ffn_dim = model.ffn_dim
        self.num_experts = model.num_experts
        self.top_k = model.top_k
        self.gate = model.gate
        self.experts = model.experts
        
        self.differences = torch.tensor(differences, dtype=torch.bfloat16)*beta
        
        self.beta = beta

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1)

        
        self.differences = self.differences.to(hidden_states.device)
        simialrity= self.differences[selected_experts[:,0], selected_experts[:,1]]#/self.differences.median()
       
       
        #mask_top1 = (routing_weights[:, 1] < self.beta* unsimialrity *routing_weights[:, 0]) #
        mask_top1 = (routing_weights[:, 1] < self.beta*routing_weights[:, 0]) #
        routing_weights[mask_top1, 1] = 0
        
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        # (batch * sequence_length, self.top_k, n_experts)
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts)
        
        expert_mask[mask_top1, 1, :] = 0
        expert_mask = expert_mask.permute(2, 1, 0)
        #print("dynamic skipping")
        # Loop over all available experts in the model and perform the computation on each expert

        for expert_idx in range(self.num_experts):
            
            expert_layer = self.experts[expert_idx]
            
            idx, top_x = torch.where(expert_mask[expert_idx])
            

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))


        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits, selected_experts
    
class NASMixtralSparseMoeBlockWrapper(nn.Module):
    def __init__(self, model: MixtralSparseMoeBlock, weight=1.0, search=False):
        super().__init__()
        try:
            assert isinstance(model, MixtralSparseMoeBlock) or isinstance(model, modeling_mixtral.MixtralSparseMoeBlock)
        except:
            assert isinstance(model, NASMixtralSparseMoeBlockWrapper)
            new_model = copy.deepcopy(model.model)
            del model
            model = new_model
            del new_model
            torch.cuda.empty_cache()
            gc.collect()
       
        assert model.top_k == 2

        self.experts_to_drop = None
        self.cache_space = CacheDataset()
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False
        
        self.model = model
        self.hidden_dim = model.hidden_dim
        self.ffn_dim = model.ffn_dim
        self.num_experts = model.num_experts
        self.top_k = model.top_k
        self.gate = model.gate
        self.experts = model.experts
        self.alpha_gate=None

        self.layer_weight=1.0
        #self.prev=prev
        self.search=search
        self.ce_loss=False


        if self.search:
            self.layer_weight= nn.Parameter(torch.tensor(weight))
            self.alpha_gate = nn.Parameter(torch.ones(self.num_experts))
            
        

    def forward(self, hidden_states: torch.Tensor, traintime=False, Cache=True) -> torch.Tensor:
        """ """
        #
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states[0]
        batch_size = 1
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        
        #values, indices = torch.topk(aa, k=int(8-self.layer_node.item()), largest=False)
        if self.experts_to_drop is not None:

            for e in self.experts_to_drop:
                router_logits[:, e] = -float('inf')

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1)

        

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        final_hidden_states_nas = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )


        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        # (batch * sequence_length, self.top_k, n_experts)
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts)

        #expert_mask[mask_top1, 1, :] = 0
        expert_mask = expert_mask.permute(2, 1, 0)
        
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            top_x, indices = torch.where(expert_mask[expert_idx])

            if indices.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            indices_list = indices.tolist()
            top_x_list = top_x.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None,
                                          indices_list].reshape(-1, hidden_dim)
            # current_hidden_states = expert_layer(
            #     current_state, routing_weights[indices_list, top_x_list, None]).

            current_hidden_states = expert_layer(current_state)*routing_weights[indices_list, top_x_list, None]
            
            if self.ce_loss and traintime is False:
                
                current_hidden_states = expert_layer(current_state)*self.alpha_gate[expert_idx]
                
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, indices, current_hidden_states.to(hidden_states.dtype))

            if traintime is True and self.search:
                current_hidden_states_nas = expert_layer(current_state)*self.alpha_gate[expert_idx]
                final_hidden_states_nas.index_add_(
                0, indices, current_hidden_states_nas.to(hidden_states.dtype))

        if  traintime is False and Cache is True: #self.experts_to_drop is None and
            self.cache_space.append(alpha=(router_logits if self.cache_logits else None), X=(hidden_states if self.cache_X else None), Z=(
            final_hidden_states if self.cache_Z else None))


        if traintime is True:
            
            final_hidden_states_nas = final_hidden_states_nas*self.layer_weight
        if traintime is True and self.search is True:

            final_hidden_states_nas = final_hidden_states_nas.reshape(
            batch_size, sequence_length, hidden_dim)
            return final_hidden_states_nas

        elif traintime is True and self.search is False:
            return final_hidden_states
        else:
            
            final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits, selected_experts


    @torch.no_grad()
    def prune(self):

        assert self.experts_to_drop is not None
        #assert len(self.experts_to_drop) == self.model.num_experts - self.r
        del self.cache_space

        if len(self.experts_to_drop) == 0:
            return

        self.cache_X = False
        self.cache_Z = False

        experts_to_reserve = sorted(
            set(range(self.model.num_experts)) - set(self.experts_to_drop))

        gate_new = torch.nn.Linear(in_features=self.model.gate.in_features,
                                   out_features=len(experts_to_reserve), bias=False, device='cpu', dtype=torch.bfloat16)
        gate_new.weight.data = self.model.gate.weight.data[list(
            experts_to_reserve)]
        self.model.gate = gate_new

        self.model.experts = torch.nn.ModuleList(
            [self.model.experts[i] for i in experts_to_reserve])

        self.model.num_experts = len(experts_to_reserve)



def txt_write(filename, content):
    with open(filename, "a") as file:
        file.write("{}\n".format(content))