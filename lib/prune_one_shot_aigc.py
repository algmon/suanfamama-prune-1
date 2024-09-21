import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 

from .ablate import AblateGPT

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_aigc_technique1(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    梯度敏感剪枝：基于权重对损失的敏感性进行剪枝。
    status: runtime error
    """
    print('aigc tech 1: Starting Gradient Sensitivity Pruning...')
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name, module in subset.items():
            W = module.weight.data
            W.requires_grad = True

            # 模拟一次前向传播和反向传播以获取梯度
            # 假设有一个损失函数，这里仅作为示例
            # 实际应用中需要根据具体任务定义损失函数
            optimizer = torch.optim.SGD([W], lr=0.001)
            optimizer.zero_grad()
            loss = W.sum()  # 示例损失
            loss.backward()

            # 计算敏感度度量
            sensitivity = torch.abs(W.grad) * torch.abs(W)
            sensitivity = sensitivity.cpu()

            # 根据敏感度选择剪枝阈值
            if prune_n != 0:
                threshold, _ = torch.topk(sensitivity.view(-1), prune_n, largest=False)
                thresh = threshold[-1]
                W_mask = sensitivity <= thresh
            else:
                thresh = torch.sort(sensitivity.view(-1), descending=False)[0][int(W.numel() * args.sparsity_ratio)]
                W_mask = sensitivity <= thresh

            # 应用剪枝掩码
            W[W_mask] = 0

            module.weight.data = W.to(device)

            print(f"Layer {i}, {name}: Pruned with sensitivity threshold {thresh.item():.6f}")

    print('Gradient Sensitivity Pruning Completed.')

def prune_aigc_technique2(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    L1 范数剪枝：基于权重的 L1 范数进行剪枝。
    status: ok and number reported
    """
    print('aigc tech 2: Starting L1 Norm Pruning...')
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name, module in subset.items():
            W = module.weight.data
            W_norm = torch.norm(W, p=1, dim=1)  # 计算每一行的 L1 范数

            # 根据 L1 范数选择剪枝阈值
            if prune_n != 0:
                threshold, _ = torch.topk(W_norm, prune_n, largest=False)
                thresh = threshold[-1]
                prune_rows = W_norm <= thresh
            else:
                thresh = torch.sort(W_norm, descending=False)[0][int(W_norm.numel() * args.sparsity_ratio)]
                prune_rows = W_norm <= thresh

            # 剪枝整行权重
            W[prune_rows, :] = 0

            module.weight.data = W.to(device)

            print(f"Layer {i}, {name}: Pruned {prune_rows.sum().item()} rows with L1 norm threshold {thresh.item():.6f}")

    print('L1 Norm Pruning Completed.')

def prune_aigc_technique3(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    结构化剪枝：基于通道的重要性进行剪枝。
    status: ok and number reported
    """
    print('aigc tech 3: Starting Structured Pruning...')
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name, module in subset.items():
            W = module.weight.data
            W_norm = torch.norm(W, p=1, dim=0)  # 计算每个通道（列）的 L1 范数

            # 根据通道 L1 范数选择剪枝阈值
            if prune_n != 0:
                threshold, _ = torch.topk(W_norm, prune_n, largest=False)
                thresh = threshold[-1]
                prune_channels = W_norm <= thresh
            else:
                thresh = torch.sort(W_norm, descending=False)[0][int(W_norm.numel() * args.sparsity_ratio)]
                prune_channels = W_norm <= thresh

            # 剪枝整通道权重
            W[:, prune_channels] = 0

            module.weight.data = W.to(device)

            print(f"Layer {i}, {name}: Pruned {prune_channels.sum().item()} channels with L1 norm threshold {thresh.item():.6f}")

    print('Structured Pruning Completed.')

from sklearn.cluster import KMeans
import numpy as np

def prune_aigc_technique4(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    K-means 聚类剪枝：通过权重聚类来实现剪枝和量化。
    status: runtime error
    """
    print('aigc tech 4: Starting K-means Clustering Pruning...')
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name, module in subset.items():
            W = module.weight.data.cpu().numpy().reshape(-1, 1)  # 转换为二维数组
            num_clusters = int(1 / args.sparsity_ratio)  # 设定聚类数量
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(W)
            cluster_centers = kmeans.cluster_centers_
            labels = kmeans.labels_

            # 分配聚类中心值
            W_pruned = cluster_centers[labels].reshape(module.weight.data.shape)
            module.weight.data = torch.from_numpy(W_pruned).to(device)

            print(f"Layer {i}, {name}: Applied K-means clustering with {num_clusters} clusters.")

    print('K-means Clustering Pruning Completed.')

def prune_aigc_technique5(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    随机剪枝：随机选择权重进行剪枝。
    status: runtime error
    """
    print('aigc tech 5: Starting Random Pruning...')
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name, module in subset.items():
            W = module.weight.data
            total_params = W.numel()
            if prune_n != 0:
                num_prune = prune_n * (W.shape[1] // prune_m)
            else:
                num_prune = int(total_params * args.sparsity_ratio)

            # 生成随机掩码
            W_mask = torch.zeros_like(W, dtype=torch.bool)
            prune_indices = torch.randperm(total_params)[:num_prune]
            W_mask.view(-1)[prune_indices] = True

            # 应用剪枝掩码
            W[W_mask] = 0

            module.weight.data = W.to(device)

            print(f"Layer {i}, {name}: Randomly pruned {W_mask.sum().item()} weights.")

    print('Random Pruning Completed.')

def prune_aigc_technique6(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    Random Pattern Pruning: Prune weights following a random pattern within each block.
    status: ok and number reported
    """
    import random

    print('aigc tech 6: Starting Random Pattern Pruning...')
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name, module in subset.items():
            W = module.weight.data
            total_params = W.numel()
            if prune_n != 0:
                num_prune = prune_n * (W.shape[1] // prune_m)
            else:
                num_prune = int(total_params * args.sparsity_ratio)

            # Generate random indices to prune
            prune_indices = torch.randperm(total_params)[:num_prune]
            W_mask = torch.zeros_like(W, dtype=torch.bool).view(-1)
            W_mask[prune_indices] = True
            W_mask = W_mask.view(W.size())

            # Apply pruning mask
            W[W_mask] = 0

            module.weight.data = W.to(device)

            print(f"Layer {i}, {name}: Randomly pruned {W_mask.sum().item()} weights.")

    print('Random Pattern Pruning Completed.')

class VariationalDropout(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight.data.clone())
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        return x  # The actual dropout is applied during pruning

def prune_aigc_technique7(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    Variational Dropout Pruning: Prune weights based on learned dropout probabilities.
    status: runtime error
    """
    print('aigc tech 7: Starting Variational Dropout Pruning...')
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name, module in subset.items():
            W = module.weight.data
            # Initialize Variational Dropout
            variational_dropout = VariationalDropout(W)
            # Simulate training to learn dropout probabilities
            optimizer = torch.optim.Adam([variational_dropout.weight], lr=1e-3)
            for epoch in range(5):  # Small number of epochs
                optimizer.zero_grad()
                output = layer.forward(W)  # Dummy forward
                loss = (variational_dropout.weight ** 2).sum()
                loss.backward()
                optimizer.step()

            # Determine pruning mask based on dropout probabilities
            dropout_probs = torch.sigmoid(variational_dropout.weight)
            if prune_n != 0:
                threshold = torch.topk(dropout_probs.view(-1), prune_n, largest=False)[0][-1]
                W_mask = dropout_probs <= threshold
            else:
                threshold = torch.quantile(dropout_probs.view(-1), args.sparsity_ratio)
                W_mask = dropout_probs <= threshold

            W[W_mask] = 0
            module.weight.data = W.to(device)

            print(f"Layer {i}, {name}: Pruned {W_mask.sum().item()} weights based on Variational Dropout.")

    print('Variational Dropout Pruning Completed.')

def prune_aigc_technique8(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    Gradient-Based Pruning: Prune weights based on their gradient magnitudes.
    status: runtime error
    """
    print('aigc tech 8: Starting Gradient-Based Pruning...')
    layers = model.model.layers

    # Dummy input and label for gradient computation
    dummy_input = tokenizer("This is a dummy input for gradient computation.", return_tensors="pt").to(device)
    dummy_label = torch.tensor([0]).to(device)  # Dummy label

    # Set model to training mode
    model.train()

    # Forward pass
    outputs = model(**dummy_input, labels=dummy_label)
    loss = outputs.loss
    loss.backward()

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name, module in subset.items():
            W = module.weight.data
            if module.weight.grad is None:
                continue
            grad = module.weight.grad.abs()
            sensitivity = grad * W.abs()

            # Determine threshold
            if prune_n != 0:
                threshold = torch.topk(sensitivity.view(-1), prune_n, largest=False)[0][-1]
                W_mask = sensitivity <= threshold
            else:
                threshold = torch.quantile(sensitivity.view(-1), args.sparsity_ratio)
                W_mask = sensitivity <= threshold

            # Apply pruning mask
            W[W_mask] = 0
            module.weight.data = W.to(device)

            print(f"Layer {i}, {name}: Pruned {W_mask.sum().item()} weights based on gradient sensitivity.")

    # Reset gradients
    model.zero_grad()
    model.eval()

    print('Gradient-Based Pruning Completed.')

def prune_aigc_technique9(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    Elastic Weight Consolidation (EWC) Pruning: Prune weights while consolidating important weights to prevent forgetting.
    status: runtime error
    """
    import copy

    print('aigc tech 9: Starting Elastic Weight Consolidation (EWC) Pruning...')
    layers = model.model.layers

    # Compute Fisher Information
    fisher = {}
    model.eval()
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name, module in subset.items():
            fisher[name] = module.weight.data.clone().fill_(0)

    # Dummy input and label for Fisher Information computation
    dummy_input = tokenizer("This is a dummy input for EWC computation.", return_tensors="pt").to(device)
    dummy_label = torch.tensor([0]).to(device)

    model.train()
    outputs = model(**dummy_input, labels=dummy_label)
    loss = outputs.loss
    loss.backward()

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name, module in subset.items():
            if module.weight.grad is None:
                continue
            fisher[name] += module.weight.grad.data.clone().pow(2)

    # Average Fisher Information
    for name in fisher:
        fisher[name] = fisher[name] / len(layers)

    # Pruning with EWC
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name, module in subset.items():
            W = module.weight.data
            importance = fisher[name]
            sensitivity = importance * W.abs()

            # Determine threshold
            if prune_n != 0:
                threshold = torch.topk(sensitivity.view(-1), prune_n, largest=False)[0][-1]
                W_mask = sensitivity <= threshold
            else:
                threshold = torch.quantile(sensitivity.view(-1), args.sparsity_ratio)
                W_mask = sensitivity <= threshold

            # Apply pruning mask
            W[W_mask] = 0
            module.weight.data = W.to(device)

            print(f"Layer {i}, {name}: Pruned {W_mask.sum().item()} weights based on EWC sensitivity.")

    # Reset gradients
    model.zero_grad()
    model.eval()

    print('Elastic Weight Consolidation (EWC) Pruning Completed.')

import torch.optim as optim
from torch.distributions import Bernoulli
import random

class PruningAgent:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.policy = {}
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99  # Discount factor

        # Initialize policy: probability of pruning each weight
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                self.policy[name] = torch.ones_like(param, requires_grad=True) * 0.5  # Start with 50% prune chance

    def select_action(self):
        actions = {}
        for name, probs in self.policy.items():
            m = Bernoulli(probs)
            actions[name] = m.sample()
        return actions

    def apply_pruning(self, actions):
        for name, action in actions.items():
            W = dict(self.model.named_parameters())[name].data
            W[action.bool()] = 0
            dict(self.model.named_parameters())[name].data = W

    def compute_reward(self, loss_before, loss_after):
        # Simple reward: reduction in loss
        return (loss_before - loss_after).item()

    def update_policy(self, actions, rewards):
        # Placeholder for policy gradient update
        # Implement the actual policy gradient computation
        pass

def prune_aigc_technique10(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    Dynamic Pruning with Reinforcement Learning: Use an RL agent to decide which weights to prune.
    status: runtime error
    """
    print('aigc tech 10: Starting Dynamic Pruning with Reinforcement Learning...')
    layers = model.model.layers
    agent = PruningAgent(model, args)

    # Dummy input and label for training
    dummy_input = tokenizer("This is a dummy input for RL pruning.", return_tensors="pt").to(device)
    dummy_label = torch.tensor([0]).to(device)

    for episode in range(args.rl_episodes):
        model.train()
        loss_before = 0.0

        # Compute loss before pruning
        outputs = model(**dummy_input, labels=dummy_label)
        loss_before = outputs.loss.item()

        # Select actions
        actions = agent.select_action()

        # Apply pruning
        agent.apply_pruning(actions)

        # Compute loss after pruning
        outputs = model(**dummy_input, labels=dummy_label)
        loss_after = outputs.loss.item()

        # Compute reward
        reward = agent.compute_reward(loss_before, loss_after)

        # Update policy
        agent.update_policy(actions, reward)

        print(f"Episode {episode+1}: Reward: {reward}")

        if reward < args.reward_threshold:
            print("Desired reward achieved. Stopping pruning.")
            break

    print('Dynamic Pruning with Reinforcement Learning Completed.')
