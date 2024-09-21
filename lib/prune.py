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

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
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
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
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
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def prune_opposite_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)

            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        # Prune the largest 'prune_n' values within each block
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=True)[1], True) 
            else:
                # Prune values above the threshold (opposite of original logic)
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*(1-args.sparsity_ratio))].cpu() 
                W_mask = (W_metric >= thresh) 

            W[W_mask] = 0

def prune_mama(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # TODO: Optimize the MAMA pruning algorithm based on the description in the paper.
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)  # Magnitude-based importance

            if prune_n != 0:
                # Structured Movement Pruning
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        block = W[:, ii:(ii + prune_m)]
                        block_metric = W_metric[:, ii:(ii + prune_m)]

                        # Find indices of weights to prune and keep
                        prune_indices = torch.topk(block_metric, prune_n, dim=1, largest=False)[1]
                        keep_indices = torch.topk(block_metric, prune_m - prune_n, dim=1, largest=True)[1]

                        # Move values from pruned weights to the average of kept weights
                        block[torch.arange(block.shape[0]).unsqueeze(1), prune_indices] = 0  # Zero out pruned weights
                        block[torch.arange(block.shape[0]).unsqueeze(1), keep_indices] += \
                            block[torch.arange(block.shape[0]).unsqueeze(1), prune_indices].sum(dim=1, keepdim=True) / (
                                        prune_m - prune_n)

                        W[:, ii:(ii + prune_m)] = block  # Update the original weight matrix

            else:
                # Unstructured Movement Pruning
                total_pruned = int(W.numel() * args.sparsity_ratio)

                # Find indices of weights to prune and keep globally
                prune_indices = torch.topk(W_metric.flatten(), total_pruned, largest=False)[1]
                keep_indices = torch.topk(W_metric.flatten(), W.numel() - total_pruned, largest=True)[1]

                # Move values from pruned weights to the average of kept weights
                W.flatten()[prune_indices] = 0  # Zero out pruned weights
                W.flatten()[keep_indices] += W.flatten()[prune_indices].sum() / (W.numel() - total_pruned)

            subset[name].weight.data = W  # Update the layer's weight

def prune_aigc_technique1(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    梯度敏感剪枝：基于权重对损失的敏感性进行剪枝。
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
