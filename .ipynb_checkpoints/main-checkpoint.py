# main.py
# python version 3.7.1
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import os
import copy
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from sklearn.mixture import GaussianMixture
import torch.nn as nn

from util.options import args_parser
from util.local_training import LocalUpdate, globaltest
from util.fedavg import FedAvg
from util.util import add_noise, get_output
from util.dataset import get_dataset
from model.build_model import build_model

# 设置打印选项
np.set_printoptions(threshold=np.inf)

def get_gmm_params(losses):
    """
    拟合 GMM 并返回参数
    返回: mu (sorted), sigma (sorted), pi (sorted), clean_component_index
    """
    losses = np.array(losses).reshape(-1, 1)
    # 拟合 2 个高斯分量 (Clean, Noisy)
    # n_init=10 增加初始化的稳定性
    gmm = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(losses)
    
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()
    
    # 均值较小的分量被认为是 "Clean"
    clean_idx = np.argmin(means)
    
    # 确保输出时按照 [Clean, Noisy] 的顺序排序，方便聚合
    if clean_idx == 1:
        means = means[[1, 0]]
        covariances = covariances[[1, 0]]
        weights = weights[[1, 0]]
    
    return means, covariances, weights

def classify_samples(losses, gmm_local, gmm_global):
    """
    根据文档要求将样本分为三类：
    Clean(0): 全局Clean 且 本地Clean
    Noisy(1): 全局Noisy 且 本地Noisy
    Complex(2): 其他情况
    """
    losses = np.array(losses).reshape(-1, 1)
    
    # --- 本地预测 ---
    local_labels = gmm_local.predict(losses)
    # 均值较小的为 Clean
    local_clean_idx = np.argmin(gmm_local.means_[:, 0])
    is_local_clean = (local_labels == local_clean_idx)
    is_local_noisy = ~is_local_clean
    
    # --- 全局预测 ---
    global_labels = gmm_global.predict(losses)
    # 均值较小的为 Clean
    global_clean_idx = np.argmin(gmm_global.means_[:, 0])
    is_global_clean = (global_labels == global_clean_idx)
    is_global_noisy = ~is_global_clean
    
    # --- 交集逻辑 ---
    is_final_clean = is_local_clean & is_global_clean
    is_final_noisy = is_local_noisy & is_global_noisy
    
    # 初始化为 2 (Complex)
    sample_types = np.full(len(losses), 2)
    sample_types[is_final_clean] = 0
    sample_types[is_final_noisy] = 1
    
    return sample_types

def compute_gradient_alignment(w_local, w_global_prev):
    """计算本地更新与上一轮全局更新的梯度对齐度 (Cosine Similarity)"""
    v_local = torch.cat([v.flatten() for v in w_local.values()])
    v_global = torch.cat([v.flatten() for v in w_global_prev.values()])
    # 防止除以零
    sim = F.cosine_similarity(v_local.unsqueeze(0), v_global.unsqueeze(0)).item()
    if np.isnan(sim): return 0.0
    return sim

if __name__ == '__main__':
    args = args_parser()
    # 修复 device 定义
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args)

    # 随机种子初始化
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 加载数据
    dataset_train, dataset_test, dict_users = get_dataset(args)
    total_classes = len(np.unique(dataset_train.targets))
    
    # 添加噪声
    y_train_clean = np.array(dataset_train.targets)
    y_train_noisy, _, _ = add_noise(args, y_train_clean, dict_users)
    dataset_train.targets = y_train_noisy

    # 构建全局模型
    netglob = build_model(args).to(args.device)
    netglob.train()
    
    # 初始化激励与历史记录
    client_reputation = {i: 1.0 for i in range(args.num_users)} # r_t 初始声誉
    w_glob_prev = copy.deepcopy(netglob.state_dict())
    
    # 激励参数 (可从args读取)
    alpha1, alpha2, alpha3 = args.alpha1, args.alpha2, args.alpha3
    
    # 全局存储变量
    global_sample_types = {}    # {global_idx: type}
    global_pseudo_labels = {}   # {global_idx: label}

    # 结果记录文件
    txtpath = "./record/txtsave/MyMethod_%s_Rounds_%d_Seed_%d.txt" % (args.dataset, args.rounds1, args.seed)
    f_acc = open(txtpath, 'w')
    
    # === 主训练循环 ===
    # 使用 rounds1 作为主要的训练轮数
    total_rounds = args.rounds1 
    
    for round_idx in range(total_rounds):
        print(f"\n--- Round {round_idx} / {total_rounds} ---")
        
        # 1. 客户端选择
        m = max(int(args.frac1 * args.num_users), 1)
        # 这里使用简单的随机采样，也可以改为基于声誉采样: p=reputation/sum(reputation)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # --- PHASE 1: 噪声滤波器更新与样本分类 ---
        # 这一步对应文档: "在协作训练全局模型的同时学习全局噪声滤波器"
        
        client_gmm_params = [] # 存储本地 GMM 参数
        client_losses = {}     # 缓存损失值，避免重复计算
        
        criterion_red = nn.CrossEntropyLoss(reduction='none')
        
        # 1.1 客户端计算损失并拟合本地 GMM
        netglob.eval()
        for idx in idxs_users:
            idxs = list(dict_users[idx])
            loader = DataLoader(Subset(dataset_train, idxs), batch_size=100, shuffle=False)
            
            # 获取当前全局模型下的损失分布
            # 注意: get_output 需要返回 (outputs, losses)，请确保 util.py 支持
            outputs, losses = get_output(loader, netglob.to(args.device), args, False, criterion_red)
            client_losses[idx] = (losses, outputs)
            
            # 拟合本地 GMM
            mu, sigma, pi = get_gmm_params(losses)
            client_gmm_params.append({
                'mu': mu, 'sigma': sigma, 'pi': pi, 'n': len(idxs)
            })

        # 1.2 服务器聚合 GMM (FedDiv 思想)
        total_samples = sum([c['n'] for c in client_gmm_params])
        mu_g = np.zeros(2)
        sigma_g = np.zeros(2)
        pi_g = np.zeros(2)
        
        for c in client_gmm_params:
            weight = c['n'] / total_samples
            mu_g += weight * c['mu']
            sigma_g += weight * c['sigma']
            pi_g += weight * c['pi']
            
        # 构建全局 GMM 对象
        gmm_global = GaussianMixture(n_components=2, random_state=args.seed)
        gmm_global.means_ = mu_g.reshape(-1, 1)
        gmm_global.covariances_ = sigma_g.reshape(-1, 1, 1)
        gmm_global.weights_ = pi_g
        # 重新计算精度矩阵，防止报错
        gmm_global.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm_global.covariances_))

        # 1.3 样本分类与重标记 (在客户端本地进行)
        client_noise_rates = {} # Sn
        client_class_diversity = {} # Ci
        
        for idx in idxs_users:
            idxs = list(dict_users[idx])
            losses, outputs = client_losses[idx]
            
            # 恢复本地 GMM 用于分类
            # 为了简单，直接重新拟合该客户端的数据（或者使用上面存的参数）
            # 这里选择根据参数重建 GMM 对象
            c_params = client_gmm_params[list(idxs_users).index(idx)]
            gmm_local = GaussianMixture(n_components=2, random_state=args.seed)
            gmm_local.means_ = c_params['mu'].reshape(-1, 1)
            gmm_local.covariances_ = c_params['sigma'].reshape(-1, 1, 1)
            gmm_local.weights_ = c_params['pi']
            gmm_local.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm_local.covariances_))

            # 执行三分类 (Clean, Noisy, Complex)
            types = classify_samples(losses, gmm_local, gmm_global)
            
            # 更新全局样本类型记录 (用于 LocalUpdate)
            for i, global_id in enumerate(idxs):
                global_sample_types[global_id] = types[i]

            # 计算噪声率 Sn (定义: 噪声样本数 / 总数)
            num_noisy = np.sum(types == 1)
            Sn = num_noisy / len(idxs) if len(idxs) > 0 else 0
            client_noise_rates[idx] = Sn
            
            # 计算类别多样性 Ci (定义: 本地类别数 / 全局类别数)
            local_classes = len(np.unique(np.array(dataset_train.targets)[idxs]))
            Ci = local_classes / total_classes
            client_class_diversity[idx] = Ci
            
            # --- 重标记与一致性筛选 (针对 Noisy 样本) ---
            noisy_indices_local = np.where(types == 1)[0]
            if len(noisy_indices_local) > 0:
                # 1. 计算预测置信度
                probs = F.softmax(torch.tensor(outputs), dim=1).numpy()
                max_probs = np.max(probs, axis=1) # pc
                preds = np.argmax(probs, axis=1)  # 伪标签
                
                # 2. 预测一致性筛选 (PCS): 训练 1 轮本地模型进行校验
                # 文档: "客户端训练 1 轮本地模型... 要求全局和局部预测一致"
                net_temp = copy.deepcopy(netglob)
                # 使用标准训练方式训练临时模型
                temp_local = LocalUpdate(args, dataset_train, idxs, sample_types=None) 
                w_temp, _ = temp_local.update_weights(net_temp, args.seed, netglob, epoch=1)
                net_temp.load_state_dict(w_temp)
                net_temp.eval()
                
                # 获取本地模型预测
                loader_temp = DataLoader(Subset(dataset_train, idxs), batch_size=100, shuffle=False)
                out_temp, _ = get_output(loader_temp, net_temp.to(args.device), args, False, criterion_red)
                preds_local = np.argmax(F.softmax(torch.tensor(out_temp), dim=1).cpu().numpy(), axis=1)
                
                # 筛选与重标记
                zeta = args.zeta # 阈值 0.75
                for loc_i in noisy_indices_local:
                    if max_probs[loc_i] >= zeta: # 置信度阈值
                        if preds[loc_i] == preds_local[loc_i]: # 一致性校验
                            # 只有都满足才信任该伪标签
                            global_id = idxs[loc_i]
                            global_pseudo_labels[global_id] = preds[loc_i]

        # --- PHASE 2: 联邦训练 (Local Training) ---
        w_locals = []
        loss_locals = []
        
        for idx in idxs_users:
            # 传入更新后的样本类型和伪标签
            local = LocalUpdate(args, dataset_train, dict_users[idx], 
                                sample_types=global_sample_types, 
                                pseudo_labels=global_pseudo_labels)
            
            w, loss = local.update_weights(net=copy.deepcopy(netglob).to(args.device), 
                                           seed=args.seed, w_g=netglob, epoch=args.local_ep)
            w_locals.append(w)
            loss_locals.append(loss)
            
            # --- PHASE 3: 贡献评估与激励更新 ---
            # 计算梯度对齐度 Phi
            # 注意: 第0轮没有上一轮模型，设为 1 或跳过
            if round_idx == 0:
                phi = 1.0
            else:
                phi = compute_gradient_alignment(w, w_glob_prev)
            
            Sn = client_noise_rates[idx]
            Ci = client_class_diversity[idx]
            
            # 激励公式: r_t = r_{t-1} + alpha1 * (phi - alpha2 * Sn + alpha3 * Ci)
            # 文档中: "将图多样性替换成噪声率... 噪声率增加则权重/价值减少"
            val_update = phi - alpha2 * Sn + alpha3 * Ci
            client_reputation[idx] += alpha1 * val_update
            
            # 记录日志
            log_str = f"Round {round_idx}, Client {idx}: Sn={Sn:.3f}, Ci={Ci:.3f}, Phi={phi:.3f}, Rep={client_reputation[idx]:.3f}"
            print(log_str)
            f_acc.write(log_str + "\n")

        # --- PHASE 4: 自适应聚合 ---
        # 权重计算: w_g = sum (m_i * e^-eta_i * w_i)
        # eta_i = Sn_i / max(Sn)
        
        all_Sn = list(client_noise_rates.values())
        max_Sn = max(all_Sn) if len(all_Sn) > 0 and max(all_Sn) > 0 else 1.0
        
        agg_weights = []
        for i, idx in enumerate(idxs_users):
            Sn = client_noise_rates[idx]
            eta = Sn / max_Sn
            m_i = len(dict_users[idx])
            
            # 动态权重
            w = m_i * np.exp(-eta)
            agg_weights.append(w)
            
        # 执行聚合
        w_glob = FedAvg(w_locals, agg_weights)
        
        # 更新全局模型
        w_glob_prev = copy.deepcopy(netglob.state_dict())
        netglob.load_state_dict(w_glob)
        
        # --- 测试 ---
        acc_test = globaltest(copy.deepcopy(netglob), dataset_test, args)
        print(f"Round {round_idx} Global Accuracy: {acc_test:.2%}")
        f_acc.write(f"Round {round_idx} Global Accuracy: {acc_test:.4f}\n")
        f_acc.flush()

    f_acc.close()