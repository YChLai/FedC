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
from datetime import datetime

from util.options import args_parser
from util.local_training import LocalUpdate, globaltest
from util.fedavg import FedAvg
from util.util import add_noise, get_output
from util.dataset import get_dataset
from model.build_model import build_model
import matplotlib.pyplot as plt

# 设置打印选项
np.set_printoptions(threshold=np.inf)

def get_gmm_params(losses):
    """
    拟合 GMM 并返回参数
    返回: mu (sorted), sigma (sorted), pi (sorted), clean_component_index
    """
    losses = np.array(losses).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(losses)
    
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()
    
    clean_idx = np.argmin(means)
    
    if clean_idx == 1:
        means = means[[1, 0]]
        covariances = covariances[[1, 0]]
        weights = weights[[1, 0]]
    
    return means, covariances, weights

def classify_samples(losses, gmm_local, gmm_global):
    """
    Clean(0): 全局Clean 且 本地Clean
    Noisy(1): 全局Noisy 且 本地Noisy
    Complex(2): 其他情况
    """
    losses = np.array(losses).reshape(-1, 1)
    
    local_labels = gmm_local.predict(losses)
    local_clean_idx = np.argmin(gmm_local.means_[:, 0])
    is_local_clean = (local_labels == local_clean_idx)
    is_local_noisy = ~is_local_clean
    
    global_labels = gmm_global.predict(losses)
    global_clean_idx = np.argmin(gmm_global.means_[:, 0])
    is_global_clean = (global_labels == global_clean_idx)
    is_global_noisy = ~is_global_clean
    
    is_final_clean = is_local_clean & is_global_clean
    is_final_noisy = is_local_noisy & is_global_noisy
    
    sample_types = np.full(len(losses), 2)
    sample_types[is_final_clean] = 0
    sample_types[is_final_noisy] = 1
    
    return sample_types

def compute_gradient_alignment(w_local, w_global_prev):
    v_local = torch.cat([v.flatten() for v in w_local.values()])
    v_global = torch.cat([v.flatten() for v in w_global_prev.values()])
    sim = F.cosine_similarity(v_local.unsqueeze(0), v_global.unsqueeze(0)).item()
    if np.isnan(sim): return 0.0
    return sim

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    dataset_train, dataset_test, dict_users = get_dataset(args)
    total_classes = len(np.unique(dataset_train.targets))
    
    y_train_clean = np.array(dataset_train.targets)
    y_train_noisy, _, _ = add_noise(args, y_train_clean, dict_users)
    dataset_train.targets = y_train_noisy

    netglob = build_model(args).to(args.device)
    netglob.train()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join("./record", current_time)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    print(f"📁 Experiment results will be saved to: {experiment_dir}")

    # 保存本次实验的所有参数配置到 config.txt
    config_path = os.path.join(experiment_dir, "config.txt")
    with open(config_path, 'w') as f_config:
        f_config.write("Experimental Configuration:\n")
        f_config.write("="*30 + "\n")
        for arg in vars(args):
            f_config.write(f"{arg}: {getattr(args, arg)}\n")
    print("📄 Configuration saved.")
    
    client_reputation = {i: 1.0 for i in range(args.num_users)} 
    w_glob_prev = copy.deepcopy(netglob.state_dict())
    
    alpha1, alpha2, alpha3 = args.alpha1, args.alpha2, args.alpha3
    
    global_sample_types = {}    
    global_pseudo_labels = {}   
    
    total_rounds = args.total_rounds
    WARMUP_ROUNDS = args.warmup_rounds

    log_filename = "MyMethod_%s_Rounds_%d_WarmupRounds_%d_Seed_%d_acc.txt" % (args.dataset, args.total_rounds, args.warmup_rounds,args.seed)
    txtpath = os.path.join(experiment_dir, log_filename)
    f_acc = open(txtpath, 'w')
    
  # 建议设置 10 到 20 轮
    
    # 原始参数备份 (用于热身结束后恢复)
    orig_lam_c = args.lambda_c
    orig_lam_n = args.lambda_n
    orig_lam_h = args.lambda_h
    acc_history = []
    for round_idx in range(total_rounds):
        print(f"\n--- Round {round_idx} / {total_rounds} ---")
        
        m = max(int(args.frac1 * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # 初始化存储容器
        client_noise_rates = {} 
        client_class_diversity = {}
        client_train_idxs = {} 
        
        # =========== 逻辑分支：热身阶段 vs 正式阶段 ===========
        if round_idx < WARMUP_ROUNDS:
            print(f"🔥 [Warm-up Phase] Round {round_idx}: Using Standard CrossEntropy Training (No Filter)")
            
            # 1. 临时调整 Loss 权重：只用 Clean Loss (CrossEntropy), 权重设为 1.0
            args.lambda_c = 1.0
            args.lambda_n = 0.0
            args.lambda_h = 0.0
            
            # 2. 跳过 GMM，强制将所有样本设为 "Clean" (Type 0) 并保留所有数据
            for idx in idxs_users:
                idxs = list(dict_users[idx])
                
                # 计算 Ci (Diversity) 仅用于记录，不影响训练
                local_classes = len(np.unique(np.array(dataset_train.targets)[idxs]))
                client_class_diversity[idx] = local_classes / total_classes
                client_noise_rates[idx] = 0.5 # 占位符
                
                # 关键：所有样本标记为 0 (Clean)，全部用于训练
                for global_id in idxs:
                    global_sample_types[global_id] = 0 # 0 = Clean
                
                client_train_idxs[idx] = idxs # 不进行筛选，使用全部本地数据

        else:
            # 恢复 Loss 权重
            if round_idx == WARMUP_ROUNDS:
                print("🚀 [Warm-up Finished] Switching to FedC Noise Filter Mode!")
            
            args.lambda_c = orig_lam_c
            args.lambda_n = orig_lam_n
            args.lambda_h = orig_lam_h
            
            # --- PHASE 1: 噪声滤波器更新与样本分类 & 筛选 (原逻辑) ---
            client_gmm_params = []
            client_losses = {}     
            criterion_red = nn.CrossEntropyLoss(reduction='none')
            
            # 1.1 计算损失和本地 GMM
            netglob.eval()
            for idx in idxs_users:
                idxs = list(dict_users[idx])
                loader = DataLoader(Subset(dataset_train, idxs), batch_size=100, shuffle=False)
                
                # 获取 Loss
                outputs, losses = get_output(loader, netglob.to(args.device), args, False, criterion_red)
                client_losses[idx] = (losses, outputs)
                
                # 拟合 GMM (此时模型已经有区分度，GMM 不会报错了)
                try:
                    mu, sigma, pi = get_gmm_params(losses)
                except Exception as e:
                    # 极少数情况如果还是报错，使用默认值兜底
                    print(f"Warning: Client {idx} GMM fit failed, using default. Error: {e}")
                    mu, sigma, pi = np.array([0.1, 0.9]), np.array([0.1, 0.1]), np.array([0.5, 0.5])

                client_gmm_params.append({
                    'mu': mu, 'sigma': sigma, 'pi': pi, 'n': len(idxs)
                })

            # 1.2 聚合全局 GMM
            total_samples = sum([c['n'] for c in client_gmm_params])
            mu_g = np.zeros(2)
            sigma_g = np.zeros(2)
            pi_g = np.zeros(2)
            
            for c in client_gmm_params:
                weight = c['n'] / total_samples
                mu_g += weight * c['mu']
                sigma_g += weight * c['sigma']
                pi_g += weight * c['pi']
                
            gmm_global = GaussianMixture(n_components=2, random_state=args.seed)
            gmm_global.means_ = mu_g.reshape(-1, 1)
            gmm_global.covariances_ = sigma_g.reshape(-1, 1, 1)
            gmm_global.weights_ = pi_g
            try:
                gmm_global.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm_global.covariances_))
            except:
                gmm_global.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm_global.covariances_ + 1e-6)) # 增加稳定性

            # 1.3 样本分类、重标记与筛选
            for idx in idxs_users:
                idxs = list(dict_users[idx])
                losses, outputs = client_losses[idx]
                
                # 恢复本地 GMM
                c_params = client_gmm_params[list(idxs_users).index(idx)]
                gmm_local = GaussianMixture(n_components=2, random_state=args.seed)
                gmm_local.means_ = c_params['mu'].reshape(-1, 1)
                gmm_local.covariances_ = c_params['sigma'].reshape(-1, 1, 1)
                gmm_local.weights_ = c_params['pi']
                try:
                    gmm_local.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm_local.covariances_))
                except:
                    gmm_local.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm_local.covariances_ + 1e-6))

                # 分类
                types = classify_samples(losses, gmm_local, gmm_global)
                
                for i, global_id in enumerate(idxs):
                    global_sample_types[global_id] = types[i]

                num_noisy_raw = np.sum(types == 1)
                Sn = num_noisy_raw / len(idxs) if len(idxs) > 0 else 0
                client_noise_rates[idx] = Sn
                
                local_classes = len(np.unique(np.array(dataset_train.targets)[idxs]))
                Ci = local_classes / total_classes
                client_class_diversity[idx] = Ci
                
                # --- 筛选逻辑 (PCS) ---
                # 准备本地验证模型 (仅当有噪声样本时)
                noisy_indices_local_idx = np.where(types == 1)[0]
                preds_local = None 
                if len(noisy_indices_local_idx) > 0:
                    net_temp = copy.deepcopy(netglob)
                    # 临时训练一轮
                    temp_local = LocalUpdate(args, dataset_train, idxs, sample_types=None) 
                    w_temp, _ = temp_local.update_weights(net_temp, args.seed, netglob, epoch=1)
                    net_temp.load_state_dict(w_temp)
                    net_temp.eval()
                    loader_temp = DataLoader(Subset(dataset_train, idxs), batch_size=100, shuffle=False)
                    out_temp, _ = get_output(loader_temp, net_temp.to(args.device), args, False, criterion_red)
                    preds_local = np.argmax(F.softmax(torch.tensor(out_temp), dim=1).cpu().numpy(), axis=1)

                probs = F.softmax(torch.tensor(outputs), dim=1).numpy()
                max_probs = np.max(probs, axis=1)
                preds_global = np.argmax(probs, axis=1)
                zeta = args.zeta 

                final_train_list = []
                for i, global_id in enumerate(idxs):
                    s_type = types[i]
                    if s_type == 0: # Clean
                        final_train_list.append(global_id)
                    elif s_type == 2: # Complex
                        final_train_list.append(global_id)
                    elif s_type == 1: # Noisy
                        is_kept = False
                        # 此时模型已热身，zeta 筛选开始生效
                        if max_probs[i] >= zeta:
                            if preds_local is not None and preds_global[i] == preds_local[i]:
                                global_pseudo_labels[global_id] = preds_global[i]
                                final_train_list.append(global_id)
                                is_kept = True
                        if not is_kept:
                            pass 
                
                client_train_idxs[idx] = final_train_list

        # --- PHASE 2: 联邦训练 (保持不变) ---
        w_locals = []
        loss_locals = []
        
        for idx in idxs_users:
            if len(client_train_idxs[idx]) == 0:
                continue

            # LocalUpdate 内部会根据 global_sample_types 使用不同的 Loss
            # Warm-up 时全为 0 -> 全用 CE Loss
            local = LocalUpdate(args, dataset_train, client_train_idxs[idx], 
                                sample_types=global_sample_types, 
                                pseudo_labels=global_pseudo_labels)
            
            w, loss = local.update_weights(net=copy.deepcopy(netglob).to(args.device), 
                                           seed=args.seed, w_g=netglob, epoch=args.local_ep)
            w_locals.append(w)
            loss_locals.append(loss)
            
            # --- PHASE 3: 贡献评估 (保持不变) ---
            if round_idx == 0:
                phi = 1.0
            else:
                phi = compute_gradient_alignment(w, w_glob_prev)
            
            # 【修改点 2】区分 Warm-up 和正式阶段
            if round_idx < WARMUP_ROUNDS:
                # Warm-up 阶段：不更新信誉，仅打印日志
                log_str = f"Round {round_idx}, Client {idx}: [Warm-up] Skipping Reputation Update."
            else:
                # 正式阶段：正常计算
                Sn = client_noise_rates[idx]
                Ci = client_class_diversity[idx]
                
                # 激励公式
                val_update = phi - alpha2 * Sn + alpha3 * Ci
                client_reputation[idx] += alpha1 * val_update
                
                log_str = f"Round {round_idx}, Client {idx}: Sn={Sn:.3f}, Ci={Ci:.3f}, Phi={phi:.3f}, Rep={client_reputation[idx]:.3f}"
            
            print(log_str)
            f_acc.write(log_str + "\n")

        # --- PHASE 4: 自适应聚合 (保持不变) ---
        if len(w_locals) > 0:
            valid_users = [u for u in idxs_users if len(client_train_idxs[u]) > 0]
            if round_idx < WARMUP_ROUNDS:
                 # Warm-up 期间使用简单的平均聚合 (FedAvg)，因为 Sn 不准
                agg_weights = [len(client_train_idxs[u]) for u in valid_users]
                w_glob = FedAvg(w_locals,agg_weights)
            else:
                all_Sn = list(client_noise_rates.values())
                max_Sn = max(all_Sn) if len(all_Sn) > 0 and max(all_Sn) > 0 else 1.0
                valid_users = [u for u in idxs_users if len(client_train_idxs[u]) > 0]
                
                agg_weights = []
                for idx in valid_users:
                    Sn = client_noise_rates[idx]
                    eta = Sn / max_Sn
                    m_i = len(client_train_idxs[idx])
                    w = m_i * np.exp(-eta)
                    agg_weights.append(w)
                
                w_glob = FedAvg(w_locals, agg_weights)
            
            w_glob_prev = copy.deepcopy(netglob.state_dict())
            netglob.load_state_dict(w_glob)
        
        # --- 测试 ---
        acc_test = globaltest(copy.deepcopy(netglob), dataset_test, args)
        print(f"Round {round_idx} Global Accuracy: {acc_test:.2%}")
        f_acc.write(f"Round {round_idx} Global Accuracy: {acc_test:.4f}\n")
        f_acc.flush()
        
        acc_history.append(acc_test) # 记录当前准确率
        
        try:
            plt.figure() 
            plt.plot(range(len(acc_history)), acc_history, marker='.', label='Global Accuracy')
            plt.title(f"Training Accuracy (Round {round_idx})")
            plt.xlabel("Communication Rounds")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.legend()
            
            # [修改] 图片保存路径也使用新的 experiment_dir
            save_path = os.path.join(experiment_dir, "accuracy_curve.png")
            plt.savefig(save_path)
            plt.close() 
        except Exception as e:
            print(f"Plotting failed: {e}")

    f_acc.close()