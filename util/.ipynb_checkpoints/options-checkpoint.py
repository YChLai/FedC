# util/options.py
# python version 3.7.1
# -*- coding: utf-8 -*-



"""
1. 激励与贡献评估参数 (Incentive & Valuation)
这些参数用于计算客户端的价值 Vi 和更新奖励 ri^t。公式原型参考：ri^t = ri^(t-1) + alpha1 (φi^t + alpha2 Pi + alpha3 Ci)。
    alpha1 (默认 0.1)：
        解释：贡献更新的整体步长或权重。控制历史贡献值与当前轮次新评估指标（梯度、噪声、多样性）之间的平衡。
    alpha2 (默认 1.5)：
        解释：噪声率项 (Pi 或相关项) 的权重。
        逻辑：用于放大或缩小 “噪声率” 对最终贡献值的影响。由于目标是鼓励高质量数据，该项通常与噪声率成反比（或作为惩罚项），权重越大，低噪声（高质量）对客户端价值的提升越明显。
    alpha3 (默认 0.4)：
        解释：类别多样性项 (Ci) 的权重。
        逻辑：权重越大，拥有更多类别数据的客户端获得的激励越多。
2. 三部分损失函数权重 (Tri-Partition Loss Weights)
    用于计算总损失 l_train = lambda_c lc + lambda_n ln + lambda_h lh [cite: 27, 28]。注意代码中这三个值的和理论上应该调整为 1.0，或者在代码内部归一化。
    lambda_c (默认 0.4)：
        解释：干净样本集 (Clean Set) 的损失权重。
        作用：控制网络向置信度高的正确标签学习的程度。
    lambda_n (默认 0.5)：
        解释：噪声样本集 (Noisy Set) 的损失权重。
        作用：控制经过 Mixup 和一致性正则化后的噪声样本对模型更新的影响。通常设置较高以充分利用这部分大量的数据进行无监督 / 半监督学习。
    lambda_h (默认 0.3)：
        解释：复杂样本集 (Complex Set) 的损失权重。
        作用：控制 GCE 损失的影响，专注于挖掘难样本（Hard Examples）的信息，同时通过 GCE 抵抗其中潜在的错误标签。
3. 噪声修正阈值 (Correction Thresholds)
    zeta (默认 0.85)：
        解释：重标记置信度阈值。
        逻辑：对应文档中的 zeta = 0.75 (FedDiv 默认) 或代码中的默认值。在处理噪声样本时，只有当全局模型对某个样本的预测概率 pc ≥ zeta 时，才认为该预测是可信的 “伪标签”，并用于替换原始噪声标签。
    clean_set_thres (默认 0.1)：
        解释：干净集合筛选阈值。
        逻辑：基于 GMM 拟合后的后验概率，用于划分样本是否属于 “干净” 集合的概率阈值。
4. 联邦学习基础参数
    beta：
        解释：Local Proximal Term 系数。
        逻辑：FedCorr/FedProx 中用于限制本地模型偏离全局模型太远的正则化项。在处理 Non-IID 数据时非常重要，能提高训练稳定性。
    level_n_system & level_n_lowerb：
        解释：分别控制系统中噪声客户端的比例 (ρ) 和 客户端噪声水平的下界 (τ)。这两个参数共同定义了实验环境中的异构噪声分布。
"""

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--iteration1', type=int, default=5, help="enumerate iteration in preprocessing stage")
    parser.add_argument('--total_rounds', type=int, default=1000, help="rounds of training in fine_tuning stage")
    parser.add_argument('--warmup_rounds', type=int, default=200, help="rounds of training in usual training stage")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs")
    parser.add_argument('--frac1', type=float, default=0.1, help="fration of selected clients in preprocessing stage")
    parser.add_argument('--frac2', type=float, default=0.1, help="fration of selected clients in fine-tuning and usual training stage")

    parser.add_argument('--num_users', type=int, default=100, help="number of uses: K")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.04, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum, default 0.9")
    parser.add_argument('--beta', type=float, default=0.1, help="coefficient for local proximal，0 for fedavg, 1 for fedprox, 5 for noise fl")

    # noise arguments
    parser.add_argument('--LID_k', type=int, default=20, help="lid")
    parser.add_argument('--level_n_system', type=float, default=0.4, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.5, help="lower bound of noise level")

    # correction
    parser.add_argument('--relabel_ratio', type=float, default=0.5, help="proportion of relabeled samples among selected noisy samples")
    parser.add_argument('--confidence_thres', type=float, default=0.5, help="threshold of model's confidence on each sample")
    parser.add_argument('--clean_set_thres', type=float, default=0.1, help="threshold of estimated noise level to filter 'clean' set used in fine-tuning stage")

    # ablation study
    parser.add_argument('--fine_tuning', action='store_false', help='whether to include fine-tuning stage')
    parser.add_argument('--correction', action='store_false', help='whether to correct noisy labels')

    # other arguments
    # parser.add_argument('--server', type=str, default='none', help="type of server")
    parser.add_argument('--model', type=str, default='resnet18', help="model name")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--pretrained', action='store_true', help="whether to use pre-trained model")
    parser.add_argument('--iid', action='store_true', help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float, default=0.7, help="non iid sampling prob for class")
    parser.add_argument('--alpha_dirichlet', type=float, default=10)
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--seed', type=int, default=13, help="random seed, default: 1")
    parser.add_argument('--mixup', action='store_true', help="whether to use mixup")
    parser.add_argument('--alpha', type=float, default=2.0, help="mixup alpha")

    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

    # [NEW] Incentive & Loss Hyperparameters
    parser.add_argument('--alpha1', type=float, default=0.1, help="Incentive weight 1")
    parser.add_argument('--alpha2', type=float, default=1.5, help="Incentive weight 2")
    parser.add_argument('--alpha3', type=float, default=0.4, help="Incentive weight 3")
    parser.add_argument('--lambda_c', type=float, default=0.4, help="Clean loss weight")
    parser.add_argument('--lambda_n', type=float, default=0.5, help="Noisy loss weight")
    parser.add_argument('--lambda_h', type=float, default=0.3, help="Hard/Complex loss weight")
    parser.add_argument('--zeta', type=float, default=0.85, help="Confidence threshold for relabeling")

    return parser.parse_args()