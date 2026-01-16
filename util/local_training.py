# util/local_training.py
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
from .loss import GCELoss, MixupMSELoss, mixup_data, mixup_criterion

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, sample_types=None, pseudo_labels=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        # sample_types: {global_idx: type} (0: Clean, 1: Noisy, 2: Complex)
        self.sample_types = sample_types 
        # pseudo_labels: {global_idx: label} (存储重标记后的标签)
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        original_idx = self.idxs[item]
        image, label = self.dataset[original_idx]
        
        # 如果有伪标签（重标记），则使用伪标签
        if self.pseudo_labels is not None and original_idx in self.pseudo_labels:
            label = self.pseudo_labels[original_idx]
            
        # 获取样本类型，默认为 Complex (2)
        s_type = 2
        if self.sample_types is not None:
            s_type = self.sample_types.get(original_idx, 2)
            
        return image, label, s_type

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, sample_types=None, pseudo_labels=None):
        self.args = args
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_gce = GCELoss(q=0.7) # GCE q parameter
        self.loss_mse = MixupMSELoss() # New MSE Loss for Noisy samples
        
        self.sample_types = sample_types
        self.pseudo_labels = pseudo_labels
        
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, sample_types, pseudo_labels), 
                                    batch_size=self.args.local_bs, shuffle=True)

    def update_weights(self, net, seed, w_g, epoch, mu=1, lr=None):
        net.train()
        if lr is None:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        
        # 损失函数权重 (lambda_c + lambda_n + lambda_h = 1)
        lam_c, lam_n, lam_h = 0.4, 0.3, 0.3 

        for iter in range(epoch):
            batch_loss = []
            for batch_idx, (images, labels, s_types) in enumerate(self.ldr_train):
                images, labels, s_types = images.to(self.args.device), labels.to(self.args.device), s_types.to(self.args.device)
                net.zero_grad()
                
                loss = 0
                
                # Masks
                clean_mask = (s_types == 0)
                noisy_mask = (s_types == 1)
                complex_mask = (s_types == 2)
                
                # --- 1. Clean Samples: Cross Entropy ---
                if clean_mask.sum() > 0:
                    pred_c = net(images[clean_mask])
                    loss += lam_c * self.loss_ce(pred_c, labels[clean_mask].long())

                # --- 2. Noisy Samples: Mixup with MSE ---
                # 文档要求：优先“同类重标记样本” 或“重标记样本 - 干净样本”混合
                # 并且使用均方误差 (MSE) 约束
                if noisy_mask.sum() > 0:
                    inputs_n = images[noisy_mask]
                    labels_n = labels[noisy_mask].long()
                    
                    # 尝试寻找 Clean 样本进行混合
                    inputs_clean = images[clean_mask]
                    labels_clean = labels[clean_mask].long()
                    
                    if len(inputs_clean) > 0:
                        # 如果当前 batch 有 Clean 样本，优先从 Clean 中随机采样作为 Mixup 对象
                        # 生成随机索引
                        rand_idx = torch.randint(0, len(inputs_clean), (len(inputs_n),)).to(self.args.device)
                        inputs_partner = inputs_clean[rand_idx]
                        labels_partner = labels_clean[rand_idx]
                    else:
                        # 如果当前 batch 没有 Clean 样本，则在 Noisy 样本内部 shuffle
                        idx = torch.randperm(len(inputs_n)).to(self.args.device)
                        inputs_partner = inputs_n[idx]
                        labels_partner = labels_n[idx]

                    # 执行 Mixup
                    alpha = self.args.alpha
                    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
                    
                    inputs_mix = lam * inputs_n + (1 - lam) * inputs_partner
                    pred_mix = net(inputs_mix)
                    
                    # 使用 MSE 计算损失 (公式 4-8)
                    loss += lam_n * self.loss_mse(pred_mix, labels_n, labels_partner, lam)

                # --- 3. Complex Samples: GCE Loss ---
                if complex_mask.sum() > 0:
                    pred_h = net(images[complex_mask])
                    loss += lam_h * self.loss_gce(pred_h, labels[complex_mask].long())

                # FedProx (Proximal term)
                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(w_g.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += self.args.beta * mu * w_diff

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0

def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total