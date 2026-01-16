# util/local_training.py
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
from .loss import GCELoss, mixup_data, mixup_criterion

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
        
        # 损失函数权重 (参考文档公式 4-10 前后的定义: lambda_c + lambda_n + lambda_h = 1)
        # 这里设置为默认值，您可以根据实验调整
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

                # --- 2. Noisy Samples: Mixup ---
                if noisy_mask.sum() > 0:
                    inputs_n = images[noisy_mask]
                    labels_n = labels[noisy_mask].long()
                    if len(inputs_n) > 1:
                        inputs_mix, targets_a, targets_b, lam = mixup_data(inputs_n, labels_n, self.args.alpha, use_cuda=True)
                        pred_mix = net(inputs_mix)
                        loss += lam_n * mixup_criterion(self.loss_ce, pred_mix, targets_a, targets_b, lam)
                    else:
                        pred_n = net(inputs_n)
                        loss += lam_n * self.loss_ce(pred_n, labels_n)

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

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

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