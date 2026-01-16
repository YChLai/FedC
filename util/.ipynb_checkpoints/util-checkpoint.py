import numpy as np
import torch
import torch.nn.functional as F
import copy
from scipy.spatial.distance import cdist


def add_noise(args, y_train, dict_users):
    np.random.seed(args.seed)

    gamma_s = np.random.binomial(1, args.level_n_system, args.num_users)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial

    y_train_noisy = copy.deepcopy(y_train)

    real_noise_level = np.zeros(args.num_users)
    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, 10, len(noisy_idx))
        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
        real_noise_level[i] = noise_ratio
    return (y_train_noisy, gamma_s, real_noise_level)


def get_output(loader, net, args, latent=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()
            if latent == False:
                outputs = net(images)
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs = net(images, True)
            # 修复：确保loss是标量/一维数组，避免拼接报错
            if criterion is not None:
                loss = criterion(outputs, labels)
                # 将tensor转换为numpy时，先处理标量情况
                loss = loss.item() if loss.dim() == 0 else loss.cpu().numpy()
            if i == 0:
                output_whole = np.array(outputs.cpu())
                if criterion is not None:
                    loss_whole = np.array([loss]) if np.isscalar(loss) else loss
            else:
                output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                if criterion is not None:
                    loss = np.array([loss]) if np.isscalar(loss) else loss
                    loss_whole = np.concatenate((loss_whole, loss), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


def lid_term(X, batch, k=20):
    eps = 1e-6
    X = np.asarray(X, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)
    
    # 定义LID计算的核心函数，增加eps避免除0/对数错误
    def calculate_lid(v):
        # v是每个样本的k个近邻距离，避免分母为0
        denominator = np.sum(np.log(v / (v[-1] + eps))) + eps
        return -k / denominator
    
    # 计算样本间的距离矩阵
    distances = cdist(X, batch)

    # 获取每个样本的前k个近邻（排除自身，取1:k+1）
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
    
    # 核心修复：将ogrid元组转为列表，支持赋值
    idx = list(np.ogrid[:sort_indices.shape[0], :sort_indices.shape[1]])
    idx[1] = sort_indices  # 现在可以正常赋值
    
    # 提取排序后的距离值
    distances_ = distances[tuple(idx)]
    
    # 计算每个样本的LID值
    lids = np.apply_along_axis(calculate_lid, axis=1, arr=distances_)
    return lids