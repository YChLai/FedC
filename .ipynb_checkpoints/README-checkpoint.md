# Federated Learning with Noise Correction & Incentive Mechanism

此项目在 **FedCorr** (CVPR 2022) 的基础上进行了扩展，引入了基于 **FairGraphFL** 的贡献评估与激励机制，以及基于 **FedDiv** 改进的自适应噪声标签处理框架。

## 1. 项目概述 (Project Overview)

本项目旨在解决联邦学习（Federated Learning, FL）中存在的两个核心问题：
1.  **数据异构与标签噪声**：客户端数据存在 Non-IID 和不同程度的标签噪声。
2.  **贡献评估与激励**：如何公平地评估不同数据质量（噪声率、多样性）客户端的贡献并给予激励。

### 核心改进 (Key Improvements)

#### 1.1 评估贡献和激励分配 (Contribution Assessment & Incentive)
在 FairGraphFL 的基础上进行了改进，使用 **Client Valuation Function** 衡量客户贡献。贡献值由以下三个标准共同决定：
* **梯度对齐 (Gradient Alignment)**：衡量本地更新与全局更新的一致性。
* **噪声率 (Noise Rate, $S_\eta$)**：通过全局噪声滤波器（基于 GMM）计算，识别客户端的噪声水平。噪声率越低，数据质量越高。
* **类别多样性 (Class Diversity, $C_i$)**：衡量客户端数据类别的丰富程度。类别越丰富，模型泛化能力越强。

激励分配机制结合了稀疏梯度法和代币系统，奖励高质量数据贡献者，惩罚低质量或恶意客户端。

#### 1.2 适应噪声标签的联邦学习框架 (Robust FL for Noisy Labels)
为了提升在噪声数据下的准确率，设计了自适应的样本筛选和损失计算策略：
* **全局噪声滤波器**：聚合本地 GMM 参数，构建全局噪声分布模型，指导样本分类。
* **样本三分类策略**：结合全局和本地滤波器的预测结果，将样本划分为：
    * **干净样本集 (Clean Set)**：两方都认为是正确标签，损失值最小。
    * **噪声样本集 (Noisy Set)**：两方都认为是噪声标签，损失值最大。
    * **复杂样本集 (Complex Set)**：介于两者之间，包含“难学的正确样本”和“易混淆的噪声样本”。
* **自适应聚合**：聚合权重根据客户端噪声率动态调整 ($w_g \propto m_i e^{-\eta_i}$)，降低高噪声客户端的权重。

---

## 2. 损失函数设计 (Loss Functions)

针对不同的样本集合，采用差异化的损失函数策略以优化训练：

$$l_{train} = \lambda_c l_c + \lambda_n l_n + \lambda_h l_h$$

其中 $\lambda_c + \lambda_n + \lambda_h = 1$。

1.  **干净样本 ($l_c$)**：
    * 直接使用**交叉熵损失 (Cross-Entropy Loss)**，充分利用正确标签信息。

2.  **噪声样本 ($l_n$)**：
    * **预处理**：进行“重标记 (Relabeling)”和“预测一致性筛选 (PCS)”。仅保留模型预测置信度 $p_c \ge \zeta$ (默认为 0.75) 且全局/本地模型预测一致的样本。
    * **Mixup 增强**：对筛选后的样本使用 Mixup 数据增强，优先混合“同类重标记样本”或“干净样本”。
    * **损失计算**：使用**均方误差 (MSE)** 约束预测分布的一致性，避免过拟合错误标签。

3.  **复杂样本 ($l_h$)**：
    * 使用**广义交叉熵损失 (Generalized Cross-Entropy, GCE)**。GCE 结合了 MAE 的鲁棒性和 CE 的收敛速度，适合处理包含难样本和部分噪声的数据集。

---

## 3. 运行环境与参数 (Environment & Arguments)

### 3.1 依赖 (Requirements)
* Python 3.7+
* PyTorch
* NumPy

### 3.2 运行示例 (Usage Examples)

**在 CIFAR-10 上训练，启用噪声修正和激励机制：**

```bash
cd OursFL/
python main.py

如果需要修改参数，可在util/options.py中修改，运行时会将本次训练的参数配置保存在record中