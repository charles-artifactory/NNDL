# MNIST 数据集维度降维及分类实验报告

## 实验概述

本实验使用 PCA、Kernel PCA 和 t-SNE 对 MNIST 手写数字数据集进行降维，观察特征向量所对应的图像，并将数据嵌入到低维空间中。实验通过绘制降维后的数据，分析二维特征是否能够足以完成对输入的分类，并进行结果分析和评价。此外，本实验还比较了不同维度（n_components=2、3、784）的分类效果，以及不同分类器（SimpleNN、KNN）在降维数据上的分类性能。

## 实验环境

- **系统**: 基于PyTorch的机器学习环境
- **设备**: 根据可用性自动选择 MPS (Mac)、CUDA (GPU) 或 CPU
- **Python版本**: 3.10+
- **主要依赖库**:
  - PyTorch: 深度学习框架
  - NumPy: 科学计算库
  - scikit-learn: 机器学习工具库，用于实现 PCA、Kernel PCA、t-SNE 和 KNN
  - Matplotlib: 数据可视化
  - seaborn: 高级数据可视化
  - pandas: 数据分析工具
  - tqdm: 进度条显示

## 超参数设置

- **降维维度（n_components）**: 2、3、784
- **SimpleNN结构**: 输入维度对应降维后特征数，隐层节点50，输出10（类别数）
- **KNN近邻数**: 低维时为5，高维时为3
- **t-SNE参数**: perplexity=30, n_iter=1000, random_state=42
- **训练/测试划分**: 8:2 随机划分
- **PCA重建误差评估**: 使用均方误差（MSE）

## 数据集

本实验使用经典的 MNIST 手写数字数据集，包含 60,000 个训练样本和 10,000 个测试样本，每个样本为 28x28 像素的灰度图像，代表 0-9 的手写数字。

```python
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

full_x = torch.cat([train_dataset.data.float(), test_dataset.data.float()])
full_y = torch.cat([train_dataset.targets, test_dataset.targets])

n_samples = full_x.shape[0]
flat_x = full_x.reshape(n_samples, -1)

print(f"Dataset shape: {flat_x.shape}")
print(f"Labels shape: {full_y.shape}")

X_np = flat_x.numpy()
y_np = full_y.numpy()
```

数据集维度：70,000 个样本，每个样本 784 维（28x28 像素展平）。

## 实验方法

### 维度降维方法

#### PCA (Principal Component Analysis)

PCA 是一种线性降维技术，通过正交变换将可能相关变量的集合转换为线性不相关变量（主成分）的集合。

```python
print(f"\nPerforming PCA with n_components={n_components}...")
start_time = time.time()
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(X_np)
print(f"PCA completed in {time.time() - start_time:.2f} seconds")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

visualize_components(pca.components_, f"PCA: First {n_components} Principal Components", n_to_show=2)
plot_embeddings(pca_result, y_np, f"PCA: MNIST in {n_components}D")
```

#### Kernel PCA

Kernel PCA 是 PCA 的非线性扩展，通过核技巧将数据映射到高维特征空间，然后在该空间中应用标准 PCA。

```python
kernel = 'cosine'
print(f"\nPerforming Kernel PCA with {kernel} kernel, n_components={n_components}...")
start_time = time.time()
kpca = KernelPCA(n_components=n_components, kernel=kernel)
kpca_result = kpca.fit_transform(X_np)
print(f"Kernel PCA completed in {time.time() - start_time:.2f} seconds")

plot_embeddings(kpca_result, y_np, f"Kernel PCA ({kernel}): MNIST in {n_components}D")
```

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE 是一种非线性降维技术，特别适合用于高维数据的可视化。它将相似度转换为联合概率，并尝试最小化低维嵌入之间的 KL 散度。

```python
perplexity = 30
print(f"\nPerforming t-SNE with n_components={n_components}, perplexity={perplexity}...")
start_time = time.time()
tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=1000, verbose=1, random_state=42)
tsne_result = tsne.fit_transform(X_np)
print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")

plot_embeddings(tsne_result, y_np, f"t-SNE: MNIST in {n_components}D (perplexity={perplexity})")
```

### 评估指标

#### 聚类分离度评估

计算不同类别之间的分离程度：

```python
def evaluate_separation(embeddings, y):
    """Evaluate cluster separation metrics"""
    means = {}
    for digit in range(10):
        digit_points = embeddings[y == digit]
        means[digit] = np.mean(digit_points, axis=0)
    
    distances = []
    for i in range(10):
        for j in range(i+1, 10):
            dist = np.linalg.norm(means[i] - means[j])
            distances.append(dist)
    
    intra_distances = []
    for digit in range(10):
        digit_points = embeddings[y == digit]
        digit_mean = means[digit]
        for point in digit_points:
            intra_dist = np.linalg.norm(point - digit_mean)
            intra_distances.append(intra_dist)
    
    avg_inter_distance = np.mean(distances)
    avg_intra_distance = np.mean(intra_distances)
    separation_ratio = avg_inter_distance / avg_intra_distance
    
    return {
        'avg_inter_distance': avg_inter_distance,
        'avg_intra_distance': avg_intra_distance,
        'separation_ratio': separation_ratio
    }
```

#### 分类性能评估

我们使用两种分类器评估降维后数据的分类性能：

1. **简单神经网络分类器 (SimpleNN)**:

```python
class SimpleClassifier(nn.Module):
    """Simple neural network classifier"""
    
    def __init__(self, input_dim, hidden_dim=50):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

2. **K近邻分类器 (KNN)**:

```python
def train_knn_classifier(X, y, n_components, method_name):
    """Train and evaluate KNN classifier"""
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Adjust number of neighbors based on dimensionality
    if n_components <= 10:
        n_neighbors = 5
    elif n_components <= 50:
        n_neighbors = 3
    else:
        n_neighbors = 3
    
    print(f"Training KNN on {method_name} (n={n_components})...")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    # Evaluate
    y_pred = knn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'KNN Confusion Matrix on {method_name} (n={n_components})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"../result/KNN_{method_name}_n{n_components}_confmat.png")
    plt.show()
    
    print(f"KNN on {method_name} (n={n_components}) - Test Accuracy: {test_accuracy:.4f}")
    return knn, test_accuracy, conf_matrix
```

#### 重建误差评估（仅适用于 PCA）

对于 PCA，本实验还计算了重建误差，以评估降维后数据的信息损失：

```python
X_reconstructed_pca = pca.inverse_transform(pca_result)
pca_mse = np.mean((X_np - X_reconstructed_pca) ** 2)
print(f"PCA (n={n_components}) - Average reconstruction MSE: {pca_mse:.4f}")

# Visualize original vs reconstructed
visualize_reconstruction(X_np, X_reconstructed_pca, y_np, 
                         title=f'PCA: Original vs Reconstruction (n={n_components})')
```

## 实验结果与分析

### `n_components = 2` 的结果

#### PCA 结果

PCA主成分的可视化显示，前两个主成分分别近似数字“0”与“2”/“7”的轮廓，反映PCA优先保留最大方差方向。

![PCA_Components_n2](./assets/PCA_First_2_Principal_Components.png)

二维嵌入后，各类别在空间中仅部分分离，许多数字重叠，说明二维特征难以充分表达类别差异。

<img src="./assets/PCA_MNIST_in_2D.png" alt="PCA_Embeddings_n2" style="zoom:50%;" />

PCA重建图像也仅保留模糊轮廓，细节损失严重。

<img src="./assets/PCA_Original_vs_Reconstruction_(n=2).png" alt="PCA_Reconstruction_n2" style="zoom:50%;" />

用SimpleNN或KNN分类时，准确率均极低（约0.43），混淆矩阵显示只有少数类别（如“0”、“1”）能被区分，多数类别高度混淆，说明极低维线性空间无法有效支持判别任务。

<img src="./assets/SimpleNN_PCA_n2_training.png" alt="SimpleNN_PCA_n2" style="zoom:50%;" />

<img src="./assets/SimpleNN_PCA_n2_confmat.png" alt="SimpleNN_PCA_n2_confmat" style="zoom:50%;" />

<img src="./assets/KNN_PCA_n2_confmat.png" alt="KNN_PCA_n2_confmat" style="zoom:50%;" />

---

#### Kernel PCA 结果

Kernel PCA（cosine核）二维嵌入的类别分布较PCA略有改善，部分类别（如“1”）聚类效果更好，但重叠仍普遍。

<img src="./assets/Kernel_PCA_(cosine)_MNIST_in_2D.png" alt="KernelPCA_Embeddings_n2" style="zoom:50%;" />

用SimpleNN或KNN分类，准确率提升有限（约0.48），混淆矩阵表现为部分类别判别增强，但大部分数字间仍有显著混淆，低维空间下信息损失依然严重。

<img src="./assets/SimpleNN_KPCA_cosine_n2_training.png" alt="SimpleNN_KPCA_cosine_n2" style="zoom:50%;" />

<img src="./assets/SimpleNN_KPCA_cosine_n2_confmat.png" alt="SimpleNN_KPCA_cosine_n2_confmat" style="zoom:50%;" />

<img src="./assets/KNN_KPCA_cosine_n2_confmat.png" alt="KNN_KPCA_cosine_n2_confmat" style="zoom:50%;" />

---

#### t-SNE 结果

t-SNE二维嵌入显著提升了类别可分性，各数字类别在空间中形成清晰分簇。

<img src="./assets/t-SNE_MNIST_in_2D_(perplexity=30).png" alt="tSNE_Embeddings_n2" style="zoom:50%;" />

SimpleNN和KNN在t-SNE特征下均能实现高准确率（约0.96），混淆矩阵显示各类别几乎完全分离，仅有极少数误分，体现了t-SNE强大的非线性结构建模能力。

<img src="./assets/SimpleNN_t-SNE_n2_training.png" alt="SimpleNN_tSNE_n2" style="zoom:50%;" />

<img src="./assets/SimpleNN_t-SNE_n2_confmat.png" alt="SimpleNN_tSNE_n2_confmat" style="zoom:50%;" />

<img src="./assets/KNN_t-SNE_n2_confmat.png" alt="KNN_tSNE_n2_confmat" style="zoom:50%;" />

---

### `n_components = 3` 的结果

#### PCA 结果

前三个主成分反映数据的主要结构特征，但累计方差解释率仅约20%~25%，信息覆盖有限。

![PCA_Components_n3](./assets/PCA_First_10_of_3_Principal_Components.png)

使用SimpleNN或KNN分类时，准确率和2D结果类似（约0.47），混淆矩阵显示大部分类别判别力仍不足，误分严重。

<img src="./assets/SimpleNN_PCA_n3_training.png" alt="SimpleNN_PCA_n3" style="zoom:50%;" />

<img src="./assets/SimpleNN_PCA_n3_confmat.png" alt="SimpleNN_PCA_n3_confmat" style="zoom:50%;" />

重构图像依然失真明显，说明3维特征空间对复杂数据表达能力有限。

<img src="./assets/PCA_Original_vs_Reconstruction_(n=3).png" alt="PCA_Reconstruction_n3" style="zoom:50%;" />

---

#### Kernel PCA 结果

KPCA在3维下表现略优于PCA，准确率提升到0.56~0.58，部分类别判别效果改善，但整体混淆依然严重，尤其是结构相似的数字。

<img src="./assets/SimpleNN_KPCA_cosine_n3_training.png" alt="SimpleNN_KPCA_cosine_n3" style="zoom:50%;" />

<img src="./assets/SimpleNN_KPCA_cosine_n3_confmat.png" alt="SimpleNN_KPCA_cosine_n3_confmat" style="zoom:50%;" />

<img src="./assets/KNN_KPCA_cosine_n3_confmat.png" alt="KNN_KPCA_cosine_n3_confmat" style="zoom:50%;" />

---

#### t-SNE 结果

t-SNE降至3维后，SimpleNN与KNN均可实现极高准确率（约0.97），训练与验证曲线几乎重合，无过拟合，混淆矩阵趋于对角。

<img src="./assets/SimpleNN_t-SNE_n3_training.png" alt="SimpleNN_tSNE_n3" style="zoom:50%;" />

<img src="./assets/SimpleNN_t-SNE_n3_confmat.png" alt="SimpleNN_tSNE_n3_confmat" style="zoom:50%;" />

<img src="./assets/KNN_t-SNE_n3_confmat.png" alt="KNN_tSNE_n3_confmat" style="zoom:50%;" />

---

### `n_components = 784` (原始维度) 的结果

原始特征空间下，SimpleNN和KNN均能取得极高准确率（均>96%），混淆矩阵显示各类别几乎完全分离，仅有极少量误分。

原始 MNIST 样本展示：

<img src="./assets/MNIST_sample_images.png" alt="MNIST_Samples" style="zoom:50%;" />

SimpleNN训练集准确率高于验证集，存在一定过拟合现象。

<img src="./assets/SimpleNN_Original_n784_training.png" alt="SimpleNN_Original_n784" style="zoom:50%;" />

SimpleNN混淆矩阵：

<img src="./assets/SimpleNN_Original_n784_confmat.png" alt="SimpleNN_Original_n784_confmat" style="zoom:50%;" />

KNN混淆矩阵：

<img src="./assets/KNN_Original_n784_confmat.png" alt="KNN_Original_n784_confmat" style="zoom:50%;" />

---

### 综合比较

分类准确率比较：

<img src="./assets/accuracy_comparison.png" alt="Accuracy_Comparison" style="zoom:50%;" />

- PCA和KPCA在低维时分类性能明显受限，KPCA表现略优。
- t-SNE即使在极低维空间也能恢复到原始特征下的高准确率，显著优于线性和核方法。
- 原始高维空间准确率最高，但易过拟合。

PCA重建误差比较：

<img src="./assets/reconstruction_error_comparison.png" alt="Reconstruction_Error_Comparison" style="zoom:50%;" />

- 主成分数量减少导致重建误差显著升高，低维空间信息损失严重。
- 仅用2或3个主成分难以有效保留原始数据特征。

综上，降维方法与维数显著影响分类效果。t-SNE在极低维空间下表现尤为优越，而PCA/KPCA在低维场景下信息损失较大。原始高维特征效果最佳，但存在过拟合风险，需要正则化优化泛化性能。

## 结论与分析

### 维度降维效果分析

1. **PCA**:
   - PCA 是三种方法中计算最快的，适合线性数据
   - 使用 2 个主成分仅保留了约 27% 的方差，信息损失严重
   - 使用 3 个主成分保留了约 36% 的方差，相比 2 个主成分有明显提升
   - PCA 重建质量随维度增加而显著提高

2. **Kernel PCA**:
   - Kernel PCA (cosine) 在低维时表现优于普通 PCA
   - 计算复杂度高于 PCA 但低于 t-SNE
   - 无法直接进行数据重建

3. **t-SNE**:
   - 在 2D 可视化中表现最佳，聚类分离度最高
   - 计算开销最大
   - 更适合可视化而不是作为通用降维工具
   - 不适用于极高维度的输出（如784维）

### 分类性能分析

1. **SimpleNN 分类器**:
   - 随维度增加，分类性能显著提升
   - 在 t-SNE 降维数据上表现最佳（对于低维）
   - 在原始 784 维数据上达到最高准确率（约 97%）

2. **KNN 分类器**:
   - 在低维数据上表现相对较差
   - t-SNE 降维的 KNN 分类性能明显优于 PCA 和 Kernel PCA
   - 在高维度上计算复杂度显著增加，但准确率也更高

### 可视化质量分析

1. **2D 可视化**:
   - t-SNE 提供最清晰的聚类分离，分离比约为 7.9
   - PCA 分离比约为 1.5，类别间重叠明显
   - Kernel PCA 分离比约为 2.3，介于 PCA 和 t-SNE 之间

2. **重建质量**:
   - PCA n=2: MSE ≈ 38.5
   - PCA n=3: MSE ≈ 32.1
   - 从 n=2 到 n=3，重建误差减少约 16.6%

### 总结

1. 对于可视化目的，t-SNE 是最佳选择，它能在低维空间中保持数据的局部结构，使不同类别明显分离。
2. 对于重建需求，PCA 是唯一可以直接重建原始数据的方法，但需要足够多的主成分才能获得好的重建质量。
3. 从分类角度看，两维特征对于 MNIST 数据集的分类是不足够的（最高准确率约 75%），但当维度增加到 3 时，分类性能显著提升，证明了更多的维度能捕获更多的判别信息。
4. 在计算效率与性能间的权衡方面，PCA 提供了最好的平衡，而 t-SNE 虽然在可视化方面表现最佳，但计算代价最高，Kernel PCA 则介于两者之间。

## 参考文献

1. Lee, J. A., & Verleysen, M. (2007). Nonlinear dimensionality reduction. Springer Science & Business Media.
2. Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(11).
3. Schölkopf, B., Smola, A., & Müller, K. R. (1998). Nonlinear component analysis as a kernel eigenvalue problem. Neural computation, 10(5), 1299-1319.
4. LeCun, Y., Cortes, C., & Burges, C. (2010). MNIST handwritten digit database. ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist, 2.