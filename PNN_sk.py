import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 设置中文显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以保证结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class VAEClassifier(nn.Module):
    def __init__(self, input_size, latent_dim=16, hidden_sizes=[64, 32], num_classes=2, dropout_rate=0.3):
        super(VAEClassifier, self).__init__()
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        self.encoder_bn = nn.ModuleList()
        
        # 第一层编码器
        self.encoder_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.encoder_bn.append(nn.BatchNorm1d(hidden_sizes[0]))
        
        # 其他编码器层
        for i in range(len(hidden_sizes)-1):
            self.encoder_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.encoder_bn.append(nn.BatchNorm1d(hidden_sizes[i+1]))
        
        # 均值和对数方差层
        self.fc_mu = nn.Linear(hidden_sizes[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_sizes[-1], latent_dim)
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.decoder_bn = nn.ModuleList()
        
        # 第一层解码器
        self.decoder_layers.append(nn.Linear(latent_dim, hidden_sizes[-1]))
        self.decoder_bn.append(nn.BatchNorm1d(hidden_sizes[-1]))
        
        # 其他解码器层
        for i in range(len(hidden_sizes)-1, 0, -1):
            self.decoder_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i-1]))
            self.decoder_bn.append(nn.BatchNorm1d(hidden_sizes[i-1]))
        
        # 最终输出层
        self.decoder_output = nn.Linear(hidden_sizes[0], input_size)
        
        # Classifier
        self.classifier_layers = nn.ModuleList()
        self.classifier_bn = nn.ModuleList()
        
        # 分类器层
        self.classifier_layers.append(nn.Linear(latent_dim, hidden_sizes[-1]))
        self.classifier_bn.append(nn.BatchNorm1d(hidden_sizes[-1]))
        
        for i in range(len(hidden_sizes)-1, 0, -1):
            self.classifier_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i-1]))
            self.classifier_bn.append(nn.BatchNorm1d(hidden_sizes[i-1]))
        
        # 分类器输出层
        self.classifier_output = nn.Linear(hidden_sizes[0], num_classes)
        
        # 其他组件
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        self.latent_dim = latent_dim
        
    def encode(self, x):
        for i, (layer, bn) in enumerate(zip(self.encoder_layers, self.encoder_bn)):
            x = layer(x)
            x = bn(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # 计算均值和对数方差
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧: z = mu + std * epsilon"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # 在评估模式下直接使用均值
    
    def decode(self, z):
        for i, (layer, bn) in enumerate(zip(self.decoder_layers, self.decoder_bn)):
            z = layer(z)
            z = bn(z)
            z = self.activation(z)
            z = self.dropout(z)
        
        # 最终输出层重构原始输入
        reconstructed = self.decoder_output(z)
        return reconstructed
    
    def classify(self, z):
        for i, (layer, bn) in enumerate(zip(self.classifier_layers, self.classifier_bn)):
            z = layer(z)
            z = bn(z)
            z = self.activation(z)
            z = self.dropout(z)
        
        # 分类输出
        logits = self.classifier_output(z)
        return logits
    
    def forward(self, x):
        # 编码
        mu, logvar = self.encode(x)
        
        # 采样潜在变量
        z = self.reparameterize(mu, logvar)
        
        # 解码
        reconstructed = self.decode(z)
        
        # 分类
        logits = self.classify(z)
        
        return reconstructed, mu, logvar, logits

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def vae_loss_function(recon_x, x, mu, logvar, class_output, target, beta=0.001):
    """
    计算VAE+分类器的组合损失函数
    params:
        - recon_x: 重构的输入
        - x: 原始输入
        - mu: 潜在变量均值
        - logvar: 潜在变量对数方差
        - class_output: 分类器输出
        - target: 真实标签
        - beta: KL散度权重
    """
    # 重构损失 - 均方误差
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD / x.size(0)  # 批量平均化
    
    # 分类损失 - 交叉熵
    CE = F.cross_entropy(class_output, target)
    
    # 总损失 = 重构损失 + beta*KL散度 + 分类损失
    return MSE + beta * KLD + CE, MSE, KLD, CE

def train_model(model, train_loader, val_loader, optimizer, scheduler, 
                num_epochs=1000, device='cuda', beta=0.001):
    model = model.to(device)
    early_stopping = EarlyStopping(patience=20)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'recon_loss': [], 'kl_loss': [], 'class_loss': []
    }
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        class_loss_sum = 0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar, class_output = model(inputs)
            
            loss, recon_loss, kl_loss, class_loss = vae_loss_function(
                recon_batch, inputs, mu, logvar, class_output, labels, beta=beta)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            class_loss_sum += class_loss.item()
            
            _, predicted = torch.max(class_output.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                recon_batch, mu, logvar, class_output = model(inputs)
                loss, _, _, _ = vae_loss_function(
                    recon_batch, inputs, mu, logvar, class_output, labels, beta=beta)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(class_output.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['recon_loss'].append(recon_loss_sum / len(train_loader))
        history['kl_loss'].append(kl_loss_sum / len(train_loader))
        history['class_loss'].append(class_loss_sum / len(train_loader))
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, '
                  f'Val Acc: {val_acc:.4f}')
        
        # 早停检查
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"早停在第 {epoch+1} 轮")
            break
    
    return history

def main():
    # 设置随机种子
    set_seed(42)
    """
    # 加载SMOTE处理后的训练数据
    print("加载SMOTE处理后的训练数据...")
    data_SMOTE_df = pd.read_excel('data_sk.xlsx')
    data_SMOTE = np.array(data_SMOTE_df)
    
    x_SMOTE = data_SMOTE[:, :-1]
    y_SMOTE = data_SMOTE[:, -1].astype(int)
    
    # 分割SMOTE数据为训练集和验证集
    x_train, x_val, y_train, y_val = train_test_split(
        x_SMOTE, y_SMOTE, test_size=0.2, random_state=42, stratify=y_SMOTE
    )
    """
    # 加载原始数据作为测试集
    print("加载原始测试数据...")
    data_df = pd.read_csv('ALL.csv')
    data = np.array(data_df)
    x = data[:, :-1]
    y = data[:, -1].astype(int)
    
    # 从原始数据中分出测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 数据标准化
    print("数据标准化...")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    
    # 转换为PyTorch张量
    train_dataset = TensorDataset(
        torch.FloatTensor(x_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(x_val),
        torch.LongTensor(y_val)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建VAE+分类器模型
    print("创建变分自编码器+分类器模型...")
    input_size = x_train.shape[1]
    latent_dim = 16  # 潜在空间维度
    
    model = VAEClassifier(
        input_size=input_size,
        latent_dim=latent_dim,
        hidden_sizes=[64, 32],
        num_classes=2,
        dropout_rate=0.3
    )

    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, verbose=True
    )
    
    # 训练模型
    print("\n开始训练...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # KL散度权重参数
    beta = 0.001
    
    history = train_model(
        model, train_loader, val_loader,
        optimizer, scheduler,
        num_epochs=1000, device=device, beta=beta
    )
    
    # 模型评估
    print("\n模型评估...")
    model.eval()
    with torch.no_grad():
        # 测试集预测
        test_tensor = torch.FloatTensor(x_test).to(device)
        _, _, _, test_outputs = model(test_tensor)
        test_probs = torch.softmax(test_outputs, dim=1)
        test_predicted = torch.argmax(test_outputs, dim=1).cpu().numpy()
        
        # 训练集预测
        train_tensor = torch.FloatTensor(x_train).to(device)
        _, _, _, train_outputs = model(train_tensor)
        train_probs = torch.softmax(train_outputs, dim=1)
        train_predicted = torch.argmax(train_outputs, dim=1).cpu().numpy()
    
    # 输出分类报告
    print("\n训练集分类报告：")
    print(classification_report(y_train, train_predicted))
    print("\n测试集分类报告：")
    print(classification_report(y_test, test_predicted))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    sns.heatmap(confusion_matrix(y_train, train_predicted),
                annot=True, fmt='d', cmap='Blues')
    plt.title('训练集混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    plt.subplot(122)
    sns.heatmap(confusion_matrix(y_test, test_predicted),
                annot=True, fmt='d', cmap='Blues')
    plt.title('测试集混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    plt.tight_layout()
    plt.show()
    
    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 绘制不同损失组件
    plt.figure(figsize=(12, 5))
    plt.plot(history['recon_loss'], label='重构损失')
    plt.plot(history['kl_loss'], label='KL散度')
    plt.plot(history['class_loss'], label='分类损失')
    plt.title('VAE不同损失组件')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    
    # 训练集ROC
    fpr_train, tpr_train, _ = roc_curve(y_train, train_probs[:, 1].cpu().numpy())
    train_auc = auc(fpr_train, tpr_train)
    plt.plot(fpr_train, tpr_train, label=f'训练集 (AUC = {train_auc:.3f})')
    
    # 测试集ROC
    fpr_test, tpr_test, _ = roc_curve(y_test, test_probs[:, 1].cpu().numpy())
    test_auc = auc(fpr_test, tpr_test)
    plt.plot(fpr_test, tpr_test, label=f'测试集 (AUC = {test_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='随机')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC曲线')
    plt.legend()
    plt.show()
    
    # 可视化潜在空间
    if latent_dim == 2:  # 如果潜在空间是2D的，可以直接可视化
        with torch.no_grad():
            # 获取所有训练数据的潜在表示
            train_tensor = torch.FloatTensor(x_train).to(device)
            mu, _ = model.encode(train_tensor)
            z = mu.cpu().numpy()
            
            # 绘制潜在空间
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(z[:, 0], z[:, 1], c=y_train, cmap='viridis', alpha=0.8)
            plt.colorbar(scatter, label='类别')
            plt.title('VAE潜在空间可视化')
            plt.xlabel('潜在维度1')
            plt.ylabel('潜在维度2')
            plt.tight_layout()
            plt.show()
    
    print(f"\n最终性能指标:")
    print(f"训练集准确率: {accuracy_score(y_train, train_predicted):.4f}")
    print(f"测试集准确率: {accuracy_score(y_test, test_predicted):.4f}")
    print(f"训练集 AUC: {train_auc:.4f}")
    print(f"测试集 AUC: {test_auc:.4f}")

if __name__ == "__main__":
    main()