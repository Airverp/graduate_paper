import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # 忽略警告

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

class BPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 16, 8], num_classes=2, dropout_rate=0.2):
        super(BPModel, self).__init__()
        
        # 使用ModuleList更灵活地构建网络层
        self.layers = nn.ModuleList()
        
        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_sizes[0])])
        
        # 构建隐藏层
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            
        # 输出层
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        for i, (layer, bn) in enumerate(zip(self.layers[:-1], self.batch_norms)):
            x = layer(x)
            x = bn(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = self.layers[-1](x)
        return x

class EarlyStopping: #设置早停机制，防止过拟合
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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=1000, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    early_stopping = EarlyStopping(patience=20)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
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
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 保存历史记录
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        """
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        """
        # 早停检查
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return history

def main():
    # 设置随机种子
    set_seed(42)
    
    # 加载数据
    data_SMOTE_df = pd.read_excel('data_sk.xlsx')
    data_SMOTE = np.array(data_SMOTE_df)
    
    x_SMOTE = data_SMOTE[:, :-1]
    y_SMOTE = data_SMOTE[:, -1].astype(int)
    
    # 分割数据
    x_train, x_val, y_train, y_val = train_test_split(
        x_SMOTE, y_SMOTE, test_size=0.2, random_state=42, stratify=y_SMOTE
        )   
    
    # 加载测试数据
    data_df = pd.read_csv('ALL.csv')
    data = np.array(data_df)
    x = data[:, :-1]
    y = data[:, -1].astype(int)
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    
    # 数据标准化
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
    
    # 创建模型
    input_size = x_train.shape[1]
    model = BPModel(
        input_size=input_size,
        hidden_sizes=[64, 32, 16],  # 可以调整隐藏层大小
        num_classes=2,
        dropout_rate=0.3
    )
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 添加L2正则化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, verbose=True
    )
    
    #记录模型训练开始时间
    start_time = time.time()

    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        num_epochs=1000, device=device
    )
    
    #记录模型训练结束时间
    end_time = time.time()

    # 模型评估
    model.eval()
    with torch.no_grad():
        # 测试集预测
        test_tensor = torch.FloatTensor(x_test).to(device)
        test_outputs = model(test_tensor)
        test_probs = torch.softmax(test_outputs, dim=1)
        test_predicted = torch.argmax(test_outputs, dim=1).cpu().numpy()
        
        # 训练集预测
        train_tensor = torch.FloatTensor(x_train).to(device)
        train_outputs = model(train_tensor)
        train_probs = torch.softmax(train_outputs, dim=1)
        train_predicted = torch.argmax(train_outputs, dim=1).cpu().numpy()
    
    # 计算并打印总的训练时间
    training_time = end_time - start_time
    print(f"总训练时长: {training_time:.2f} seconds")

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

if __name__ == "__main__":
    main()
#
#