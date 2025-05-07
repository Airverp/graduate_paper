import torch
import torch.nn as nn
import torch.optim as optim
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

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2, dropout_rate=0.3):
        super(RNNModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN层
        self.rnn = nn.LSTM(
            input_size=input_size,  
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True  # 使用双向LSTM
        )
        
        # 批标准化层
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)  # *2是因为双向LSTM
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_size),        
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_size // 2),
            
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 重塑输入为 (batch_size, sequence_length, input_size)
        x = x.view(batch_size, -1, self.input_size) 
        
        # 通过RNN层
        rnn_out, _ = self.rnn(x)
        
        # 取最后一个时间步的输出
        last_output = rnn_out[:, -1, :]
        
        # 批标准化
        normed_output = self.batch_norm(last_output)
        
        # 通过全连接层
        out = self.fc_layers(normed_output)
        return out

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
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val
        
        scheduler.step(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return history

def main():
    # 设置随机种子
    set_seed(42)
    
    # 加载SMOTE后的数据
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
    input_size = x_train.shape[1]  # 确保input_size为13
    model = RNNModel(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        num_classes=2,
        dropout_rate=0.3
    )
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, verbose=True
    )
    
    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        num_epochs=1000, device=device
    )
    
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

#测试集0.9