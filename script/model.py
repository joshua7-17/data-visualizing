# -*- coding: utf-8 -*-
"""
CCKS2021 中文NLP地址相关性任务 - 模型训练
基于预训练模型微调 (BERT/NEZHA/ MacBERT)
参考冠军方案：对抗训练(FGM) + Multi-sample Dropout + 动态加权 + 阈值搜索

数据格式: {"sentence1": "query地址", "sentence2": "candidate地址", "label": "exact_match/partial_match/not_match"}
"""

import os
import sys
import json
import random
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

warnings.filterwarnings('ignore')


# ===================== 配置 =====================

def parse_args():
    parser = argparse.ArgumentParser(description='CCKS2021 地址相关性训练')
    
    # 数据路径
    parser.add_argument('--train_file', type=str, default='../data/train.json',
                        help='训练数据文件')
    parser.add_argument('--dev_file', type=str, default='../data/dev.json',
                        help='验证数据文件')
    parser.add_argument('--output_dir', type=str, default='../output',
                        help='输出目录')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='hfl/chinese-bert-wwm-ext',
                        help='预训练模型名称或路径')
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help='最大序列长度')
    parser.add_argument('--num_labels', type=int, default=3,
                        help='分类标签数')
    
    # 训练参数
    parser.add_argument('--do_train', action='store_true', default=True,
                        help='是否训练')
    parser.add_argument('--do_eval', action='store_true', default=True,
                        help='是否评估')
    parser.add_argument('--do_predict', action='store_true', default=False,
                        help='是否预测')
    parser.add_argument('--kfold', type=int, default=3,
                        help='K折交叉验证')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='训练batch size')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='评估batch size')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='warmup比例')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    
    # 技巧参数
    parser.add_argument('--use_fgm', action='store_true', default=True,
                        help='是否使用FGM对抗训练')
    parser.add_argument('--fgm_epsilon', type=float, default=0.5,
                        help='FGM扰动大小')
    parser.add_argument('--use_multidropout', action='store_true', default=True,
                        help='是否使用Multi-sample Dropout')
    parser.add_argument('--dropout_num', type=int, default=5,
                        help='Multi-sample Dropout次数')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.3,
                        help='Dropout概率')
    parser.add_argument('--use_swa', action='store_true', default=False,
                        help='是否使用SWA')
    parser.add_argument('--use_rdrop', action='store_true', default=False,
                        help='是否使用R-Drop')
    
    # 手工特征融合
    parser.add_argument('--use_handcraft_features', action='store_true', default=False,
                        help='是否使用手工特征与BERT融合')
    parser.add_argument('--train_feature_file', type=str, default='../output/train_features.csv',
                        help='训练集手工特征CSV路径')
    parser.add_argument('--dev_feature_file', type=str, default='../output/dev_features.csv',
                        help='验证集手工特征CSV路径')
    parser.add_argument('--feature_hidden_size', type=int, default=64,
                        help='手工特征映射维度')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='梯度累积步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='梯度裁剪')
    parser.add_argument('--logging_steps', type=int, default=500,
                        help='日志步数')
    parser.add_argument('--save_steps', type=int, default=1000,
                        help='保存步数')
    parser.add_argument('--early_stop_patience', type=int, default=3,
                        help='早停耐心值')
    
    args = parser.parse_args()
    return args


args = parse_args()


# ===================== 工具函数 =====================

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def label_to_id(label):
    """标签转ID"""
    mapping = {'exact_match': 0, 'partial_match': 1, 'not_match': 2}
    return mapping[label]


def id_to_label(label_id):
    """ID转标签"""
    mapping = {0: 'exact_match', 1: 'partial_match', 2: 'not_match'}
    return mapping[label_id]


# ===================== FGM对抗训练 =====================

class FGM:
    """Fast Gradient Method 对抗训练"""
    def __init__(self, model, emb_name='word_embeddings', epsilon=0.5):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# ===================== Multi-sample Dropout 分类器 =====================

class MultiSampleDropoutClassifier(nn.Module):
    """Multi-sample Dropout 分类器"""
    def __init__(self, hidden_size, num_labels, dropout_prob=0.3, dropout_num=5):
        super().__init__()
        self.dropout_num = dropout_num
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_prob * (i + 1) / dropout_num),
                nn.Linear(hidden_size, num_labels)
            )
            for i in range(dropout_num)
        ])

    def forward(self, hidden_states):
        # hidden_states: [batch_size, hidden_size]
        logits_list = [classifier(hidden_states) for classifier in self.classifiers]
        logits = torch.stack(logits_list, dim=0)  # [dropout_num, batch_size, num_labels]
        logits = logits.mean(dim=0)  # [batch_size, num_labels]
        return logits


# ===================== 模型定义 =====================

class AddressMatchingModel(nn.Module):
    """地址匹配模型，支持Multi-sample Dropout + 手工特征融合"""
    def __init__(self, model_name, num_labels, use_multidropout=False, 
                 dropout_prob=0.3, dropout_num=5,
                 use_handcraft_features=False, handcraft_feature_dim=0,
                 feature_hidden_size=64):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.config.hidden_dropout_prob = dropout_prob
        
        # 加载预训练模型
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
            ignore_mismatched_sizes=True
        )
        
        self.use_multidropout = use_multidropout
        self.use_handcraft_features = use_handcraft_features
        self.hidden_size = self.config.hidden_size
        self.num_labels = num_labels
        
        if use_multidropout and not use_handcraft_features:
            # 纯BERT + Multi-sample Dropout
            self.bert.classifier = MultiSampleDropoutClassifier(
                self.hidden_size, num_labels, dropout_prob, dropout_num
            )
        
        if use_handcraft_features:
            # 手工特征映射层
            self.feature_proj = nn.Sequential(
                nn.Linear(handcraft_feature_dim, feature_hidden_size),
                nn.BatchNorm1d(feature_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            )
            # 融合后的分类器
            fused_hidden = self.hidden_size + feature_hidden_size
            if use_multidropout:
                self.fused_classifier = MultiSampleDropoutClassifier(
                    fused_hidden, num_labels, dropout_prob, dropout_num
                )
            else:
                self.fused_classifier = nn.Sequential(
                    nn.Dropout(dropout_prob),
                    nn.Linear(fused_hidden, num_labels)
                )
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, 
                labels=None, handcraft_features=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels if not self.use_handcraft_features else None,
            output_hidden_states=True,
            return_dict=True
        )
        
        if self.use_handcraft_features and handcraft_features is not None:
            # 取最后一层CLS向量
            cls_vec = outputs.hidden_states[-1][:, 0, :]  # [batch, hidden]
            # 映射手工特征
            feat_vec = self.feature_proj(handcraft_features)  # [batch, feat_hidden]
            # 拼接
            fused = torch.cat([cls_vec, feat_vec], dim=-1)
            # 用融合分类器
            logits = self.fused_classifier(fused)
            
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                return type('Output', (), {'loss': loss, 'logits': logits})()
            return type('Output', (), {'logits': logits, 'loss': None})()
        
        return outputs


# ===================== 数据集 =====================

class AddressDataset(Dataset):
    """地址匹配数据集，支持可选的手工特征"""
    def __init__(self, texts1, texts2, labels=None, tokenizer=None, max_len=128,
                 handcraft_features=None, handcraft_feature_cols=None):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.handcraft_features = handcraft_features
        self.handcraft_feature_cols = handcraft_feature_cols

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, idx):
        text1 = str(self.texts1[idx])
        text2 = str(self.texts2[idx])
        
        encoding = self.tokenizer(
            text1, text2,
            truncation='longest_first',
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.handcraft_features is not None:
            item['handcraft_features'] = torch.tensor(
                self.handcraft_features[idx], dtype=torch.float)
        
        return item


def dynamic_collate_fn(batch):
    """动态padding的collate函数"""
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    
    # 动态padding到batch内最大长度
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
    }
    
    if 'labels' in batch[0]:
        result['labels'] = torch.stack([item['labels'] for item in batch])
    
    if 'handcraft_features' in batch[0]:
        result['handcraft_features'] = torch.stack([item['handcraft_features'] for item in batch])
    
    return result


# ===================== 数据加载 =====================

def load_jsonl_data(filepath):
    """加载JSONL格式数据"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return pd.DataFrame(data)


def prepare_data(df, tokenizer, max_len=128, handcraft_feature_file=None):
    """准备数据集，可选加载手工特征"""
    texts1 = df['sentence1'].tolist()
    texts2 = df['sentence2'].tolist()
    
    handcraft_data = None
    handcraft_cols = None
    
    if handcraft_feature_file and os.path.exists(handcraft_feature_file):
        feat_df = pd.read_csv(handcraft_feature_file)
        # 只取数值特征列（排除文本和标签列）
        exclude_cols = {'sentence1', 'sentence2', 'label', 'label_id'}
        handcraft_cols = [c for c in feat_df.columns if c not in exclude_cols]
        handcraft_data = feat_df[handcraft_cols].fillna(0).values
        print(f"  加载手工特征: {len(handcraft_cols)}维 ({handcraft_feature_file})")
    
    if 'label' in df.columns:
        labels = df['label'].map(label_to_id).tolist()
        dataset = AddressDataset(texts1, texts2, labels, tokenizer, max_len,
                                handcraft_data, handcraft_cols)
        return dataset, labels, handcraft_cols
    else:
        dataset = AddressDataset(texts1, texts2, None, tokenizer, max_len,
                                handcraft_data, handcraft_cols)
        return dataset, None, handcraft_cols


# ===================== 训练器 =====================

class Trainer:
    """模型训练器"""
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.best_f1 = 0.0
        self.patience_counter = 0
        
    def train(self, train_loader, dev_loader=None):
        """训练模型"""
        args = self.args
        
        # 优化器
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() 
                       if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() 
                       if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        
        # 学习率调度
        total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        # FGM
        fgm = FGM(self.model, epsilon=args.fgm_epsilon) if args.use_fgm else None
        
        # 训练循环
        global_step = 0
        for epoch in range(args.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            print('='*50)
            
            self.model.train()
            total_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f'Training')
            for step, batch in enumerate(progress_bar):
                # 数据移到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                model_kwargs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'labels': labels,
                }
                
                # 如果有手工特征
                if 'handcraft_features' in batch:
                    model_kwargs['handcraft_features'] = batch['handcraft_features'].to(self.device)
                
                # 前向传播
                outputs = self.model(**model_kwargs)
                loss = outputs.loss
                
                # R-Drop (如果启用)
                if args.use_rdrop:
                    outputs2 = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
                    # KL散度损失
                    p1 = F.log_softmax(outputs.logits, dim=-1)
                    p2 = F.softmax(outputs2.logits, dim=-1)
                    kl_loss = F.kl_div(p1, p2, reduction='batchmean')
                    loss = loss + 0.3 * kl_loss
                
                # 梯度累积
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                # FGM对抗训练
                if fgm is not None:
                    fgm.attack()
                    adv_kwargs = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'token_type_ids': token_type_ids,
                        'labels': labels,
                    }
                    if 'handcraft_features' in batch:
                        adv_kwargs['handcraft_features'] = batch['handcraft_features'].to(self.device)
                    outputs_adv = self.model(**adv_kwargs)
                    loss_adv = outputs_adv.loss / args.gradient_accumulation_steps
                    loss_adv.backward()
                    fgm.restore()
                
                total_loss += loss.item() * args.gradient_accumulation_steps
                
                # 梯度裁剪
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{total_loss / (step + 1):.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                # 日志
                if global_step % args.logging_steps == 0 and global_step > 0:
                    avg_loss = total_loss / (step + 1)
                    print(f"\nStep {global_step}: loss = {avg_loss:.4f}")
            
            # 每个epoch结束后评估
            if dev_loader is not None:
                metrics = self.evaluate(dev_loader)
                f1 = metrics['f1_macro']
                print(f"\nEpoch {epoch + 1} 验证集: F1(macro)={f1:.4f}, Acc={metrics['accuracy']:.4f}")
                print(f"分类报告:\n{metrics['report']}")
                
                # 保存最佳模型
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.patience_counter = 0
                    self._save_model()
                    print(f"✓ 保存最佳模型, F1(macro)={f1:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"✗ F1未提升, 连续{self.patience_counter}次")
                    
                # 早停
                if self.patience_counter >= args.early_stop_patience:
                    print(f"早停触发！")
                    break
    
    def evaluate(self, dev_loader):
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                model_kwargs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                }
                if 'handcraft_features' in batch:
                    model_kwargs['handcraft_features'] = batch['handcraft_features'].to(self.device)
                
                outputs = self.model(**model_kwargs)
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # 按类别计算F1
        report = classification_report(
            all_labels, all_preds,
            target_names=['完全匹配', '部分匹配', '不匹配'],
            digits=4
        )
        
        return {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'accuracy': accuracy,
            'report': report,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def predict(self, test_loader):
        """预测"""
        self.model.eval()
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Predicting'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                
                model_kwargs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                }
                if 'handcraft_features' in batch:
                    model_kwargs['handcraft_features'] = batch['handcraft_features'].to(self.device)
                
                outputs = self.model(**model_kwargs)
                
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        return np.array(all_probs), np.array(all_preds)
    
    def _save_model(self):
        """保存模型"""
        save_dir = os.path.join(args.output_dir, 'best_model')
        os.makedirs(save_dir, exist_ok=True)
        # 保存底层bert模型
        self.model.bert.save_pretrained(save_dir)
        # 保存整个模型状态
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'model_state.pt'))
        print(f"模型已保存到 {save_dir}")


# ===================== 阈值优化 =====================

class ThresholdOptimizer:
    """预测概率阈值优化器，用于提升Macro F1"""
    def __init__(self):
        self.best_coef = None
        
    def fit(self, probs, labels):
        """搜索最优缩放系数"""
        from functools import partial
        import scipy.optimize as opt
        
        def objective(coef, probs, labels):
            scaled = probs * coef
            preds = np.argmax(scaled, axis=1)
            return -f1_score(labels, preds, average='macro')
        
        # 使用Nelder-Mead搜索最优系数
        initial_coef = np.ones(probs.shape[1])
        result = opt.minimize(
            objective, initial_coef,
            args=(probs, labels),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        self.best_coef = result.x
        print(f"最优缩放系数: {self.best_coef}")
        return self.best_coef
    
    def predict(self, probs):
        """使用最优系数预测"""
        if self.best_coef is None:
            return np.argmax(probs, axis=1)
        scaled = probs * self.best_coef
        return np.argmax(scaled, axis=1)


# ===================== K折交叉验证 =====================

def run_kfold(args):
    """运行K折交叉验证"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    train_df = load_jsonl_data(args.train_file)
    dev_df = load_jsonl_data(args.dev_file)
    
    # 合并训练和验证集用于交叉验证
    full_df = pd.concat([train_df, dev_df], ignore_index=True)
    
    # 准备标签
    labels = full_df['label'].map(label_to_id).values
    
    print(f"总数据量: {len(full_df)}")
    print(f"标签分布: {Counter(labels)}")
    
    # 加载tokenizer
    print(f"加载tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # K折交叉验证
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{args.kfold}")
        print(f"训练集: {len(train_idx)}, 验证集: {len(val_idx)}")
        print('='*60)
        
        # 划分数据
        train_fold = full_df.iloc[train_idx].reset_index(drop=True)
        val_fold = full_df.iloc[val_idx].reset_index(drop=True)
        
        # 准备数据集
        train_dataset, _, _ = prepare_data(train_fold, tokenizer, args.max_seq_len,
                                           args.train_feature_file if args.use_handcraft_features else None)
        val_dataset, _, handcraft_cols = prepare_data(val_fold, tokenizer, args.max_seq_len,
                                          args.dev_feature_file if args.use_handcraft_features else None)
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.train_batch_size, 
            shuffle=True, collate_fn=dynamic_collate_fn,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.eval_batch_size,
            shuffle=False, collate_fn=dynamic_collate_fn,
            num_workers=0
        )
        
        # 初始化模型
        set_seed(args.seed + fold)
        hc_dim = len(handcraft_cols) if args.use_handcraft_features and handcraft_cols else 0
        model = AddressMatchingModel(
            model_name=args.model_name,
            num_labels=args.num_labels,
            use_multidropout=args.use_multidropout,
            dropout_prob=args.hidden_dropout_prob,
            dropout_num=args.dropout_num,
            use_handcraft_features=args.use_handcraft_features,
            handcraft_feature_dim=hc_dim,
            feature_hidden_size=args.feature_hidden_size
        )
        model.to(device)
        
        # 训练
        trainer = Trainer(model, device, args)
        trainer.train(train_loader, val_loader)
        
        # 记录最佳指标
        fold_metrics.append(trainer.best_f1)
        print(f"\nFold {fold + 1} 最佳 F1(macro): {trainer.best_f1:.4f}")
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"K折交叉验证结果:")
    for i, f1 in enumerate(fold_metrics):
        print(f"  Fold {i + 1}: F1(macro) = {f1:.4f}")
    print(f"  平均: {np.mean(fold_metrics):.4f} ± {np.std(fold_metrics):.4f}")
    print('='*60)


# ===================== 主训练流程 =====================

def main():
    """主训练流程"""
    print("="*60)
    print("CCKS2021 中文地址相关性 - 模型训练")
    print("="*60)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"参数: {args}")
    
    if args.do_train and args.kfold > 1:
        # K折交叉验证训练
        run_kfold(args)
        return
    
    # 单次训练
    # 加载数据
    print("\n加载数据...")
    train_df = load_jsonl_data(args.train_file)
    dev_df = load_jsonl_data(args.dev_file)
    
    print(f"训练集: {len(train_df)}条")
    print(f"验证集: {len(dev_df)}条")
    
    # 加载tokenizer
    print(f"\n加载tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 准备数据集
    train_dataset, _, handcraft_cols = prepare_data(train_df, tokenizer, args.max_seq_len,
                                       args.train_feature_file if args.use_handcraft_features else None)
    dev_dataset, _, _ = prepare_data(dev_df, tokenizer, args.max_seq_len,
                                     args.dev_feature_file if args.use_handcraft_features else None)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size,
        shuffle=True, collate_fn=dynamic_collate_fn,
        num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.eval_batch_size,
        shuffle=False, collate_fn=dynamic_collate_fn,
        num_workers=0
    )
    
    # 初始化模型
    print(f"\n加载预训练模型: {args.model_name}")
    hc_dim = len(handcraft_cols) if args.use_handcraft_features and handcraft_cols else 0
    if args.use_handcraft_features:
        print(f"手工特征维度: {hc_dim}, 将映射到{args.feature_hidden_size}维后与BERT拼接")
    model = AddressMatchingModel(
        model_name=args.model_name,
        num_labels=args.num_labels,
        use_multidropout=args.use_multidropout,
        dropout_prob=args.hidden_dropout_prob,
        dropout_num=args.dropout_num,
        use_handcraft_features=args.use_handcraft_features,
        handcraft_feature_dim=hc_dim,
        feature_hidden_size=args.feature_hidden_size
    )
    model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 训练
    print("\n开始训练...")
    trainer = Trainer(model, device, args)
    trainer.train(train_loader, dev_loader)
    
    print(f"\n训练完成！最佳 F1(macro): {trainer.best_f1:.4f}")
    
    # 评估
    if args.do_eval:
        print("\n最终评估:")
        # 加载最佳模型
        best_model_path = os.path.join(args.output_dir, 'best_model')
        if os.path.exists(best_model_path):
            print(f"加载最佳模型: {best_model_path}")
            best_model = AddressMatchingModel(
                model_name=args.model_name,
                num_labels=args.num_labels,
                use_multidropout=args.use_multidropout,
                dropout_prob=args.hidden_dropout_prob,
                dropout_num=args.dropout_num,
                use_handcraft_features=args.use_handcraft_features,
                handcraft_feature_dim=hc_dim,
                feature_hidden_size=args.feature_hidden_size
            )
            state_dict = torch.load(os.path.join(best_model_path, 'model_state.pt'), map_location=device)
            best_model.load_state_dict(state_dict)
            best_model.to(device)
            trainer.model = best_model
        
        metrics = trainer.evaluate(dev_loader)
        print(f"\n最终验证集结果:")
        print(f"  F1 (macro):   {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"\n分类报告:\n{metrics['report']}")
        
        # 阈值优化
        print("\n阈值优化...")
        opt = ThresholdOptimizer()
        
        # 收集验证集概率
        all_probs = []
        all_labels = []
        best_model = trainer.model
        best_model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc='收集概率'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)
                
                eval_kwargs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                }
                if 'handcraft_features' in batch:
                    eval_kwargs['handcraft_features'] = batch['handcraft_features'].to(device)
                
                outputs = best_model(**eval_kwargs)
                probs = F.softmax(outputs.logits, dim=-1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 优化阈值
        opt.fit(all_probs, all_labels)
        opt_preds = opt.predict(all_probs)
        opt_f1 = f1_score(all_labels, opt_preds, average='macro')
        print(f"阈值优化后 F1(macro): {opt_f1:.4f}")
        
        # 保存结果
        results = {
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'accuracy': metrics['accuracy'],
            'opt_f1_macro': float(opt_f1),
            'best_coef': opt.best_coef.tolist() if opt.best_coef is not None else None
        }
        
        with open(os.path.join(args.output_dir, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到 {os.path.join(args.output_dir, 'results.json')}")


if __name__ == '__main__':
    main()
