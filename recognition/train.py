import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from model import FaceMobileNet, ResIRSE
from model.metric import ArcFace, CosFace
from model.loss import FocalLoss
from dataset import load_data
from config import config as conf

# Data Setup
dataloader, class_num = load_data(conf, training=True)
embedding_size = conf.embedding_size
device = conf.device

# Network Setup
if conf.backbone == 'resnet':
    net = ResIRSE(embedding_size, conf.drop_ratio).to(device)
else:
    net = FaceMobileNet(embedding_size).to(device)

if conf.metric == 'arcface':
    metric = ArcFace(embedding_size, class_num).to(device)
else:
    metric = CosFace(embedding_size, class_num).to(device)

net = nn.DataParallel(net)
metric = nn.DataParallel(metric)

# Training Setup
if conf.loss == 'focal_loss':
    criterion = FocalLoss(gamma=2)
else:
    criterion = nn.CrossEntropyLoss()

if conf.optimizer == 'sgd':
    optimizer = optim.SGD([{'params': net.parameters()}, {'params': metric.parameters()}], 
                            lr=conf.lr, weight_decay=conf.weight_decay)
else:
    optimizer = optim.Adam([{'params': net.parameters()}, {'params': metric.parameters()}],
                            lr=conf.lr, weight_decay=conf.weight_decay)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_step, gamma=0.1)

# Checkpoints Setup
checkpoints = conf.checkpoints
os.makedirs(checkpoints, exist_ok=True)

if conf.restore:
    weights_path = osp.join(checkpoints, conf.restore_model)
    net.load_state_dict(torch.load(weights_path, map_location=device))

# Start training
net.train()

for e in range(conf.epoch):
    for data, labels in tqdm(dataloader, desc=f"Epoch {e}/{conf.epoch}",
                             ascii=True, total=len(dataloader)):
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        embeddings = net(data)
        thetas = metric(embeddings, labels)
        loss = criterion(thetas, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {e}/{conf.epoch}, Loss: {loss}")

    backbone_path = osp.join(checkpoints, f"{e}.pth")
    torch.save(net.state_dict(), backbone_path)
    scheduler.step()