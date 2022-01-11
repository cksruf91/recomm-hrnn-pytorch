import os
import pickle

import pandas as pd
import torch
from torch.optim import Adam

from common.dataloader import DataLoader
from model.loss_functions import BPRLoss, TOP1Loss
from config import CONFIG
from model.hrnn4recom import HRNN
from model.metrics import nDCG, RecallAtK


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('loading data...')
    train_dataset = pickle.load(open(os.path.join(CONFIG.DATA, 'train.pkl'), 'rb'))
    valid_dataset = pickle.load(open(os.path.join(CONFIG.DATA, 'valid.pkl'), 'rb'))
    item_meta = pd.read_csv(os.path.join(CONFIG.DATA, 'item_meta.csv'))
    item_size = item_meta.item_id.nunique()

    train_dataloader = DataLoader(train_dataset, batch_size=64, device=device)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, device=device)
    total = len(train_dataloader)

    hrnn = HRNN(128, item_size, device=device)
    loss_func = TOP1Loss()
    optimizer = Adam(list(hrnn.parameters()), lr=0.05, eps=0.00001)
    print(device)
    
    metrics = [nDCG(), RecallAtK()]

    hrnn.fit(
        10, train_dataloader, valid_dataloader, loss_func=loss_func, optimizer=optimizer,
        metrics=metrics, sample=0.1
    )