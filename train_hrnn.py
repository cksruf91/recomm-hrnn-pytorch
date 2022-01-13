import os
import pickle

import pandas as pd
import torch
from torch.optim import Adagrad

from common.dataloader import DataLoader, NegativeSampler
from config import CONFIG
from model.callbacks import ModelCheckPoint
from model.hrnn4recom import HRNN
from model.loss_functions import TOP1Loss
from model.metrics import nDCG, RecallAtK

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('loading data...')
    train_dataset = pickle.load(open(os.path.join(CONFIG.DATA, 'train.pkl'), 'rb'))
    valid_dataset = pickle.load(open(os.path.join(CONFIG.DATA, 'valid.pkl'), 'rb'))
    item_meta = pd.read_csv(os.path.join(CONFIG.DATA, 'item_meta.csv'))
    item_size = item_meta.item_id.nunique()

    n_sampler = NegativeSampler(item_meta, sample_size=30)

    train_dataloader = DataLoader(train_dataset, batch_size=64, device=device, negative_sampler=None)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, device=device)
    total = len(train_dataloader)

    hrnn = HRNN(100, item_size, device=device, k=25, dropout=0.2)
    loss_func = TOP1Loss()
    optimizer = Adagrad(hrnn.parameters(), lr=0.1, eps=0.00001, weight_decay=0.0)
    print(f"device : {device}")

    metrics = [nDCG(), RecallAtK()]
    callbacks = [
        ModelCheckPoint(os.path.join('.', 'result', 'hrnn_v1_e{epoch:02d}-loss{val_loss:1.4f}.zip'))
    ]

    hrnn.fit(
        10, train_dataloader, valid_dataloader, loss_func=loss_func, optimizer=optimizer,
        metrics=metrics, callback=callbacks, sample=0.1
    )
