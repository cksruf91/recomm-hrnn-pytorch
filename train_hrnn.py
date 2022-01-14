import argparse
import os
import pickle

import pandas as pd
import torch
from torch.optim import Adagrad, Adam, Adadelta

from common.dataloader import DataLoader, NegativeSampler
from config import CONFIG
from model.callbacks import ModelCheckPoint, MlflowLogger
from model.hrnn4recom import HRNN
from model.loss_functions import TOP1Loss
from model.metrics import nDCG, RecallAtK


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--model_version', required=True, help='모델 버전', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    argument = args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_params = {
        'negativeSampleSize': 0,
        'learningRate': 1.,
        'loss': 'TOP1Loss',
        'optimizer': 'Adadelta',
        'hiddenSize': 100,
        'k': 10, 'dropout': 0.2,
        'batchSize': 32
    }

    print('loading data...')
    train_dataset = pickle.load(open(os.path.join(CONFIG.DATA, 'train.pkl'), 'rb'))
    valid_dataset = pickle.load(open(os.path.join(CONFIG.DATA, 'valid.pkl'), 'rb'))
    item_meta = pd.read_csv(os.path.join(CONFIG.DATA, 'item_meta.csv'))
    item_size = item_meta.item_id.nunique()
    model_params['itemSize'] = item_size

    n_sampler = NegativeSampler(item_meta, sample_size=model_params['negativeSampleSize'])

    train_dataloader = DataLoader(
        train_dataset, batch_size=model_params['batchSize'], device=device, negative_sampler=None
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=model_params['batchSize'], device=device
    )
    total = len(train_dataloader)

    hrnn = HRNN(100, item_size, device=device, k=model_params['k'], dropout=model_params['dropout'])
    loss_func = TOP1Loss()
    # optimizer = Adagrad(hrnn.parameters(), lr=model_params['learningRate'], eps=0.00001, weight_decay=0.0)
    optimizer = Adadelta(hrnn.parameters(), lr=model_params['learningRate'], eps=1e-06, weight_decay=0.0)
    
    print(f"device : {device}")

    metrics = [nDCG(), RecallAtK()]
    callbacks = [
        ModelCheckPoint(
            os.path.join('.', 'result', f'hrnn_v{argument.model_version}' + '_e{epoch:02d}-loss{val_loss:1.4f}.zip')),
        MlflowLogger('Movielens10M', model_params, run_name=f'hrnn-v{argument.model_version}')
    ]

    hrnn.fit(
        10, train_dataloader, valid_dataloader, loss_func=loss_func, optimizer=optimizer,
        metrics=metrics, callback=callbacks, sample=1.
    )
