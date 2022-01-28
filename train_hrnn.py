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
    parser.add_argument('-d', '--dataset', default='10M', choices=['10M', '1M', 'BRUNCH'], help='데이터셋', type=str)
    parser.add_argument('-k', '--eval_k', default=25, help='', type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.1, help='learning rate', type=float)
    parser.add_argument('-bs', '--batch_size', default=50, help='batch size', type=int)
    parser.add_argument('-ns', '--negative_sample', default=0, help='negative sample size', type=int)
    return parser.parse_args()


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    argument = args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_params = {
        'hiddenUnits': 100,  # 500
        'negativeSampleSize': argument.negative_sample,
        'learningRate': argument.learning_rate,
        'loss': 'TOP1Loss',
        'optimizer': 'Adagrad',
        'k': argument.eval_k, 'dropout': 0.2,
        'batchSize': argument.batch_size
    }

    print('loading data...')
    data_dir = os.path.join(CONFIG.DATA, argument.dataset)
    train_dataset = pickle.load(open(os.path.join(data_dir, f'train.pkl'), 'rb'))
    valid_dataset = pickle.load(open(os.path.join(data_dir, f'valid.pkl'), 'rb'))
    item_meta = pd.read_csv(os.path.join(data_dir, f'item_meta.csv'))
    item_size = int(item_meta.item_id.max()) + 1
    model_params['itemSize'] = item_size

    n_sampler = NegativeSampler(item_meta, sample_size=model_params['negativeSampleSize'])

    train_dataloader = DataLoader(
        train_dataset, batch_size=model_params['batchSize'], device=device, 
        negative_sampler=n_sampler if model_params['negativeSampleSize'] > 0 else None
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=model_params['batchSize'], device=device
    )
    total = len(train_dataloader)

    hrnn = HRNN(
        model_params['hiddenUnits'], item_size, device=device, k=model_params['k'], 
        dropout=model_params['dropout']
    )
    loss_func = TOP1Loss()
    optimizer = Adagrad(hrnn.parameters(), lr=model_params['learningRate'], eps=0.00001, weight_decay=0.0)
    # optimizer = Adadelta(hrnn.parameters(), lr=model_params['learningRate'], eps=1e-06, weight_decay=0.0)
    
    print(f"device : {device}")

    metrics = [nDCG(), RecallAtK()]
    callbacks = [
        ModelCheckPoint(
            os.path.join('.', 'result', argument.dataset, 
                f'hrnn_v{argument.model_version}' + '_e{epoch:02d}-loss{val_loss:1.4f}.zip')),
        MlflowLogger(f'{argument.dataset}', model_params, run_name=f'hrnn-v{argument.model_version}')
    ]

    hrnn.fit(
        30, train_dataloader, valid_dataloader, loss_func=loss_func, optimizer=optimizer,
        metrics=metrics, callback=callbacks, sample=0.3
    )
