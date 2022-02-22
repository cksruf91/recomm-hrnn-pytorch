import argparse
import os
import pickle
from typing import Callable

import pandas as pd
import torch
from torch.optim import Adagrad, Adadelta, Adam

from common.data_iterator import DataLoader, NegativeSampler, TestIterator
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
    parser.add_argument('-o', '--dropout', default=0.2, help='dropout', type=float)
    parser.add_argument('-bs', '--batch_size', default=50, help='batch size', type=int)
    parser.add_argument('-ns', '--negative_sample', default=0, help='negative sample size', type=int)
    return parser.parse_args()


def get_optimizer(model, name: str, lr: float) -> Callable:
    """ optimizer를 return 하는 함수
    Args:
        name: optimizer name
        lr: learning rate

    Returns: pytorch optimizer function
    """
    
    functions = {
        'Adagrad': Adagrad(model.parameters(), lr=lr, eps=0.00001, weight_decay=0.0),
        'Adadelta': Adadelta(model.parameters(), lr=lr, eps=1e-06, weight_decay=0.0),
        'Adam': Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.0, amsgrad=False)
    }
    try:
        return functions[name]
    except KeyError:
        raise ValueError(f'optimizer [{name}] not exist, available optimizer {list(functions.keys())}')


if __name__ == '__main__':
    argument = args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_params = {
        'hiddenUnits': 100,  # 500
        'negativeSampleSize': argument.negative_sample,
        'learningRate': argument.learning_rate,
        'loss': 'TOP1Loss',
        'optimizer': 'Adagrad', #Adagrad
        'k': argument.eval_k, 'dropout': argument.dropout,
        'batchSize': argument.batch_size
    }

    print('loading data...')
    data_dir = os.path.join(CONFIG.DATA, argument.dataset)
    train_dataset = pickle.load(open(os.path.join(data_dir, f'train.pkl'), 'rb'))
    context_dataset = pickle.load(open(os.path.join(data_dir, f'valid.pkl'), 'rb'))
    item_meta = pd.read_csv(os.path.join(data_dir, f'item_meta.tsv'), sep='\t', low_memory=False)
    item_size = int(item_meta.item_id.max() + 1)
    model_params['itemSize'] = item_size

    n_sampler = NegativeSampler(item_meta, sample_size=model_params['negativeSampleSize'])

    train_dataloader = DataLoader(
        train_dataset, batch_size=model_params['batchSize'], device=device,
        negative_sampler=n_sampler if model_params['negativeSampleSize'] > 0 else None
    )
    test_iterator = TestIterator(
        os.path.join(data_dir, 'negative_test.dat'), context_dataset, device=device
    )

    hrnn = HRNN(
        model_params['hiddenUnits'], item_size, device=device, k=model_params['k'],
        dropout=model_params['dropout']
    )
    loss_func = TOP1Loss()
    optimizer = get_optimizer(hrnn, model_params['optimizer'], lr=model_params['learningRate'])

    

    metrics = [nDCG(), RecallAtK()]
    model_name = f'hrnn_v{argument.model_version}'
    callbacks = [
        ModelCheckPoint(
            os.path.join(
                '.', 'result', argument.dataset,
                model_name + '-e{epoch:02d}-loss{val_loss:1.4f}-nDCG{val_nDCG:1.3f}.zip'),
            monitor='val_nDCG', mode='max'
        )
        ,
        MlflowLogger(f'{argument.dataset}', model_params, run_name=model_name, log_model=False)
    ]
    print(hrnn)
    print(f"device : {device}")
    hrnn.fit(
        50, train_dataloader, test_iterator, loss_func=loss_func, optimizer=optimizer,
        metrics=metrics, callback=callbacks, sample=1.
    )
