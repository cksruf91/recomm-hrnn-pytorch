import argparse
import os
import pickle

import pandas as pd
import torch

from common.data_iterator import data_iterator
from config import CONFIG
from model.hrnn4recom import HRNN


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', required=True, help='확인할 유저 번호', type=int)
    parser.add_argument('-d', '--dataset', default='10M', choices=['10M', '1M', 'BRUNCH'], help='데이터셋', type=str)
    parser.add_argument('-w', '--weight', required=True, help='모델 가중치값', type=str)
    parser.add_argument('-k', '--eval_k', default=25, help='', type=int)
    return parser.parse_args()


def get_user_test_data(test_data, user_id):
    with open(test_data, 'r') as file:
        for line in file:
            line = [int(l) for l in line.split('\t')]
            if line[0] == user_id:
                return line[1], line[2:]
    raise ValueError(f'User {user_id} is not exist')


if __name__ == '__main__':
    argument = args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_dir = os.path.join(CONFIG.DATA, argument.dataset)
    
    # loading data
    context_dataset = pickle.load(open(os.path.join(data_dir, f'valid.pkl'), 'rb'))
    item_meta = pd.read_csv(os.path.join(data_dir, f'item_meta.tsv'), sep='\t', low_memory=False)
    positive_item, negative_item = get_user_test_data(os.path.join(data_dir, 'negative_test.dat'), argument.user)
    user_contexts = context_dataset[argument.user]
    item_size = int(item_meta.item_id.max() + 1)
    
    # loading model
    model_params = {
        'hiddenUnits': 100,  # 50
        'k': argument.eval_k, 'dropout': 0.2, 'item_size': item_size
    }
    hrnn = HRNN(
        model_params['hiddenUnits'], model_params['item_size'], device=device, k=model_params['k'],
        dropout=model_params['dropout']
    )
    weight = os.path.join('result', argument.dataset, argument.weight)
    hrnn.load(weight)

    # prediction
    recommend_items = hrnn.get_recommendation(
        data_iterator(user_contexts, device=device), argument.eval_k
    )

    # print result
    display_cols = ['item_id', 'user_id', 'Title', 'Genres']
    if argument.dataset == 'BRUNCH':
        display_cols = ['item_id', 'user_id', 'title', 'keyword_list', 'display_url']
    
    history = pd.DataFrame(
        {'item_id': [int(c['outputItem']) for c in user_contexts],
         'rating': [int(c['rating']) for c in user_contexts],
         'timestamp': [int(c['timestamp']) for c in user_contexts]}
    )
    history = history.merge(item_meta, on='item_id', how='left')
    print(history[ display_cols + ['rating', 'timestamp'] ])

    test_dataset = pd.DataFrame({'item_id': [positive_item] + negative_item})
    test_dataset = test_dataset.merge(item_meta, on='item_id', how='left', validate='1:m')
    print(test_dataset.shape)
    print(test_dataset[display_cols].head())

    recommend = pd.DataFrame({'item_id': recommend_items[0]})
    recommend = recommend.merge(item_meta, on='item_id', how='left', validate='1:m')
    print(recommend.shape)
    print(recommend[display_cols])
