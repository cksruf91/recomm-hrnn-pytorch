import argparse
import os
import pickle
from itertools import accumulate
from typing import Dict, Tuple

import pandas as pd

from common.utils import progressbar, DefaultDict
from common.data_preprocess import loading_movielens
from config import CONFIG


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='10M', choices=['10M', '1M'], help='데이터셋', type=str)
    return parser.parse_args()


def split_test_by_session(df: pd.DataFrame, n_context_session: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    last_session = df.groupby('user_id')['session_id'].max().reset_index()
    last_session['test'] = 1

    df = df.merge(
        last_session, on=['user_id', 'session_id'], how='left', validate='m:1'
    )
    df['test'].fillna(0, inplace=True)
    train_data = df[df['test'] == 0].drop('test', axis=1)
    test_data = df[df['test'] == 1].drop('test', axis=1)

    test_data = test_data[test_data["item_id"].isin(test_data["item_id"].unique())]

    # 테스트 데이터에 user representation 생성을 위한 context 데이터 추가
    context_session = train_data[
        train_data.groupby("user_id")["session_id"].transform('max') - n_context_session <= train_data['session_id']
        ].copy()

    train_data['context'] = False
    test_data['context'] = False
    context_session['context'] = True
    test_data = pd.concat([test_data, context_session], axis=0).sort_values(['user_id', 'session_id'])
    assert train_data['context'].sum() == 0

    # 테스트데이터 각 유저 첫 라인에 user_mask = 1 처리
    indexes = [min(group.index) for uid, group in test_data.groupby('user_id')]
    test_data.loc[indexes, 'user_mask'] = 1

    return train_data, test_data


def format_dataset(df: pd.DataFrame) -> Dict:
    """ formatting dataFrame to dictionary for DataLoader class

    Args:
        df(DataFrame): pandas DataFrame which is contain user to item interactions
        ex) 	user_id	item_id	target_id	session_id	Timestamp	MovieID	user_mask	session_mask
                0	7	1184	5196	0	1049764366	1210	1.0	1.0
                1	7	5196	1242	0	1049764395	5292	0.0	0.0 ...

    Returns(dict): dictionary format data
        예시) {user_id1 : [row data, row data ....], user_id2 : [row data, row data ....], .... },
            row data = {'inputItem': 7, 'outputItem': 5196, 'userMask': 1.0, 'sessionMask': 0.0, 'conText': True}

    """

    # interactions = namedtuple('Interactions', ['inputItem', 'outputItem', 'userMask', 'sessionMask'])

    def parsing_row(row):
        return {
            'inputItem': row['item_id'], 'outputItem': row['target_id'], 'userMask': row['user_mask'],
            'sessionMask': row['session_mask'], 'conText': row['context']
        }

    data = {}
    total_user = df.user_id.nunique()

    for i, (user_id, df) in enumerate(df.groupby('user_id')):
        progressbar(total_user, i + 1, prefix='parsing data ')
        data[user_id] = [parsing_row(row) for index, row in df.iterrows()]
    
    print(' Done')
    return data


if __name__ == '__main__':
    argument = args()
    
    interactions, item_meta = loading_movielens(argument.dataset)
    print(interactions.head())

    train, test = split_test_by_session(interactions, n_context_session=3)
    train, valid = split_test_by_session(train, n_context_session=3)

    # get each item's popularity for negative sampling
    train['item_count'] = 1
    item_counts = train.groupby('item_id')['item_count'].sum().reset_index()
    item_counts['cumulate_count'] = [c for c in accumulate(item_counts.item_count)]

    item_meta = item_meta.merge(
        item_counts, on='item_id', how='left', validate='1:1'
    )
    item_meta[['item_count', 'cumulate_count']] = item_meta[['item_count', 'cumulate_count']].fillna(0).astype(int)

    train_dataset = format_dataset(train)
    valid_dataset = format_dataset(valid)
    test_dataset = format_dataset(test)

    pickle.dump(train_dataset, open(os.path.join(CONFIG.DATA, f'train_{argument.dataset}.pkl'), 'wb'))
    pickle.dump(valid_dataset, open(os.path.join(CONFIG.DATA, f'valid_{argument.dataset}.pkl'), 'wb'))
    pickle.dump(test_dataset, open(os.path.join(CONFIG.DATA, f'test_{argument.dataset}.pkl'), 'wb'))

    item_meta.to_csv(os.path.join(CONFIG.DATA, f'item_meta_{argument.dataset}.csv'), index=False)
