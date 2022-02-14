import argparse
import os
import pickle
from itertools import accumulate
from typing import Dict, Tuple

import pandas as pd
from pandas import DataFrame

from common.loading_functions import loading_data
from common.utils import progressbar
from config import CONFIG


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='10M', choices=['10M', '1M', 'BRUNCH'], help='데이터셋', type=str)
    return parser.parse_args()


def create_session_id(df: pd.DataFrame, time_delta: int):
    df.sort_values(['user_id', 'Timestamp'], inplace=True)
    df['timedelta'] = df.groupby('user_id')['Timestamp'].diff(1)
    df['timedelta'].fillna(0, inplace=True)
    df['session_id'] = 0
    df.loc[df['timedelta'] > time_delta, 'session_id'] = 1
    df['session_id'] = df.groupby('user_id')['session_id'].cumsum()
    return df


def drop_sparse_item(df: pd.DataFrame, count: int, item_id: str):
    return df[df.groupby(item_id)["user_id"].transform('count') >= count]


def drop_sparse_session(df: pd.DataFrame, count: int, item_id: str):
    return df[
        df.groupby(["user_id", "session_id"])[item_id].transform('count') >= count
        ]


def drop_sparse_user(df, min_cnt, max_cnt):
    session_count = df.groupby("user_id")["session_id"].transform('nunique')
    return df[session_count.between(min_cnt, max_cnt)]


def get_last_n_session(df: pd.DataFrame, n: int):
    return df[
        df["session_id"] >= df.groupby("user_id")["session_id"].transform(lambda x: max(max(x) - n, 0))
        ]


def split_test_by_session(df: pd.DataFrame, n_context_session: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    last_session = df.groupby('user_id')['session_id'].max().reset_index()
    last_session['test'] = 1

    df = df.merge(
        last_session, on=['user_id', 'session_id'], how='left', validate='m:1'
    )
    df['test'].fillna(0, inplace=True)
    train = df[df['test'] == 0].drop('test', axis=1)
    test = df[df['test'] == 1].drop('test', axis=1)

    test = test[test["item_id"].isin(test["item_id"].unique())]

    # 테스트 데이터에 user representation 생성을 위한 context 데이터 추가
    context_session = train[
        train.groupby("user_id")["session_id"].transform('max') - n_context_session <= train['session_id']
    ].copy()

    train['context'] = False
    test['context'] = False
    context_session['context'] = True
    test = pd.concat([test, context_session], axis=0).sort_values(['user_id', 'session_id'])
    assert train['context'].sum() == 0

    # 테스트데이터 각 유저 첫 라인에 user_mask = 1 처리
    indexes = [min(group.index) for uid, group in test.groupby('user_id')]
    test_data.loc[indexes, 'user_mask'] = 1

    return train, test


def format_dataset(df: pd.DataFrame) -> Dict:
    """ formatting dataFrame to dictionary for DataLoader class

    Args:
        df(DataFrame): pandas DataFrame which is contain user to item interactions
        ex) 	user_id	item_id	target_id	session_id	Timestamp	MovieID	user_mask	session_mask
                0	7	1184	5196	0	1049764366	1210	1.0	1.0
                1	7	5196	1242	0	1049764395	5292	0.0	0.0 ...

    Returns(dict): dictionary format data
        예시) {user_id1 : [row data, row data ....], user_id2 : [row data, row data ....], .... },
            row data = {'inputItem': 7, 'outputItem': 5196, 'userMask': 1.0, 'sessionMask': 0.0}

    """

    # interactions = namedtuple('Interactions', ['inputItem', 'outputItem', 'userMask', 'sessionMask'])

    def parsing_row(row):
        return {
            'inputItem': row['item_id'], 'outputItem': row['target_id'], 'userMask': row['user_mask'],
            'sessionMask': row['session_mask'], 'rating': row['Rating'], 'timestamp': row['Timestamp']
        }

    data = {}
    total_user = df.user_id.nunique()

    for i, (user_id, df) in enumerate(df.groupby('user_id')):
        progressbar(total_user, i + 1, prefix='parsing data ')
        data[user_id] = [parsing_row(row) for index, row in df.iterrows()]

    return data


def movielens_preprocess(train: DataFrame, test: list, items: DataFrame, users: DataFrame) -> tuple[
    DataFrame, list, DataFrame, DataFrame]:
    train = create_session_id(train, time_delta=60 * 60 * 24)
    print(f"train dataset : {len(train)}")
    # train = get_last_n_session(train, 30)
    # print(f"get last n session : {len(train)}")
    # train = drop_sparse_item(train, 10, item_id='item_id')
    # print(f"drop sparse item : {len(train)}")
    # train = drop_sparse_session(train, 3, item_id='item_id')
    # print(f"drop sparse session : {len(train)}")
    # train = drop_sparse_user(train, 3, 99)
    # print(f"drop sparse user : {len(train)}")

    # output value(target id) 생성 | input : 현재 item id, output 다음 item id
    target_id = train.groupby(['user_id', 'session_id'])['item_id'].apply(
        lambda col: pd.concat([col[1:], pd.Series([-1])])
    )
    train['target_id'] = target_id.tolist()
    train = train[train.target_id != -1].copy()

    # user mask (유저 아이디 변경 지점)
    train.loc[train.user_id.diff(1) != 0, 'user_mask'] = 1
    train['user_mask'].fillna(0, inplace=True)
    # session mask (세션 아이디 변경 지점)
    train.loc[train.session_id.diff(1) != 0, 'session_mask'] = 1
    train['session_mask'].fillna(0, inplace=True)

    train = train[['item_id', 'user_id', 'Rating', 'target_id', 'session_id', 'Timestamp', 'user_mask', 'session_mask']]
    items = items[["item_id", "MovieID", "Title", "Genres"]]
    return train, test, items, users


def brunch_preprocess(train: DataFrame, test: list, items: DataFrame, users: DataFrame) -> tuple[
    DataFrame, list, DataFrame, DataFrame]:
    train = create_session_id(train, time_delta=60 * 60 * 24)
    print(f"train dataset : {len(train)}")

    # output value(target id) 생성 | input : 현재 item id, output 다음 item id
    target_id = train.groupby(['user_id', 'session_id'])['item_id'].apply(
        lambda col: pd.concat([col[1:], pd.Series([-1])])
    )
    train['target_id'] = target_id.tolist()
    train = train[train.target_id != -1].copy()

    print(f'test set size : {len(test)}')
    user_list = train['user_id'].unique()
    test = [t for t in test if t[0] in user_list]  # 학습데이터가 없는 유저 제거
    print(f'-> test set size : {len(test)}')

    # user mask (유저 아이디 변경 지점)
    train.loc[train.user_id.diff(1) != 0, 'user_mask'] = 1
    train['user_mask'].fillna(0, inplace=True)
    # session mask (세션 아이디 변경 지점)
    train.loc[train.session_id.diff(1) != 0, 'session_mask'] = 1
    train['session_mask'].fillna(0, inplace=True)

    train['Rating'] = 5  # 사용자가 해당글을 좋아한다고 가정

    train = train[['item_id', 'user_id', 'Rating', 'target_id', 'session_id', 'Timestamp', 'user_mask', 'session_mask']]
    items = items[["item_id", "magazine_id", "user_id", "title", "sub_title", "keyword_list", "display_url", "id"]]
    users = users[['user_id', 'keyword_list', 'following_list', 'id']]
    return train, test, items, users


def preprocess_data(data_type: str, train: DataFrame, test: list, items: DataFrame, users: DataFrame) -> tuple[
    DataFrame, list, DataFrame, DataFrame]:
    if data_type == '10M':
        loading_function = movielens_preprocess
    elif data_type == '1M':
        loading_function = movielens_preprocess
    elif data_type == 'BRUNCH':
        loading_function = brunch_preprocess
    else:
        raise ValueError(f"unknown data type {data_type}")

    return loading_function(train, test, items, users)


if __name__ == '__main__':
    argument = args()

    train_data, test_data, item_meta, user_meta = loading_data(argument.dataset)
    train_data, test_data, item_meta, user_meta = preprocess_data(argument.dataset, train_data, test_data, item_meta,
                                                                  user_meta)
    context_data = get_last_n_session(train_data, 5)
    # train, valid = split_test_by_session(interactions, n_context_session=3)

    # get each item's popularity for negative sampling
    train_data['item_count'] = 1
    item_counts = train_data.groupby('item_id')['item_count'].sum().reset_index()
    item_counts['cumulate_count'] = [c for c in accumulate(item_counts.item_count)]

    item_meta = item_meta.merge(
        item_counts, on='item_id', how='left', validate='1:1'
    )
    item_meta[['item_count', 'cumulate_count']] = item_meta[['item_count', 'cumulate_count']].fillna(0).astype(int)

    train_dataset = format_dataset(train_data)
    context_dataset = format_dataset(context_data)

    save_dir = os.path.join(CONFIG.DATA, argument.dataset)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    pickle.dump(
        train_dataset, open(os.path.join(save_dir, f'train.pkl'), 'wb')
    )
    pickle.dump(
        context_dataset, open(os.path.join(save_dir, f'valid.pkl'), 'wb')
    )

    item_meta.to_csv(os.path.join(save_dir, f'item_meta.tsv'), index=False, sep='\t')
    user_meta.to_csv(os.path.join(save_dir, f'user_meta.tsv'), index=False, sep='\t')

    with open(os.path.join(save_dir, 'negative_test.dat'), 'w') as f:
        for row in test_data:
            row = '\t'.join([str(v) for v in row])
            f.write(row + '\n')
