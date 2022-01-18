import argparse
import os
import pickle
from itertools import accumulate
from typing import Dict, Tuple

import pandas as pd

from common.utils import progressbar, DefaultDict
from config import CONFIG


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='10M', choices=['10M', '1M'], help='데이터셋', type=str)
    return parser.parse_args()


def loading_movielens(data_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    if data_type == '10M':
        MOVIELENS = os.path.join(CONFIG.DATA, 'movielens', 'ml-10M100K')
    elif data_type == '1M':
        MOVIELENS = os.path.join(CONFIG.DATA, 'movielens', 'ml-1m')
    else:
        raise ValueError(f"unknown data type {datatype}")
    
    ratings_header = "UserID::MovieID::Rating::Timestamp"
    tags_header = "UserID::MovieID::Tag::Timestamp"
    movies_header = "MovieID::Title::Genres"

    def create_session_id(df: pd.DataFrame):
        df.sort_values(['UserID', 'Timestamp'], inplace=True)
        df['timedelta'] = df.groupby('UserID')['Timestamp'].diff(1)
        df['timedelta'].fillna(0, inplace=True)
        df['session_id'] = 0
        df.loc[df['timedelta'] > 60 * 60, 'session_id'] = 1
        df['session_id'] = df.groupby('UserID')['session_id'].cumsum()
        return df

    def drop_sparse_item(df, count):
        return df[df.groupby("MovieID")["UserID"].transform('count') >= count]

    def drop_sparse_session(df, count):
        return df[
            df.groupby(["UserID", "session_id"])["MovieID"].transform('count') >= count
            ]

    def drop_sparse_user(df, min_cnt, max_cnt):
        session_count = df.groupby("UserID")["session_id"].transform('nunique')
        return df[session_count.between(min_cnt, max_cnt)]

    ratings = pd.read_csv(
        os.path.join(MOVIELENS, 'ratings.dat'),
        sep='::', header=None, names=ratings_header.split('::'),
        engine='python'
    )

    movies = pd.read_csv(
        os.path.join(MOVIELENS, 'movies.dat'),
        sep='::', header=None, names=movies_header.split('::'),
        engine='python', encoding='iso-8859-1'
    )

    print(f"ratings dataset : {len(ratings)}")
    ratings = create_session_id(ratings)
    ratings = drop_sparse_item(ratings, 10)
    print(f"drop sparse item : {len(ratings)}")
    ratings = drop_sparse_session(ratings, 5)
    print(f"drop sparse session : {len(ratings)}")
    ratings = drop_sparse_user(ratings, 5, 99)
    print(f"drop sparse user : {len(ratings)}")

    item_id_mapper = DefaultDict(None, {
        movie_id: item_id for item_id, movie_id in enumerate(ratings['MovieID'].unique())
    })
    print(f"unique item count : {len(item_id_mapper)}")

    movies['item_id'] = movies['MovieID'].map(lambda x: item_id_mapper[x])
    movies = movies[movies.item_id.notnull()]
    movies.item_id = movies.item_id.astype(int)
    
    ratings['item_id'] = ratings['MovieID'].map(lambda x: item_id_mapper[x])
    ratings.rename(columns={'UserID': 'user_id'}, inplace=True)

    # output value(target id) 생성 | input : 현재 item id, output 다음 item id
    target_id = ratings.groupby(['user_id', 'session_id'])['item_id'].apply(
        lambda col: pd.Series(col[1:].append(pd.Series([-1])))
    )
    ratings['target_id'] = target_id.tolist()
    ratings = ratings[ratings.target_id != -1]

    # user mask (유저 아이디 변경 지점)
    ratings.loc[ratings.user_id.diff(1) != 0, 'user_mask'] = 1
    ratings['user_mask'].fillna(0, inplace=True)

    # session mask (세션 아이디 변경 지점)
    ratings.loc[ratings.session_id.diff(1) != 0, 'session_mask'] = 1
    ratings['session_mask'].fillna(0, inplace=True)

    ratings = ratings[
        ['user_id', 'item_id', 'target_id', 'session_id', 'Timestamp', 'MovieID', 'user_mask', 'session_mask']
    ]
    movies = movies[['item_id', 'MovieID', 'Title', 'Genres']]
    return ratings, movies


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
