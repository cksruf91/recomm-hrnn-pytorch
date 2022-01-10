import os
import pickle
from typing import Dict
from typing import Tuple

import pandas as pd

from common.utils import progressbar
from config import CONFIG

MOVIELENS = os.path.join(CONFIG.DATA, 'movielens', 'ml-10M100K')


def loading_movielens() -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        engine='python'
    )

    print(f"ratings dataset : {len(ratings)}")
    ratings = create_session_id(ratings)
    ratings = drop_sparse_item(ratings, 10)
    print(f"drop sparse item : {len(ratings)}")
    ratings = drop_sparse_session(ratings, 5)
    print(f"drop sparse session : {len(ratings)}")
    ratings = drop_sparse_user(ratings, 5, 99)
    print(f"drop sparse user : {len(ratings)}")

    item_id_mapper = {
        movie_id: item_id for item_id, movie_id in enumerate(movies['MovieID'])
    }
    print(f"unique item count : {len(item_id_mapper)}")

    movies['item_id'] = movies['MovieID'].map(lambda x: item_id_mapper[x])
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
    movies = movies[['item_id', 'MovieID', 'Title']]
    return ratings, movies


def split_test_by_session(df):
    last_session = df.groupby('user_id')['session_id'].max().reset_index()
    last_session['test'] = 1

    df = df.merge(
        last_session, on=['user_id', 'session_id'], how='left', validate='m:1'
    )
    df['test'].fillna(0, inplace=True)
    train_data = df[df['test'] == 0].drop('test', axis=1)
    test_data = df[df['test'] == 1].drop('test', axis=1)

    test_data = test_data[test_data["item_id"].isin(train["item_id"].unique())]

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
            row data = {'inputItem': 7, 'outputItem': 5196, 'userMask': 1.0, 'sessionMask': 0.0}

    """

    # interactions = namedtuple('Interactions', ['inputItem', 'outputItem', 'userMask', 'sessionMask'])

    def parsing_row(row):
        return {
            'inputItem': row['item_id'], 'outputItem': row['target_id'], 'userMask': row['user_mask'],
            'sessionMask': row['session_mask']
        }

    data = {}
    total_user = df.user_id.nunique()

    for i, (user_id, df) in enumerate(df.groupby('user_id')):
        progressbar(total_user, i + 1, prefix='parsing data ')
        data[user_id] = [parsing_row(row) for index, row in df.iterrows()]
    print(' Done')
    return data


if __name__ == '__main__':
    interactions, item_meta = loading_movielens()
    print(interactions.head())

    train, test = split_test_by_session(interactions)
    train, valid = split_test_by_session(train)

    train_dataset = format_dataset(train)
    valid_dataset = format_dataset(valid)
    test_dataset = format_dataset(test)

    pickle.dump(train_dataset, open(os.path.join(CONFIG.DATA, 'train.pkl'), 'wb'))
    pickle.dump(valid_dataset, open(os.path.join(CONFIG.DATA, 'valid.pkl'), 'wb'))
    pickle.dump(test_dataset, open(os.path.join(CONFIG.DATA, 'test.pkl'), 'wb'))

    item_meta.to_csv(os.path.join(CONFIG.DATA, 'item_meta.csv'), index=False)
