from typing import Dict, Tuple
from common.utils import progressbar, DefaultDict
import os

import pandas as pd

from config import CONFIG

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