import json
import os
import re
from typing import Tuple

import pandas as pd

from common.utils import DefaultDict, progressbar, to_timestampe
from config import CONFIG


def create_session_id(df: pd.DataFrame, time_delta: int):
    df.sort_values(['UserID', 'Timestamp'], inplace=True)
    df['timedelta'] = df.groupby('UserID')['Timestamp'].diff(1)
    df['timedelta'].fillna(0, inplace=True)
    df['session_id'] = 0
    df.loc[df['timedelta'] > time_delta, 'session_id'] = 1
    df['session_id'] = df.groupby('UserID')['session_id'].cumsum()
    return df


def drop_sparse_item(df: pd.DataFrame, count: int, item_id: str):
    return df[df.groupby(item_id)["UserID"].transform('count') >= count]


def drop_sparse_session(df: pd.DataFrame, count: int, item_id: str):
    return df[
        df.groupby(["UserID", "session_id"])[item_id].transform('count') >= count
        ]


def drop_sparse_user(df, min_cnt, max_cnt):
    session_count = df.groupby("UserID")["session_id"].transform('nunique')
    return df[session_count.between(min_cnt, max_cnt)]


def get_last_n_session(df: pd.DataFrame, n: int):
    return df[
        df["session_id"] >= df.groupby("UserID")["session_id"].transform(lambda x: max(max(x) - n, 0))
        ]


def loading_brunch(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logfile_dir = os.path.join(file_path, 'read')
    metadata_file = os.path.join(file_path, 'metadata.json')

    def logfile_to_df(logfile_dir):
        files = os.listdir(logfile_dir)

        # logfile to Dataframe
        interactions = {
            'UserID': [], 'article_id': [], 'Timestamp': []
        }

        total = len(files)
        for i, file in enumerate(files):
            progressbar(total, i + 1, prefix='loading...')
            if re.match('\..*', file) is None:  # 숨김파일 제외
                with open(os.path.join(logfile_dir, file), 'r') as f:
                    for lines in f:
                        line = lines.rstrip('\n').strip()
                        user, *items = line.split()
                        for item in items:
                            interactions['UserID'].append(user)
                            interactions['article_id'].append(item)

                            log_datetime = file.split('_')[0]
                            interactions['Timestamp'].append(to_timestampe(log_datetime, '%Y%m%d%H'))

        progressbar(total, total, prefix='loading...')
        print('Done')
        return pd.DataFrame(interactions)

    def metadata_to_df(metadata_file):
        # metadata to Dataframe
        meta_data = {}
        result = os.popen(f'wc -l {metadata_file}').read()
        total = int(result.split()[0])
        with open(metadata_file, 'r') as f:
            for i, line in enumerate(f):
                progressbar(total, i + 1, prefix='metadata loading...')
                meta_data[i] = json.loads(line)

        progressbar(total, total, prefix='metadata loading...')
        print('Done')
        return pd.DataFrame.from_dict(meta_data, orient='index')

    interactions = logfile_to_df(logfile_dir)
    meta_data = metadata_to_df(metadata_file)

    # preprocess
    interactions = create_session_id(interactions, time_delta=60 * 60 * 24)
    print(f"interactions dataset : {len(interactions)}")
    interactions = get_last_n_session(interactions, 30)
    print(f"get last n session : {len(interactions)}")
    interactions = drop_sparse_item(interactions, 10, item_id='article_id')
    print(f"drop sparse item : {len(interactions)}")
    interactions = drop_sparse_session(interactions, 3, item_id='article_id')
    print(f"drop sparse session : {len(interactions)}")
    interactions = drop_sparse_user(interactions, 3, 99)
    print(f"drop sparse user : {len(interactions)}")

    item_id_mapper = DefaultDict(None, {
        article_id: item_id for item_id, article_id in enumerate(interactions['article_id'].unique())
    })
    interactions['item_id'] = interactions.article_id.map(lambda x: item_id_mapper[x])
    print(f"unique item count : {len(item_id_mapper)}")

    interactions.rename(columns={'UserID': 'user_id'}, inplace=True)

    # user_id 변환
    user_id_mapper = DefaultDict(None, {
        user_value: user_id for user_id, user_value in enumerate(interactions['user_id'].unique())
    })
    interactions['user_id'] = interactions['user_id'].map(lambda x: user_id_mapper[x])

    # user mask (유저 아이디 변경 지점)
    interactions.loc[interactions.user_id.diff(1) != 0, 'user_mask'] = 1
    interactions['user_mask'].fillna(0, inplace=True)
    # session mask (세션 아이디 변경 지점)
    interactions.loc[interactions.session_id.diff(1) != 0, 'session_mask'] = 1
    interactions['session_mask'].fillna(0, inplace=True)

    # target_id 생성
    interactions.sort_values(['user_id', 'Timestamp'], inplace=True)
    interactions['target_id'] = interactions.item_id.tolist()[1:] + [-1]
    interactions.loc[interactions['user_mask'] == 1, 'target_id'] = -1
    interactions = interactions[interactions.target_id != -1]

    # 연속으로 글을 봤을경우 1로 처리
    interactions = interactions[interactions.item_id != interactions.target_id]

    # create item_meta dataframe
    item_meta = interactions.groupby('article_id')['Timestamp'].count().reset_index()
    item_meta.columns = ['article_id', 'counts']

    meta_data['article_id'] = meta_data['id']
    meta_data.drop('id', axis=1, inplace=True)

    item_meta = item_meta.merge(meta_data, on='article_id', how='left', validate='1:1')
    item_meta['item_id'] = item_meta.article_id.map(lambda x: item_id_mapper[x])

    interactions = interactions[
        ['user_id', 'item_id', 'target_id', 'session_id', 'Timestamp', 'article_id', 'user_mask', 'session_mask']
    ]

    item_meta = item_meta[
        ['item_id', 'article_id', 'title', 'sub_title', 'keyword_list', 'display_url', 'user_id', 'magazine_id',
         'reg_ts']
    ]

    return interactions, item_meta


def loading_movielens(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings_header = "UserID::MovieID::Rating::Timestamp"
    tags_header = "UserID::MovieID::Tag::Timestamp"
    movies_header = "MovieID::Title::Genres"

    ratings = pd.read_csv(
        os.path.join(file_path, 'ratings.dat'),
        sep='::', header=None, names=ratings_header.split('::'),
        engine='python'
    )

    movies = pd.read_csv(
        os.path.join(file_path, 'movies.dat'),
        sep='::', header=None, names=movies_header.split('::'),
        engine='python', encoding='iso-8859-1'
    )

    print(f"ratings dataset : {len(ratings)}")
    ratings = create_session_id(ratings, time_delta=60 * 60)
    ratings = drop_sparse_item(ratings, 10, item_id='MovieID')
    print(f"drop sparse item : {len(ratings)}")
    ratings = drop_sparse_session(ratings, 5, item_id='MovieID')
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


def loading_data(data_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if data_type == '10M':
        file_path = os.path.join(CONFIG.DATA, 'movielens', 'ml-10M100K')
        loading_function = loading_movielens
    elif data_type == '1M':
        file_path = os.path.join(CONFIG.DATA, 'movielens', 'ml-1m')
        loading_function = loading_movielens
    elif data_type == 'BRUNCH':
        file_path = os.path.join(CONFIG.DATA, 'brunch_view')
        loading_function = loading_brunch
    else:
        raise ValueError(f"unknown data type {data_type}")

    return loading_function(file_path)
