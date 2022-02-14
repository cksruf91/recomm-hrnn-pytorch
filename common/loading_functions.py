import json
import os
import random
import re
from itertools import accumulate
from typing import Any, List, Tuple

import pandas as pd
from pandas import DataFrame

from common.utils import progressbar, to_timestampe
from config import CONFIG

random.seed(42)


def train_test_split(df: DataFrame, user_col: str, time_col: str) -> Tuple[DataFrame, DataFrame]:
    """ 학습 테스트 데이터 분리 함수
    각 유저별 마지막 interaction 읕 테스트로 나머지를 학습 데이터셋으로 사용

    Args:
        df: 전체 데이터
        user_col: 기준 유저 아이디 컬럼명
        time_col: 기준 아이템 아이디 컬럼명

    Returns: 학습 데이터셋, 테스트 데이터셋
    """
    last_action_time = df.groupby(user_col)[time_col].transform('max')

    test = df[df[time_col] == last_action_time]
    train = df[df[time_col] != last_action_time]

    test = test.groupby(user_col).first().reset_index()

    print(f'test set size : {len(test)}')
    user_list = train[user_col].unique()
    drop_index = test[test[user_col].isin(user_list) == False].index
    test.drop(drop_index, inplace=True)
    print(f'-> test set size : {len(test)}')

    return train, test


def loading_movielens_1m(file_path):
    ratings_header = "UserID::MovieID::Rating::Timestamp"
    movies_header = "MovieID::Title::Genres"
    user_header = "UserID::Gender::Age::Occupation::Zip-code"

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

    users = pd.read_csv(
        os.path.join(file_path, 'users.dat'),
        sep='::', header=None, names=user_header.split('::'),
        engine='python', encoding='iso-8859-1'
    )

    # MovieID -> item_id
    org_movie_id = set(ratings['MovieID'].unique().tolist() + movies['MovieID'].unique().tolist())
    movie_id_mapper = {
        movie_id: item_id for item_id, movie_id in enumerate(org_movie_id)
    }

    ratings['item_id'] = ratings['MovieID'].map(lambda x: movie_id_mapper[x])
    movies['item_id'] = movies['MovieID'].map(lambda x: movie_id_mapper[x])

    # UserID -> user_id
    org_user_id = set(ratings['UserID'].unique().tolist() + users['UserID'].unique().tolist())
    user_id_mapper = {
        user_id: user_index_id for user_index_id, user_id in enumerate(org_user_id)
    }

    ratings['user_id'] = ratings['UserID'].map(lambda x: user_id_mapper[x])
    users['user_id'] = users['UserID'].map(lambda x: user_id_mapper[x])

    train, test = train_test_split(ratings, user_col='UserID', time_col='Timestamp')

    print(f'train data size : {len(train)}, test data size : {len(test)}')
    print(f'total item size : {len(movies)}, total user size : {len(users)}')

    return train, test, movies, users


def loading_movielens_10m(file_path):
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

    users = pd.DataFrame({'UserID': ratings['UserID'].unique()})

    # MovieID -> item_id
    org_movie_id = set(ratings['MovieID'].unique().tolist() + movies['MovieID'].unique().tolist())
    movie_id_mapper = {
        movie_id: item_id for item_id, movie_id in enumerate(org_movie_id)
    }
    ratings['item_id'] = ratings['MovieID'].map(lambda x: movie_id_mapper[x])
    movies['item_id'] = movies['MovieID'].map(lambda x: movie_id_mapper[x])

    # UserID -> user_id
    org_user_id = ratings['UserID'].unique().tolist()
    user_id_mapper = {
        user_id: user_index_id for user_index_id, user_id in enumerate(org_user_id)
    }
    ratings['user_id'] = ratings['UserID'].map(lambda x: user_id_mapper[x])
    users['user_id'] = users['UserID'].map(lambda x: user_id_mapper[x])

    train, test = train_test_split(ratings, user_col='UserID', time_col='Timestamp')

    print(f'train data size : {len(train)}, test data size : {len(test)}')
    print(f'total item size : {len(movies)}, total user size : {len(users)}')

    return train, test, movies, users


def loading_brunch(file_path):
    logfile_dir = os.path.join(file_path, 'read')
    item_meta_file = os.path.join(file_path, 'metadata.json')
    user_meta_file = os.path.join(file_path, 'users.json')

    def logfile_to_df(dir):
        files = os.listdir(dir)

        # logfile to Dataframe
        interactions = {
            'UserID': [], 'ArticleID': [], 'Timestamp': []
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
                            interactions['ArticleID'].append(item)

                            log_datetime = file.split('_')[0]
                            interactions['Timestamp'].append(to_timestampe(log_datetime, '%Y%m%d%H'))

        return pd.DataFrame(interactions)

    def metadata_to_df(meta_data_file):
        # metadata to Dataframe
        meta_data = {}
        result = os.popen(f'wc -l {meta_data_file}').read()
        total = int(result.split()[0])
        with open(meta_data_file, 'r') as f:
            for i, line in enumerate(f):
                progressbar(total, i + 1, prefix='metadata loading...')
                meta_data[i] = json.loads(line)

        return pd.DataFrame.from_dict(meta_data, orient='index')

    interactions = logfile_to_df(logfile_dir)
    user_meta = metadata_to_df(user_meta_file)
    item_meta = metadata_to_df(item_meta_file)

    # random sampling
    total_user = interactions['UserID'].nunique()
    total_interactions = len(interactions)
    user_id_list = interactions['UserID'].unique().tolist()
    user_id_list = random.sample(user_id_list, k=int(len(user_id_list) * 0.1))

    # sampling interactions
    drop_index = interactions[interactions['UserID'].isin(user_id_list) == False].index
    interactions.drop(drop_index, inplace=True)
    print(f"UserID {total_user} -> {len(user_id_list)}")
    print(f"UserID {total_interactions} -> {len(interactions)}")

    # sampling user dataset
    total_user_meta = len(user_meta)
    drop_index = user_meta[user_meta['id'].isin(user_id_list) == False].index
    user_meta.drop(drop_index, inplace=True)
    print(f"user meta data {total_user_meta} -> {len(user_meta)}")

    # sampling item dataset
    total_item_meta = len(item_meta)
    item_sample_list = interactions['ArticleID'].unique().tolist()
    drop_index = item_meta[item_meta['id'].isin(item_sample_list) == False].index
    item_meta.drop(drop_index, inplace=True)
    print(f"item meta data {total_item_meta} -> {len(item_meta)}")

    # UserID -> user_id
    user_id_mapper = {
        UserID: user_id for user_id, UserID in enumerate(interactions.UserID.unique())
    }
    interactions['user_id'] = interactions['UserID'].map(lambda x: user_id_mapper[x])
    user_meta['user_id'] = user_meta['id'].map(lambda x: user_id_mapper[x])

    # MovieID -> item_id
    article_id_mapper = {
        ArticleID: item_id for item_id, ArticleID in enumerate(interactions.ArticleID.unique())
    }
    interactions['item_id'] = interactions['ArticleID'].map(lambda x: article_id_mapper[x])
    item_meta['item_id'] = item_meta['id'].map(lambda x: article_id_mapper[x])

    train, test = train_test_split(interactions, user_col='user_id', time_col='Timestamp')

    print(f'train data size : {len(train)}, test data size : {len(test)}')
    print(f'total item size : {len(item_meta)}, total user size : {len(user_meta)}')

    return train, test, item_meta, user_meta


def uniform_random_sample(n: int, exclude_items: List, items: List) -> List:
    """ 랜덤 샘플링

    Args:
        n: 샘플 갯수
        exclude_items: 샘플에서 제외되어야 할 아이템 list
        items: 샘플링할 모집단 아이템 list

    Returns: n개의 샘플 아이템 list
    """
    sample = []
    while len(sample) < n:
        n_item = random.choice(items)
        if n_item in exclude_items:
            continue
        if n_item in sample:
            continue
        sample.append(n_item)
    assert len(sample) == n
    return sample


def weighted_random_sample(n, exclude_items, items, cum_sums):
    """ 아이템 인기도에 의한 가중 샘플링

    Args:
        n: 샘플 갯수
        exclude_items: 샘플에서 제외되어야 할 아이템 list
        items: 샘플링할 모집단 아이템 list
        cum_sums: 아이템 출현 횟수 누적합

    Returns: n개의 샘플 아이템 list
    """
    n_items = len(exclude_items)
    samples = random.choices(
        items, cum_weights=cum_sums, k=n_items + n + 100
    )

    sample = list(set(samples) - exclude_items)
    sample = sample[:n]
    assert len(sample) == n
    return sample


def get_negative_samples(train_df, test_df, user_col, item_col, n_sample=99, method='random'):
    negative_sampled_test = []

    # 샘플링을 위한 아이템들의 누적합
    train_df['item_count'] = 1
    item_counts = train_df.groupby(item_col)['item_count'].sum().reset_index()
    item_counts['cumulate_count'] = [c for c in accumulate(item_counts.item_count)]

    # 샘플링을 위한 변수
    item_list = item_counts[item_col].tolist()
    item_cumulate_count = item_counts['cumulate_count'].tolist()

    # 유저가 이전에 interaction 했던 아이템들
    user_interactions = train_df.groupby(user_col)[item_col].agg(lambda x: set(x.tolist()))

    for uid, iid in zip(test_df[user_col].tolist(), test_df[item_col].tolist()):
        row = [uid, iid]

        try:
            inter_items = user_interactions[uid]
        except KeyError as e:
            continue

        if method == 'random':
            sample = uniform_random_sample(n_sample, inter_items, item_list)
        elif method == 'weighted':
            sample = weighted_random_sample(n_sample, inter_items, item_list, cum_sums=item_cumulate_count)
        else:
            raise ValueError(f"invalid sampling method {method}")

        row.extend(sample)

        negative_sampled_test.append(row)

    return negative_sampled_test


def loading_data(data_type: str) -> tuple[Any, list[list], Any, Any]:
    user_col = 'user_id'
    item_col = 'item_id'

    if data_type == '10M':
        file_path = os.path.join(CONFIG.DATA, 'movielens', 'ml-10M100K')
        train, test, item, user = loading_movielens_10m(file_path)
    elif data_type == '1M':
        file_path = os.path.join(CONFIG.DATA, 'movielens', 'ml-1m')
        train, test, item, user = loading_movielens_1m(file_path)
    elif data_type == 'BRUNCH':
        file_path = os.path.join(CONFIG.DATA, 'brunch_view')
        train, test, item, user = loading_brunch(file_path)
    else:
        raise ValueError(f"unknown data type {data_type}")

    test_negative = get_negative_samples(train, test, user_col, item_col, n_sample=99, method='random')

    return train, test_negative, item, user
