import json
import os
import random
import re
from itertools import accumulate
from typing import Any

import pandas as pd

from common.utils import progressbar, to_timestampe
from config import CONFIG

random.seed(42)


def train_test_split(df, user_col, item_col):
    last_action_time = df.groupby(user_col)[item_col].transform('max')

    test = df[df[item_col] == last_action_time]
    train = df[df[item_col] != last_action_time]

    test = test.groupby(user_col).first().reset_index()

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

    train, test = train_test_split(ratings, user_col='UserID', item_col='Timestamp')

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

    train, test = train_test_split(ratings, user_col='UserID', item_col='Timestamp')

    print(f'train data size : {len(train)}, test data size : {len(test)}')
    print(f'total item size : {len(movies)}, total user size : {len(users)}')

    return train, test, movies, users


def loading_brunch(file_path):
    logfile_dir = os.path.join(file_path, 'read')
    item_meta_file = os.path.join(file_path, 'metadata.json')
    user_meta_file = os.path.join(file_path, 'users.json')

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

        progressbar(total, total, prefix='metadata loading...')
        return pd.DataFrame.from_dict(meta_data, orient='index')

    interactions = logfile_to_df(logfile_dir)
    item_meta = metadata_to_df(item_meta_file)
    user_meta = metadata_to_df(user_meta_file)

    train, test = train_test_split(interactions, user_col='UserID', item_col='Timestamp')

    print(f'train data size : {len(train)}, test data size : {len(test)}')
    print(f'total item size : {len(item_meta)}, total user size : {len(user_meta)}')

    return train, test, item_meta, user_meta


def uniform_random_sample(n, exclude_items, items):
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
            inter_items = set([])

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

    if data_type == '10M':
        item_col = 'item_id'
        file_path = os.path.join(CONFIG.DATA, 'movielens', 'ml-10M100K')
        loading_function = loading_movielens_10m
    elif data_type == '1M':
        item_col = 'item_id'
        file_path = os.path.join(CONFIG.DATA, 'movielens', 'ml-1m')
        loading_function = loading_movielens_1m
    elif data_type == 'BRUNCH':
        item_col = 'article_id'
        file_path = os.path.join(CONFIG.DATA, 'brunch_view')
        loading_function = loading_brunch
    else:
        raise ValueError(f"unknown data type {data_type}")

    train, test, item, user = loading_function(file_path)

    test_negative = get_negative_samples(train, test, user_col, item_col, n_sample=99, method='random')

    return train, test_negative, item, user
