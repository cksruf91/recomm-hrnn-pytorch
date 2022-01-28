import random
from typing import Iterator, List, Tuple, Dict

import numpy as np
import torch
from pandas import DataFrame
from torch import Tensor


class NegativeSampler:

    def __init__(self, item_meta: DataFrame, sample_size: int):
        """ 아이템의 인기를 가중치로 아이템을 random sample 하는 샘플링 모듈

        Args:
            item_meta: item_id와 cumulate_count(아이템의 출현횟수 누적합) 컬럼을 가지고 있는 데이터 프레임
            sample_size: 네거티브 샘플링할 샘플의 개수
        """
        item_meta = item_meta.sort_values('cumulate_count', ascending=True, inplace=False)
        
        self.item_meta = item_meta[item_meta['cumulate_count'] != 0]
        self.item_list = self.item_meta.item_id.tolist()
        self.item_cumulate_count = self.item_meta.cumulate_count.tolist()
        self.sample_size = sample_size

    def __call__(self, positive_items: list) -> np.array:
        """ 네거티브 샘플링 메소드

        Args:
            positive_items: 타겟 아이템

        Returns: 샘플링된 아이템에서 positive_items 는 제외한 후 반환

        """

        n_sample = len(positive_items) + self.sample_size
        samples = random.choices(
            self.item_list, cum_weights=self.item_cumulate_count, k=n_sample
        )

        sample = list(set(samples) - set(positive_items))[:self.sample_size]  # 네거티브 샘플에서 정상적인 아이템은 제거

        return torch.tensor(sample, dtype=torch.int64)


def data_iterator(user_data: List[Dict]) -> Iterator:
    """ 유저 interaction 데이터를 순서대로 생성하는 Iterator 생성하는 함수

    Args:
        user_data: 학습데이터
            예시) [row data, row data ....],
                - row data = {'inputItem': 7, 'outputItem': 5196, 'userMask': 1.0, 'sessionMask': 0.0, 'conText': True}

    Returns: Iterator
    """
    for data in user_data:
        yield data


class DataLoader:

    def __init__(self, data: Dict[int, list], batch_size: int = 4, device: torch.device = torch.device('cpu'),
                 negative_sampler: NegativeSampler = None):
        """ session-parallel mini-batch mechanism 을 위한 데이터 제너레이터
        paper : https://arxiv.org/pdf/1706.04148.pdf

        Args:
            data(dict): 학습데이터
                예시) {user_id1 : [row data, row data ....], user_id2 : [row data, row data ....], .... },
                    row data = {
                        'inputItem': 7, 'outputItem': 5196, 'userMask': 1.0, 'sessionMask': 0.0, 'conText': True
                    }
            batch_size(int): 배치 사이즈, 1 이상 전체 유저수 이하
            device: 모델 학습 device, torch.device('cpu') or torch.device('cuda')
            negative_sampler: 아이템의 인기도를 가중치로 아이템 아이디를 list로 반환하는 샘플러
        """
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.negative_sampler = negative_sampler
        self.n_sample_size = self.negative_sampler.sample_size if self.negative_sampler is not None else 0

        self.user_list = list(self.data.keys())
        self.i = 0
        self.batch_users_iter = []  # list fo Iterator (data_iterator)
        self.shuffle_user()

    def __len__(self) -> int:
        """전체 데이터 길이
        배치 사이즈 만큼 데이터를 생성할 유저가 부족하면 iteration이 끝나기 때문에
        실제 생성되는 데이터의 건수는 더 적을수 있음(random)

        Returns(int): length of total dataset
        """
        length = 0
        for user_id in self.data:
            length += len(self.data[user_id])
        return int(length / self.batch_size)

    def __iter__(self):
        """ iter(self) """
        return self  # 현재 인스턴스를 반환

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Returns(torch.tensor):
            input 아이템 index, output 아이템 index,
            user_mask(유저가 변경될 경우 1 아님 0),
            session_mask(세션이 변경될 경우 1 아님 0)
            context(테스트를 위한 context 세션일 경우 True 아님 False)
        """
        input_item = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        output_item = torch.zeros(self.batch_size + self.n_sample_size, dtype=torch.int64, device=self.device)
        user_mask = torch.zeros([self.batch_size, 1], dtype=torch.int64, device=self.device)
        session_mask = torch.zeros([self.batch_size, 1], dtype=torch.int64, device=self.device)
        context = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        for j in range(self.batch_size):
            try:
                
                batch_data = next(self.batch_users_iter[j])
                
            except StopIteration as e:  # iteration 이 끝난경우
                self.batch_users_iter[j] = self.assign_iterator()  # 다른유저로 새 Iterator 생성
                batch_data = next(self.batch_users_iter[j])
            
            input_item[j] = batch_data['inputItem']
            output_item[j] = batch_data['outputItem']
            user_mask[j] = batch_data['userMask']
            session_mask[j] = batch_data['sessionMask']
            context[j] = batch_data['conText']
        
        n_samples = self.negative_sampler(output_item.cpu().tolist()) \
            if self.negative_sampler is not None else torch.tensor([], dtype=torch.int64)

        output_item[j + 1:] = n_samples

        return input_item, output_item, user_mask, session_mask, context

    def assign_iterator(self) -> Iterator:
        """ 유저 데이터를 순서대로 다 사용했을 경우 새로운 유저의 데이터 제너레이터를 생성
        더이상 유저가 없을경우 iteration 을 종료 하도록 StopIteration 에러 발생

        Returns (Iterator) : 데이터 제너레이터
        """
        try:
            user_id = self.user_list[self.i]
        except IndexError:  # 더이상 유저가 없는 경우
            self.shuffle_user()
            raise StopIteration  # iteration 종료

        self.i += 1
        return data_iterator(self.data[user_id])

    def shuffle_user(self) -> None:
        """한 에폭이 끝나면 전체 유저의 순서를 섞음

        Returns : None
        """
        random.shuffle(self.user_list)

        if len(self.user_list) < self.batch_size:
            raise ValueError(f'batch size {self.batch_size} greater then number of user : {len(self.user_list)}')

        self.i = 0
        self.batch_users_iter = []
        for _ in range(self.batch_size):
            self.batch_users_iter.append(self.assign_iterator())
