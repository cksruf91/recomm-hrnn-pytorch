from typing import Iterator, List, Tuple, Dict
from collections import defaultdict

import pandas as pd


class PopularItemRecommendation:
    
    def __init__(self, train_dataset: Dict[int, list], item_meta: pd.DataFrame, k: int, min_cnt: int=10):
        
        item_meta.sort_values('item_count', ascending=False, inplace=True)
        self.item_meta = item_meta[item_meta['item_count'] > min_cnt]
        self.pop_item_list = item_meta.item_id.tolist()
        self.k = k
        
        self.user_interactions = defaultdict(list)
        for i, uid in enumerate(train_dataset):
            for row in train_dataset[uid]:
                self.user_interactions[uid].append(row['inputItem'])

    def get_recommendation(self, user_id, item_id):
        
        self.user_interactions[user_id].append(item_id)
        user_items = self.user_interactions[user_id]
        
        recomm_items = []
        j = 0
        while len(recomm_items) < self.k:
            try:
                pop_item = self.pop_item_list[j]
            except IndexError as e:
                break
            
            if pop_item not in user_items:
                recomm_items.append(pop_item)    
            j += 1
        
        
        return recomm_items

