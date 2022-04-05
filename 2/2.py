# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 10:47:40 2022

@author: Lenovo
"""

#1. Реализовать бейзлайн Weighted random recommender
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix
from copy import deepcopy

# Детерминированные алгоритмы
from implicit.nearest_neighbours import ItemItemRecommender, CosineRecommender, TFIDFRecommender, BM25Recommender

# Метрики
from implicit.evaluation import train_test_split
from implicit.evaluation import precision_at_k, mean_average_precision_at_k, AUC_at_k, ndcg_at_k

data = pd.read_csv('retail_train.csv')
print(data.head())
print(data.shape)

#Разделим выборку как в уроке, только в качестве тестовой выборки возьмем данные за последние 4 недели (~1 месяц)
test_size_weeks = 4

data_train = data[data['week_no'] < data['week_no'].max() - test_size_weeks]
data_test = data[data['week_no'] >= data['week_no'].max() - test_size_weeks]

data_train.shape[0], data_test.shape[0]

#список покупок у конкретных пользователей
result = data_test.groupby('user_id')['item_id'].unique().reset_index()
result.columns=['user_id', 'actual']
result.head(2)

#для применения алгоритма Weighted random recommender необходимо получить список товаров с определенным весом
#взвешивать будем по sales_value

goods_list_train = data_train.groupby('item_id').sum()
goods_list_train = pd.DataFrame({'item_id': goods_list_train.index, 'sales_value': goods_list_train.sales_value})
print(goods_list_train.head(5))
print(len(goods_list_train))

#применяем нормирование по минимаксу и делим на сумму полученных значений
sales_min = np.min(goods_list_train.sales_value)
sales_max = np.max(goods_list_train.sales_value)
goods_list_train.sales_value = (goods_list_train.sales_value-sales_min)/(sales_max-sales_min)
goods_list_train.sales_value = goods_list_train.sales_value/np.sum(goods_list_train.sales_value)
goods_list_train.columns = ['item_id', 'weight']
print(goods_list_train.head(5))
print('sales_value: min: {}, max: {}'.format(sales_min,sales_max))

def weighted_random_recommendation(items_weights, n=5):
    """Случайные рекоммендации
    
    Input
    -----
    items_weights: pd.DataFrame
        Датафрейм со столбцами item_id, weight. Сумма weight по всем товарам = 1
    """
    
    # Подсказка: необходимо модифицировать функцию np.random.choice(items, size=n, replace=False, p=)
    # your_code
    recs = np.random.choice(items_weights['item_id'], size=n, replace=False, p=items_weights['weight'])
    
    return recs.tolist()

# your_code
rec_res = weighted_random_recommendation(goods_list_train)
print(rec_res)