# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 17:22:54 2022

@author: a_bredihin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

# Функции из 1-ого вебинара
import os, sys


#загрузим данные о покупках
global data
global needed_goods

data = pd.read_csv('transaction_data.csv')

data.columns = [col.lower() for col in data.columns]
data.rename(columns={'household_key': 'user_id',
                    'product_id': 'item_id'},
           inplace=True)


test_size_weeks = 3

data_train = data[data['week_no'] < data['week_no'].max() - test_size_weeks]
data_test = data[data['week_no'] >= data['week_no'].max() - test_size_weeks]

data_train.head(2)


#загрузим данные о продуктах


item_features = pd.read_csv('product.csv')
item_features.columns = [col.lower() for col in item_features.columns]
item_features.rename(columns={'product_id': 'item_id'}, inplace=True)

item_features.head(2)


#фильтрация товаров


def prefilter_items():
    # Уберем самые популярные товары (их и так купят) - В КУРСОВОМ ПОЛЬЗОВАТЬ ТОЛЬКО ЕСЛИ РАСТЕТ СКОР
    global data
    global needed_goods
    
    popularity = data_train.groupby('item_id')['user_id'].nunique().reset_index() #ЗДЕСЬ НЕПРАВИЛЬНО - id товара ДЕЛЯТСЯ
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)    
    popularity.share_unique_users = popularity.share_unique_users / data_train['user_id'].nunique() 
    
    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]
    
    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    
    # Уберем не интересные для рекоммендаций категории (department)
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб. 
    
    # Уберем слишком дорогие товарыs
    
    # ...
    

#Вычисление рекомендаций

'''
def get_recommendations(user, model, sparse_user_item, N=5):
    """Рекомендуем топ-N товаров"""
    
    res = [id_to_itemid[rec[0]] for rec in 
                    model.recommend(userid=userid_to_id[user], 
                                    user_items=sparse_user_item,   # на вход user-item matrix
                                    N=N, 
                                    filter_already_liked_items=False, 
                                    filter_items=[itemid_to_id[999999]],  # !!! 
                                    recalculate_user=True)]
    return res


#Рекомендации товаров на основе покупок конкретного пользователя


def get_similar_items_recommendation(user, model, N=5):
    """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
    
    # your_code
    
    return res


#Рекомендации товаров на основе наиболее часто покупаемых товаров конкретным пользователем


def get_similar_users_recommendation(user, model, N=5):
    """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
    # your_code
    
    return res

'''
#Подготавливаем матрицу user-item и обучаем модель ALS


#УБИРАЕМ НЕПОПУЛЯРНЫЕ ТОВАРЫ
prefilter_items()

#пересоздаем выборки по новой
data_train = data[data['week_no'] < data['week_no'].max() - test_size_weeks]
data_test = data[data['week_no'] >= data['week_no'].max() - test_size_weeks]

user_item_matrix = pd.pivot_table(data_train, 
                                  index='user_id', columns='item_id', 
                                  values='quantity', # Можно пробоват другие варианты
                                  aggfunc='count', 
                                  fill_value=0
                                 )

user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit

user_item_matrix.head(3)




userids = user_item_matrix.index.values
itemids = user_item_matrix.columns.values

matrix_userids = np.arange(len(userids))
matrix_itemids = np.arange(len(itemids))

id_to_itemid = dict(zip(matrix_itemids, itemids))
id_to_userid = dict(zip(matrix_userids, userids))

itemid_to_id = dict(zip(itemids, matrix_itemids))
userid_to_id = dict(zip(userids, matrix_userids))




user_item_matrix = bm25_weight(user_item_matrix.T).T  # Применяется к item-user матрице !
user_item_matrix




model = AlternatingLeastSquares(factors=44,
                                regularization=0.001,
                                iterations=20,
                                calculate_training_loss=True, 
                                use_gpu=False)

model.fit(csr_matrix(user_item_matrix).T.tocsr(),  # На вход item-user matrix
          show_progress=True)

def get_recommendations(user, model, sparse_user_item, N=5):
    """Рекомендуем топ-N товаров"""
    
    res = [id_to_itemid[rec[0]] for rec in 
                    model.recommend(userid=userid_to_id[user], 
                                    user_items=sparse_user_item[user],   # на вход user-item matrix
                                    N=N, 
                                    filter_already_liked_items=False, 
                                    #filter_items=[itemid_to_id[999999]],  # !!! 
                                    recalculate_user=True)]
    return res


#проверяем модель на тестовых данных
result = data_test.groupby('user_id')['item_id'].unique().reset_index()
result.columns=['user_id', 'actual']
result.head(2)

#создаем копию матрицы user-item в виде numpy array
user_item_matrix_csr = user_item_matrix.tocsr()
del user_item_matrix #освободим память... все равно вылетает
result['bm25'] = result['user_id'].apply(lambda x: get_recommendations(x, model, user_item_matrix_csr, N=5))