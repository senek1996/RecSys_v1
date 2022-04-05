# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:27:40 2022

@author: Lenovo
"""

import numpy as np
from copy import deepcopy

recommended_list = [143, 156, 1134, 991, 27, 1543, 3345, 533, 11, 43, 32] #id товаров
bought_list = [521, 32, 143, 991]

#1 - hit rate at k
def hit_rate_at_k(recommended_list, bought_list, k=5):
    
    # your_code
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    
    flags = np.isin(bought_list, recommended_list)
    
    hit_rate = (flags.sum() > 0) * 1
    
    return hit_rate

hit_rate_at_k(recommended_list, bought_list)

#2 - money precision at k
prices_recommended = [150, 115, 70, 95, 200, 450, 45, 280, 140, 230, 110] #стоимости товаров

def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
        
    # your_code
    # Лучше считать через скалярное произведение, а не цикл
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    
    #МЕНЯЕМ МЕСТАМИ, ЧТОБЫ ВЫЧИСЛЯТЬ ТОЧНОСТЬ ИМЕННО ПО ЦЕНАМ РЕКОМЕНДАЦИЙ
    flags = np.isin(recommended_list, bought_list)
    
    precision = np.dot(prices_recommended, flags)/np.sum(prices_recommended)
    
    return precision

money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5)

#3 - recall_at_k
def recall_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    
    flags = np.isin(bought_list, recommended_list)
    
    recall = flags.sum() / len(bought_list)
    
    return recall

recall_at_k(recommended_list, bought_list, k=5)


#4. Реализовать money_recall_at_k
def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    #принцип - деление суммы рекомендованных товаров среди релевантных (купленных) к общей сумме покупки
    
    # your_code
    bought_list = np.array(bought_list)
    prices_bought = np.array(prices_bought)
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    
    #вычисляем сумму рекомендованных товаров среди купленных (числитель)
    flags = np.isin(bought_list, recommended_list)    
    recall = np.dot(prices_bought, flags)/np.sum(prices_bought)
    
    return recall

prices_bought = [90, 110, 150, 95]
money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5)


#5. Реализовать map@k
def precision_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    #assert len(bought_list) > len(recommended_list)
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = flags.sum() / len(recommended_list)
    
    
    return precision


def ap_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    
    if sum(flags) == 0:
        return 0
    
    sum_ = 0
    for i in range(0, k-1):
        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i+1)
            sum_ += p_k
            
    result = sum_ / sum(flags)
    
    return result

ap_k(recommended_list, bought_list, k=5)


def map_k(recommended_list, bought_list, k=5, u=1):
    
    # your_code
    sum_ = 0
    for i in range(u):
        sum_ += ap_k(recommended_list[i], bought_list[i], k = k)
    
    return sum_/u

recommended_list = [[143, 156, 1134, 991, 27, 1543, 3345, 533, 11, 43],
                    [146, 156, 1134, 991, 27, 1543, 3345, 533, 11, 43],] #id товаров
bought_list = [[521, 32, 143, 991], [146, 29]]

map_k(recommended_list, bought_list, k=5, u=2)