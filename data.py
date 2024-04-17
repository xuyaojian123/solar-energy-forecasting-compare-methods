#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : feature_engineering1.py
@Author: XuYaoJian
@Date  : 2024/04/15 15:06
@Desc  : 
"""
import math

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import RobustScaler, StandardScaler

from time_feature import time_features

warnings.simplefilter('ignore')


def fill_data(df):
    # 1、Dir缺失值使用后值填充 161个缺失值
    # 2、Spd、Temp缺失值使用线性插补 161个缺失值，8个缺失值
    print(df.isnull().sum())
    df['Dir'].bfill(inplace=True)
    df['Dir'] = (df['Dir'] - df['Dir'].min()) / (df['Dir'].max() - df['Dir'].min())  # 风向归一化
    df['Spd'].interpolate(inplace=True)
    df['Temp'].interpolate(inplace=True)
    if df.isnull().any().any():
        print("数据有NAN值")
    else:
        return df


def get_data(settings, path):
    print("Loading train data")
    df = pd.read_csv(path)
    print('Adding features')

    df['date'] = pd.to_datetime(df['date'])
    data_stamp = time_features(pd.to_datetime(df['date'].values), freq='1h') # 创建有关时间衍生特征
    data_stamp = data_stamp.transpose(1,0)
    df = pd.concat([df, pd.DataFrame(data_stamp)], axis=1)
    df = df.drop(columns=['date'])
    
    print("data cols", df.columns)
    train_features = df.copy()
    train_targets = df['Radiance']
    return np.array(train_features), np.array(train_targets)


