#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : metrics.py
@Author: XuYaoJian
@Date  : 2024/4/16 16:22
@Desc  : 
"""
import numpy as np

def MSE(prediction, groundTruth):
    """
        计算均方误差(MSE)指标
        参数：
        y_true: 真实值数组
        y_pred: 预测值数组
        返回：
        mse: 均方误差
        """
    assert len(prediction) == len(groundTruth), "输入数组长度不一致"
    y_pred = np.array(prediction)
    y_true = np.array(groundTruth)
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


def MAE(prediction, groundTruth):
    """
       计算平均绝对误差(MAE)指标
       参数：
       y_true: 真实值数组
       y_pred: 预测值数组
       返回：
       mae: 平均绝对误差
       """
    assert len(prediction) == len(groundTruth), "输入数组长度不一致"
    y_pred = np.array(prediction)
    y_true = np.array(groundTruth)
    mae = np.mean(np.abs(y_true - y_pred))
    return mae