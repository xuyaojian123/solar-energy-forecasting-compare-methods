#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : cnn_lstm.py
@Author: XuYaoJian
@Date  : 2024/04/16 21:19
@Desc  : 
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 15


true_solar_energy = pd.read_csv("results/ANN/prediction_length_150.csv")['groundTruth']

Ann = pd.read_csv("results/ANN/prediction_length_150.csv")['prediction']
cnn_lstm = pd.read_csv("results/cnn_lstm/prediction_length_150.csv")['prediction']
lstm = pd.read_csv("results/lstm/prediction_length_150.csv")['prediction']
transformer = pd.read_csv("results/transformer/prediction_length_150.csv")['prediction']
our_method = pd.read_csv("results/ours/prediction_length150.csv")['prediction']

t = np.arange(len(true_solar_energy))
fig, ax = plt.subplots(figsize=(11.2, 4.2), layout='constrained')
plt.ylim((0,1.01))
plt.xlim((0,150))
plt.yticks(np.linspace(0,1,11))
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)

ax.plot(t , lstm, lw=2,label = 'LSTM')
ax.plot(t , cnn_lstm, lw=2,label = 'CNN-LSTM')
ax.plot(t , transformer, lw=2,label = 'Transformer')
ax.plot(t , Ann,lw=2, label = 'ANN')
ax.plot(t , our_method, lw=2,label = 'Proposed')
ax.plot(t , true_solar_energy,lw=2, label = 'GroundTruth')


ax.legend(prop={'family' : 'Times New Roman', 'size'   : 12}, ncol = 6, bbox_to_anchor=(0.93,1.15))

#     ax.set_xlabel('prediction-Length-150 mse:'+str(mse)+' mae:' + str(mae),fontdict={'family' : 'Times New Roman', 'size'   : 15}) 
ax.set_xlabel("Forecasting Length",fontdict={'family' : 'Times New Roman', 'size'   : 16}) 
ax.set_ylabel('Solar Enery Production',fontdict={'family' : 'Times New Roman', 'size'   : 16}) 
ax.grid(True,alpha=0.4)


# plt.savefig('plot.svg', format='svg' ,bbox_inches='tight')  # 保存图形
plt.savefig('plot.eps', format='eps' ,bbox_inches='tight')  # 保存图形
# plt.savefig('plot.png',dpi=600,bbox_inches='tight')
plt.show()

