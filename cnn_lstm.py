#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : cnn_lstm.py
@Author: XuYaoJian
@Date  : 2024/04/15 22:19
@Desc  : 
"""

import os
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data import get_data

from metrics import MSE, MAE

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
torch.cuda.set_device(device)
print(device)

def seed_everything(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class TrainDataset(Dataset):
    def __init__(self, features, targets, settings):
        super(TrainDataset, self).__init__()
        self.features = features
        self.targets = targets
        self.input_len = settings['input_len']
        self.output_len = settings['output_len']

    def __len__(self):
        return len(self.features) - self.output_len - self.input_len + 1

    def __getitem__(self, index):
        output_begin = index + self.input_len
        output_end = index + self.input_len + self.output_len
        return self.features[index: output_begin].astype('float32'), \
               self.targets[output_begin: output_end].reshape(-1).astype('float32')

class CNN_LSTM_Model(nn.Module):
    def __init__(self, settings):
        super(CNN_LSTM_Model, self).__init__()
        self.output_len = settings["output_len"]
        self.input_size = settings["in_var"]
        self.hidR = 20
        self.num_layers = settings["rnn_layer"]

        self.conv = nn.Sequential( # in_channels的概念相当于自然语言处理中的embedding，因此输入通道数为input_size表示输入的特征数
            nn.Conv1d(in_channels=self.input_size, out_channels= 14, kernel_size=3), # 一维卷积中卷积操作是针对input_len维度进行的
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )

        self.dropout = nn.Dropout(settings["dropout"])
        self.rnn = nn.LSTM(input_size= 14, hidden_size=self.hidR, num_layers=self.num_layers, batch_first=True)
        self.projection = nn.Linear(self.hidR, self.output_len)

    def forward(self, x_enc): # forward也有三种方式，可参考lstm.py forward
        '''
        :param x_enc: [batch, input_len , num_features]
        :return:
        '''
        x_enc = x_enc.permute(0, 2, 1) # 交换后的输入数据将符合CNN的输入
        x_enc = self.conv(x_enc)
        x_enc = x_enc.permute(0, 2, 1)
        rnn_out, _ = self.rnn(x_enc)
        x = self.projection(rnn_out)
        predict = x[:, -1, :]
        return predict


def forecast_gru(settings):
    model_filenames = ['cnn_lstm_o_15.pt', 'cnn_lstm_o_45.pt', 'cnn_lstm_o_90.pt', 'cnn_lstm_o_150.pt']
    path = settings['checkpoints']
    predictions = []

    for index, model_name in enumerate(model_filenames):
        name = path + model_name
        model = torch.load(name)  # 读取整个模型
        model.eval()
        train_features, _ = get_data(settings, settings['path'])
        seq_len = settings['input_len']
        data = np.array(train_features[-seq_len:]).astype('float32')
        data = torch.unsqueeze(torch.from_numpy(data), 0).to(device)
        with torch.no_grad():
            outputs = model(data)
        pred = outputs.detach().cpu().numpy()
        pred = np.clip(np.array(pred), a_min=0., a_max=1.).reshape(-1)
        for i in range(len(pred)):
            if (i + 1) % 15 == 0 or i % 15 == 0:
                pred[i] = 0.0 # 根据数据观察规律，6点和20点的太阳能产量为0，把预测值强制设为0
        predictions.append(pred)
    return predictions



def prep_env():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        'checkpoints': "./checkpoints/cnn_lstm/",
        'path': "./data/solar_enery.csv",
        'remove_features': ['Day', 'Dir','hour_cos'],
        'rnn_layer': 1,
        "input_len": 50,
        "output_len": 150,
        'batch_size': 32,
        'in_var': 8,
        "dropout": 0.05,
        'epoch_num': 200,
        'learning_rate': 0.001,
        "horizons": [15, 45, 90, 150],
    }
    return settings

if __name__ == "__main__":
    seed_everything(2024)
    settings = prep_env()

    # train phase
    # for horizon in settings['horizons']:
    #     settings["output_len"] = horizon

    #     train_features, train_targets = get_data(settings, "./data/solar_enery.csv")
    #     train_dataset = TrainDataset(train_features, train_targets, settings)
    #     train_dataloader = DataLoader(train_dataset, batch_size=settings['batch_size'], shuffle=True)

    #     model = CNN_LSTM_Model(settings).to(device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    #     criterion = nn.MSELoss(reduction='mean')
    #     steps_per_epoch = len(train_dataloader)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99, verbose=True)

    #     model.train()
    #     for epoch in range(settings['epoch_num']):
    #         scheduler.step()
    #         train_loss = 0
    #         t = time.time()
    #         for step, batch in enumerate(train_dataloader):
    #             features, targets = batch
    #             features = features.to(device)

    #             targets = targets[:, -settings['output_len']:].to(device)

    #             optimizer.zero_grad()
    #             output = model(features)

    #             loss = criterion(output, targets)
    #             loss.backward()

    #             optimizer.step()
    #             train_loss += loss.item()

    #         print("horizons:{} epoch {}, Loss: {:.3f} Time: {:.1f}s, lr: {:.8f}"
    #               .format(horizon, epoch + 1, train_loss / steps_per_epoch, time.time() - t, scheduler.get_lr()[0]))
    #     torch.save(model, settings['checkpoints'] + f"cnn_lstm_o_{horizon}.pt")
    #     torch.cuda.empty_cache()

    
    # forecast future solar energy
    gru_forecast_results = forecast_gru(settings)

    solar_energy_true = pd.read_csv("./data/test.csv")['Radiance'].values[-150:]

    for i in range(len(gru_forecast_results)):
        
        forecast = gru_forecast_results[i]
        mse = MSE(forecast, solar_energy_true[:len(forecast)])
        mae = MAE(forecast, solar_energy_true[:len(forecast)])

        # 清除之前的图形
        plt.clf()
        plt.plot(forecast, label='Predicted Values', marker='o')
        plt.plot(solar_energy_true[:len(forecast)], label='True Values', marker='x')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title('Predicted vs True Values')
        plt.legend()
        plt.savefig('./results/cnn_lstm/predicted_vs_true_values' + str(i) + '.png')
        print(f"mse:{mse}, mae:{mae}")

        predictions = pd.DataFrame({
            'prediction': forecast,
            'groundTruth': solar_energy_true[:len(forecast)]
        })
        L = settings['horizons'][i]
        predictions.to_csv(f"./results/cnn_lstm/prediction_length_{L}.csv", index=False)


        


