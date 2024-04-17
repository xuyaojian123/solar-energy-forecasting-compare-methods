
import math
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, settings):
        super(TransformerModel, self).__init__()
        self.d_model = 16
        self.input_size = settings["in_var"]
        self.output_len = settings["output_len"]
        self.input_len = settings['input_len']

        self.input_fc = nn.Linear(self.input_size, self.d_model)
        self.output_fc = nn.Linear(self.input_size, self.d_model)
        self.pos_emb = PositionalEncoding(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=4 * self.d_model,
            dropout=0.1,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=4,
            dropout=0.1,
            dim_feedforward=4 * self.d_model,
            batch_first=True,
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.fc = nn.Linear(self.output_len * self.d_model, self.output_len)
        self.fc1 = nn.Linear(self.input_len * self.d_model, self.d_model)
        self.fc2 = nn.Linear(self.d_model, self.output_len)


    def forward(self, x):
        '''
        :param x_enc: [batch, input_len , num_features]
        :return:
        '''
        x = self.input_fc(x)
        x = self.pos_emb(x)
        x = self.encoder(x)
        # 不经过解码器
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        out = self.fc2(x)

        return out


def forecast_transformer(settings):
    model_filenames = ['transformer_o_15.pt', 'transformer_o_45.pt', 'transformer_o_90.pt', 'transformer_o_150.pt']
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
        'checkpoints': "./checkpoints/lstm/",
        'path': "./data/solar_enery.csv",
        'remove_features': ['Day', 'Dir','hour_cos'],
        'rnn_layer': 1,
        "input_len": 50,
        "output_len": 150,
        'batch_size': 32,
        'in_var': 8,
        "dropout": 0.05,
        'epoch_num': 100,
        'learning_rate': 0.001,
        "horizons": [15, 45, 90, 150],
    }
    return settings

if __name__ == "__main__":
    seed_everything(2024)
    settings = prep_env()

    # train phase
    for horizon in settings['horizons']:
        settings["output_len"] = horizon

        train_features, train_targets = get_data(settings, "./data/solar_enery.csv")
        train_dataset = TrainDataset(train_features, train_targets, settings)
        train_dataloader = DataLoader(train_dataset, batch_size=settings['batch_size'], shuffle=True)

        model = TransformerModel(settings).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
        criterion = nn.MSELoss(reduction='mean')
        steps_per_epoch = len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99, verbose=True)

        model.train()
        for epoch in range(settings['epoch_num']):
            scheduler.step()
            train_loss = 0
            t = time.time()
            for step, batch in enumerate(train_dataloader):
                features, targets = batch
                features = features.to(device)

                targets = targets[:, -settings['output_len']:].to(device)

                optimizer.zero_grad()
                output = model(features)

                loss = criterion(output, targets)
                loss.backward()

                optimizer.step()
                train_loss += loss.item()

            print("horizons:{} epoch {}, Loss: {:.3f} Time: {:.1f}s, lr: {:.8f}"
                  .format(horizon, epoch + 1, train_loss / steps_per_epoch, time.time() - t, scheduler.get_lr()[0]))
        torch.save(model, settings['checkpoints'] + f"transformer_o_{horizon}.pt")
        torch.cuda.empty_cache()

    
    # forecast future solar energy
    gru_forecast_results = forecast_transformer(settings)

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
        plt.savefig('./results/transformer/predicted_vs_true_values' + str(i) + '.png')
        print(f"mse:{mse}, mae:{mae}")

        predictions = pd.DataFrame({
            'prediction': forecast,
            'groundTruth': solar_energy_true[:len(forecast)]
        })
        L = settings['horizons'][i]
        predictions.to_csv(f"./results/transformer/prediction_length_{L}.csv", index=False)


        


