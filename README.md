![](/plot.png)

### Requirements

```bash
conda create --name solar_compare python=3.9
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Run

Run ANN and forecasting

```
python ANN.py
```

Run LSTM and forecasting

```
python lstm.py
```

Run LSTM-CNN and forecasting

```
python cnn_lstm.py
```

Run Transformer and forecasting

```
python transformer.py
```

`data` folder contains training data and test data. `results` folder save the solar energy forecasting reuslts of the corresponding model.
