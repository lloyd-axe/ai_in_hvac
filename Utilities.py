import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

SERVER_LOAD = 'server_load'
OUTSIDE_TEMPERATURE = 'outside_temperature'
HUMIDITY = 'humidity'
HOUR = 'hour'
DAYOFWEEK = 'dayofweek'
ROOM_TEMPERATURE = 'room_temperature'


class SimulatedRoomConditionsDataset(Dataset):
    input_features = [SERVER_LOAD, OUTSIDE_TEMPERATURE, HUMIDITY, HOUR, DAYOFWEEK]
    output_feature = ROOM_TEMPERATURE

    def __init__(self, sample_count: int):
        self.sample_count = sample_count
        self.timestamps = pd.date_range(start='2023-01-01', periods=self.sample_count, freq='H')
        self.data = self.__generate_data()
        self.input_data = self.data[self.input_features].values
        self.input_data = self.scale(data=self.input_data)
        self.output_data = self.data[self.output_feature].values

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        input_data = torch.tensor(self.input_data[idx], dtype=torch.float32)
        output_data = torch.tensor(self.output_data[idx], dtype=torch.float32)
        return input_data, output_data

    def scale(self, data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def __generate_data(self) -> pd.DataFrame:
        data = pd.DataFrame({
            'timestamp': self.timestamps,
            SERVER_LOAD: np.random.uniform(20, 90, size=self.sample_count),
            OUTSIDE_TEMPERATURE: np.random.uniform(15, 35, size=self.sample_count),
            HUMIDITY: np.random.uniform(30, 80, size=self.sample_count),
        })
        data[HOUR] = data['timestamp'].dt.hour
        data[DAYOFWEEK] = data['timestamp'].dt.dayofweek
        return self.__simulate_room_temp(data)

    def __simulate_room_temp(self, data: pd.DataFrame) -> pd.DataFrame:
        noise = np.random.normal(0, 1, self.sample_count)
        data[ROOM_TEMPERATURE] = (0.5 * data[SERVER_LOAD] +  # assumed effect of each feature to the temp
                                  0.3 * data[OUTSIDE_TEMPERATURE] +
                                  0.1 * data[HUMIDITY] +
                                  0.05 * data[HOUR] -
                                  0.2 * data[DAYOFWEEK] + noise
                                  ) / 10 + 18  # bring value to range 18-30 C only
        return data


class ScikitDataEval:
    @staticmethod
    def evaluate_regression(y_pred: np.ndarray, y_test: np.ndarray, threshold: float = 0.1) \
            -> (float, float, float):
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        score = 0
        for y_t, y_p in zip(y_test, y_pred):
            diff = abs(y_t - y_p)
            if diff >= threshold:
                score += 1
        print(f'SCORE: {score} / {len(y_test)} | Accuracy: {(score/len(y_test) * 100):0.2f} %')
        print(f'Mean Absolute Error: {mae:.2f}')
        print(f'Mean Squared Error: {rmse:.2f}')
        print(f'R²: {r2:.2f}')
        return score, mae, rmse, r2
