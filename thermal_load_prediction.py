import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Utilities import SimulatedRoomConditionsDataset, ScikitDataEval


class GradientBoostedRegressionTree:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=self.n_estimators,
                                      random_state=self.random_state)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) \
            -> tuple:
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        score, mae, rmse, r2 = ScikitDataEval.evaluate_regression(y_pred=y_pred, y_test=y_test)
        return score, mae, rmse, r2

    def inference(self, x):
        y_pred = self.model.predict(x.reshape(1, -1))
        print(f'Predicted room temperature = {y_pred[0]:.2f} °C')


class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size: int):
        super(NeuralNetworkModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


class NeuralNetworkModelPipeline:
    def __init__(self, input_size: int, model_save_path: str = 'best_model.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size

        self.model = NeuralNetworkModel(input_size=self.input_size).to(self.device)
        self.model_save_path = model_save_path

    def run(self, data, optimizer: torch.optim.Optimizer, criterion: nn.Module, test_size: int, batch_size: int = 24,
            epochs: int = 100, shuffle: bool = True, patience: int = 10):
        train_dataset, test_dataset = torch.utils.data.random_split(data, [int((1-test_size) * len(data)),
                                                                           int(test_size * len(data))])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        best_model = self.train(train_dataloader, optimizer, criterion, epochs, patience)
        self.test(best_model, test_dataloader)

    def train(self, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module,
              epochs: int = 100, patience: int = 10):
        best_loss = float('inf')
        epochs_without_improvement = 0
        best_model = None
        for epoch in range(epochs):
            epoch_loss, optimizer = self._train_epoch(train_dataloader, epoch, optimizer, criterion)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.model_save_path)
                best_model = self.model
                print(f'Saved best model | loss= {best_loss} to: {self.model_save_path}')
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                break
        return best_model or self.model

    def _train_epoch(self, data_loader: DataLoader, epoch: int, optimizer: torch.optim.Optimizer,
                     criterion: nn.Module) -> (float, torch.optim.Optimizer):
        running_loss = 0.0
        self.model.train()
        with tqdm(data_loader, desc=f'Epoch {epoch + 1}...', unit='batch') as tbar:
            for input_data, output_data in tbar:
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)
                predictions = self.model(input_data)

                loss = criterion(predictions, output_data)
                running_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        avg_loss = running_loss / len(data_loader)
        return avg_loss, optimizer

    def test(self, model, test_dataloader: DataLoader) -> tuple:
        model.eval()
        y_preds = []
        y_true = []
        with torch.no_grad():
            with tqdm(test_dataloader, desc=f'Testing ...', unit='batch') as tbar:
                for input_data, output_data in tbar:
                    input_data = input_data.to(self.device)
                    output_data = output_data.to(self.device)
                    output = model(input_data)
                    y_preds.append(output)
                    y_true.append(output_data)

        score, mae, rmse, r2 = ScikitDataEval.evaluate_regression(y_pred=torch.cat(y_preds).cpu().numpy(),
                                                                  y_test=torch.cat(y_true).cpu().numpy())
        return score, mae, rmse, r2


if __name__ == '__main__':
    batch_size = 42
    shuffle = True
    test_size = 0.2
    epochs = 100
    learning_rate = 0.001

    # For pytorch Models
    simulated_room_data = SimulatedRoomConditionsDataset(sample_count=1000)
    x_train, x_test, y_train, y_test = train_test_split(simulated_room_data.input_data,
                                                        simulated_room_data.output_data,
                                                        test_size=test_size,
                                                        shuffle=shuffle)

    print('USING NEURAL NEWORK MODEL')
    model1 = NeuralNetworkModelPipeline(input_size=len(simulated_room_data.input_features))
    optimizer = optim.Adam(model1.model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model1.run(simulated_room_data, optimizer, criterion, test_size, batch_size, epochs)

    print('USING XGBOOST REGRESSION TREES')
    model2 = GradientBoostedRegressionTree()
    model2_res = model2.fit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)