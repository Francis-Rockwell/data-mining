from .model import Model
from .neural_networks.mlp import MLP
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class NeuralNetwork(Model):
    def __init__(self, feature, label, split=0.2):
        super().__init__(feature, label, split)
        self.model = MLP(
            input_size=self.train_feature.shape[1],
            output_size=1,
        )
        self.train_loader = DataLoader(
            TensorDataset(
                torch.Tensor(self.train_feature),
                torch.Tensor(np.array(self.train_label)),
            ),
            batch_size=100,
            shuffle=True,
        )
        self.evaluate_loader = DataLoader(
            TensorDataset(
                torch.Tensor(self.evaluate_feature),
                torch.Tensor(np.array(self.evaluate_label)),
            ),
            batch_size=100,
            shuffle=True,
        )
        self.criterion = torch.nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        max_epoches = 100
        early_stop_patience = 5
        best_val_loss = float("inf")
        self.best_model = None

        for epoch in range(max_epoches):
            self.model.train()
            total_train_loss = 0.0
            for batch_data, batch_targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_targets.unsqueeze(1).float())
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(self.train_loader)

            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch_data, batch_targets in self.evaluate_loader:
                    outputs = self.model(batch_data)
                    loss = self.criterion(outputs, batch_targets.unsqueeze(1).float())
                    total_val_loss += loss.item()
                avg_val_loss = total_val_loss / len(self.evaluate_loader)

            print(
                f"Epoch [{epoch+1}/{max_epoches}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                current_patience = 0
                self.best_model = self.model.state_dict()
            else:
                current_patience += 1
                if current_patience >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    def predict(self, feature):
        self.model.load_state_dict(self.best_model)
        self.model.eval()
        return self.model(torch.Tensor(feature)).squeeze().detach()
