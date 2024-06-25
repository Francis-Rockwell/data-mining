from .model import Model
from .neural_networks.mlp import MLP
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import copy


class NeuralNetwork(Model):
    def __init__(
        self, train_feature, train_label, validation_feature, validation_label
    ):
        super().__init__(
            train_feature, train_label, validation_feature, validation_label
        )
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
        self.validation_loader = DataLoader(
            TensorDataset(
                torch.Tensor(self.validation_feature),
                torch.Tensor(np.array(self.validation_label)),
            ),
            batch_size=self.validation_feature.shape[0],
            shuffle=False,
        )
        self.criterion = torch.nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.path = "models/neural_networks/MLP.pth"

    def train(self):
        max_epoches = 1000
        early_stop_patience = 10
        best_val_auc = 0.0
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
            total_val_auc = 0.0
            with torch.no_grad():
                for batch_data, batch_targets in self.validation_loader:
                    outputs = self.model(batch_data)
                    loss = self.criterion(outputs, batch_targets.unsqueeze(1).float())
                    val_auc = roc_auc_score(
                        batch_targets.unsqueeze(1).float(), outputs.squeeze().detach()
                    )
                    total_val_loss += loss.item()
                    total_val_auc += val_auc
                avg_val_loss = total_val_loss / len(self.validation_loader)
                avg_val_auc = total_val_auc / len(self.validation_loader)

                print(
                    f"Epoch [{epoch+1}/{max_epoches}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {avg_val_auc:.4f}"
                )

                if avg_val_auc > best_val_auc:
                    best_val_auc = avg_val_auc
                    current_patience = 0
                    self.best_model = copy.deepcopy(self.model.state_dict())
                else:
                    current_patience += 1
                    if current_patience >= early_stop_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

        torch.save(self.best_model, self.path)

    def predict(self, feature):
        self.model.load_state_dict(self.best_model)
        # self.model.load_state_dict(torch.load(self.path))
        self.model.eval()
        return self.model(torch.Tensor(feature)).squeeze().detach()
