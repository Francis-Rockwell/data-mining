import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = [2 * self.input_size, self.input_size]

        self.fc1 = nn.Linear(self.input_size, self.hidden_size[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size[0], self.output_size)
        self.fc2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size[1], self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
