import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()

        self.input_layer = nn.Linear(input_dim, 20)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(20, 20) for _ in range(8)
        ])

        self.tanh = nn.Tanh()

        self.output_layer = nn.Linear(20, output_dim)

    def forward(self, x):
        x = self.tanh(self.input_layer(x))

        for layer in self.hidden_layers:
            x = self.tanh(layer(x))

        x = self.output_layer(x)

        phi = x[:, 0:1]
        p = x[:, 1:2]
        return (phi, p)
    