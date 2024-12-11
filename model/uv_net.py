import torch
import torch.nn as nn
import pickle
import yaml

NUM_MATRIX = 8 

class UVNet(nn.Module):
    def __init__(self, config_file):
        super(UVNet, self).__init__()

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        matrices = torch.tensor([config['matrix_{:02d}'.format(i)] for i in range(1, NUM_MATRIX + 1)], dtype=torch.float32).unsqueeze(0)
        self.matrices = nn.Parameter(matrices)
        
    def forward(self, x1, x2):
        index1 = x1.type(torch.int32).unsqueeze(1)
        index2 = x2.type(torch.int32).unsqueeze(1)
        result = self.matrices[:, index1, index2]
        return result


if __name__ == "__main__":

    model = UVNet("config/uv_matrix.yaml")

    batch_size = 4
    x1_batch = torch.tensor([[5.0]*batch_size]).T
    x2_batch = torch.tensor([[8.0]*batch_size]).T

    print("x1_batch{} --- {}".format(x1_batch.shape, x1_batch))
    print("x2_batch{} --- {}".format(x2_batch.shape, x2_batch))
    result_batch = model(x1_batch, x2_batch)
    print(result_batch)
