from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np

class StoredData(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

inputs = np.arange(10)
targets = inputs

transition_dataset = StoredData(inputs, targets)
sampler = RandomSampler(transition_dataset, replacement=True)
loader = DataLoader(transition_dataset, batch_size=1, sampler=sampler)
for i in range(3):
    for k, (x, target) in enumerate(loader):
        print(x,target)
    print("-----")
print(k)
