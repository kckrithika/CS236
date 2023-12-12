import os

import torch
from torch import nn
from torch import optim
from models.conditional_flow.data_loader import create_train_dataset, tokenizer
from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.base import CompositeTransform
from models.conditional_flow.model import MaskedUnconstrainedPiecewiseCubicAutoregressiveTransform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.Sequential(
          nn.Linear(4, 512),
          nn.ReLU(),
          nn.Linear(512, 512),
          nn.ReLU(),
          nn.Linear(512, 2)
        )

num_layers = 5
base_dist = ConditionalDiagonalNormal(shape=[1],context_encoder=model)

transforms = []

for _ in range(num_layers):
    transforms.append(ReversePermutation(features=1000))
    transforms.append(MaskedUnconstrainedPiecewiseCubicAutoregressiveTransform(features=1000, hidden_features=256,
                                                                               context_features=1000,
                                                                               num_bins=10))

transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist).to(device)

optimizer = optim.Adam(flow.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,600], gamma=0.3)

x_train, cond_train = create_train_dataset()
num_iter = 50
os.makedirs('models', exist_ok=True)
for i in range(num_iter):
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x_train.to(device), context=cond_train.to(device)).mean()
    if i%10 == 0:
      print('iteration',i,':',loss.item())
      torch.save(flow.state_dict(), f'models/model_{i}.pth')
    loss.backward()
    optimizer.step()
    scheduler.step()
