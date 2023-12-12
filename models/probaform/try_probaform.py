import torch
from models.wgan import ConditionalWGAN
from data_loader import create_train_dataset, create_test_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x, cond = create_train_dataset()
model = ConditionalWGAN().cuda()
model.fit(x.to(device), cond.to(device))
torch.save(model.state_dict(), 'wgan_model.pth')
