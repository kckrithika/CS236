import os

from tqdm import tqdm
import torch
from data_loader import tokenizer, get_training_dataloader
from model import Net
from torch import nn, optim

NUM_EPOCHS = 20000


def train_network(epoch_range, train_iter):
    net.train()
    for epoch in epoch_range:
        losses = 0.
        with tqdm(total=len(train_iter)) as pbar:
            for i, (inputs, targets) in enumerate(train_iter):
                optimizer.zero_grad()
                pred = net(inputs.to(device), targets[:-1, ].to(device))
                loss = criterion(pred.to('cpu'), targets[1:, ].view(-1))
                loss.backward()
                optimizer.step()

                losses += loss.detach().item()
                pbar.set_description(f'training loss: {losses / (i + 1):.4f}')
                pbar.update(1)
        if epoch % 10 == 0:
            torch.save(net.state_dict(), f'models_100/model_{epoch}.pth')
        print(f'Epoch {epoch:2}, train loss: {(losses / (i + 1)):.6f}')


if __name__ == '__main__':
    os.makedirs('models_100', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader = get_training_dataloader()
    x_vocab = y_vocab = len(tokenizer.vocab)
    net = Net(x_vocab, y_vocab).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer['PAD_None'])
    optimizer = optim.Adam(net.parameters())
    train_network(range(1, NUM_EPOCHS + 1), train_dataloader)
