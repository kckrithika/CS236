import torch
import torch.nn as nn
D_MODEL = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, DE_VOCAB_SIZE, EN_VOCAB_SIZE):
        super(Net, self).__init__()
        self.DE_VOCAB_SIZE = DE_VOCAB_SIZE
        self.EN_VOCAB_SIZE = EN_VOCAB_SIZE
        self.de_embed = nn.Embedding(self.DE_VOCAB_SIZE, D_MODEL)
        self.en_embed = nn.Embedding(self.EN_VOCAB_SIZE, D_MODEL)
        self.transformer = nn.Transformer(d_model=D_MODEL,
                                          num_encoder_layers=2, num_decoder_layers=2,
                                          dropout=0.5, dim_feedforward=2048)
        self.fc1 = nn.Linear(D_MODEL, EN_VOCAB_SIZE)

    def forward(self, inputs, targets):
        x = self.de_embed(inputs)
        y = self.en_embed(targets)
        tgt_mask = torch.triu(torch.ones(targets.size(0), targets.size(0)), diagonal=1).bool().to(device)
        out = self.transformer(x, y, tgt_mask=tgt_mask)
        out = self.fc1(out.permute(1, 0, 2))  # (batch, sequence, feature)
        return out.permute(1, 0, 2).reshape(-1, self.EN_VOCAB_SIZE)  # (sequence, batch, feature)
