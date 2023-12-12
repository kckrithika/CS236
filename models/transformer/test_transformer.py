from model import Net, device
import torch
from data_loader import tokenizer, get_test_dataloader
import os

os.makedirs('predictions', exist_ok=True)
os.makedirs('targets', exist_ok=True)
x_vocab = y_vocab = len(tokenizer.vocab)
loaded_model = Net(x_vocab, y_vocab)
loaded_model.load_state_dict(torch.load('models_100/model_30.pth'))
loaded_model.to(device)
losses = 0.
scores = 0.
cnt = 0
loaded_model.eval()
test_iter = get_test_dataloader()
for i, (inputs, targets) in enumerate(test_iter):
    my_targets = targets[:1]
    while len(my_targets) < 1000:
        pred = loaded_model(inputs.to(device), my_targets.to(device))
        my_targets = torch.cat((
            my_targets.to(device),
            pred[-1,].argmax().unsqueeze(dim=0).unsqueeze(dim=0).to(device)
        ))

    target_midi = tokenizer.tokens_to_midi(targets.reshape(-1).tolist())
    pred_midi = tokenizer.tokens_to_midi(my_targets.reshape(-1).tolist())
    pred_midi.dump(f'predictions/{i}.mid')
    print(f'Test {i} complete.')