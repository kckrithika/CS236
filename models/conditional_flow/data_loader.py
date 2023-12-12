import os.path
import torch
from tqdm import tqdm
import time
from torch.nn.utils.rnn import pad_sequence
from miditok import REMI, TokenizerConfig
from torch.utils.data import DataLoader, Dataset
from miditoolkit import MidiFile
import glob


def create_tokenizer():
    PITCH_RANGE = (21, 109)
    BEAT_RES = {(0, 1): 8, (1, 2): 4, (2, 4): 2, (4, 8): 1}
    NB_VELOCITIES = 24
    SPECIAL_TOKENS = ["PAD", "MASK", "BOS", "EOS"]
    USE_CHORDS = False
    USE_RESTS = True
    USE_TEMPOS = True
    USE_TIME_SIGNATURE = False
    USE_PROGRAMS = True
    NB_TEMPOS = 32
    TEMPO_RANGE = (50, 200)  # (min_tempo, max_tempo)
    TOKENIZER_PARAMS = {
        "pitch_range": PITCH_RANGE,
        "beat_res": BEAT_RES,
        "nb_velocities": NB_VELOCITIES,
        "special_tokens": SPECIAL_TOKENS,
        "use_chords": USE_CHORDS,
        "use_rests": USE_RESTS,
        "use_tempos": USE_TEMPOS,
        "use_time_signatures": USE_TIME_SIGNATURE,
        "use_programs": USE_PROGRAMS,
        "nb_tempos": NB_TEMPOS,
        "tempo_range": TEMPO_RANGE,
    }
    config = TokenizerConfig(**TOKENIZER_PARAMS)
    return REMI(config)


tokenizer = create_tokenizer()


class CustomDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.__iter_count = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        return item

    def __iter__(self):
        return self

    def __next__(self):
        if self.__iter_count >= len(self):
            self.__iter_count = 0
            raise StopIteration
        else:
            self.__iter_count += 1
            return self[self.__iter_count - 1]

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return "No data loaded" if len(self) == 0 else f"{len(self.samples)} samples"


def get_tokens(x_path, y_path):
    start = time.time()
    x_tokens = tokenizer.midi_to_tokens(MidiFile(x_path)).ids
    x_tokens = x_tokens + [tokenizer['PAD_None']] * (1000 - len(x_tokens))
    y_tokens = tokenizer.midi_to_tokens(MidiFile(y_path)).ids
    y_tokens = y_tokens + [tokenizer['PAD_None']] * (1000 - len(y_tokens))
    print(f'Completed tokenization in {time.time()-start} seconds')
    return x_tokens, y_tokens


def create_train_dataset():
    print('Creating training dataset')
    train_x_paths = sorted(glob.glob('../../processed_data/train/input/*.mid'))
    x_train = []
    cond_train = []
    for x_path in tqdm(train_x_paths[:10]):
        y_path = f'../../processed_data/train/target/{os.path.basename(x_path)}'
        if not os.path.exists(y_path):
            continue
        try:
            x_tokens, y_tokens = get_tokens(x_path, y_path)
            x_train.append(y_tokens[:1000])
            cond_train.append(x_tokens[:1000])
        except Exception as exc:
            print(exc)

    print('Created Train dataset')
    return torch.tensor(x_train), torch.tensor(cond_train)


def create_test_dataset():
    print('Creating test dataset')
    test_x_paths = sorted(glob.glob('../../processed_data/train/input/*.mid'))
    test_y_paths = sorted(glob.glob('../../processed_data/train/target/*.mid'))

    x_test = []
    cond_test = []
    for x_path, y_path in tqdm(zip(test_x_paths[100:110], test_y_paths[100:110])):
        if os.path.basename(x_path) != os.path.basename(y_path):
            print(f'Warning! Names do not match. {x_path}, {y_path}')
            continue
        x_tokens = tokenizer.midi_to_tokens(MidiFile(x_path)).ids
        y_tokens = tokenizer.midi_to_tokens(MidiFile(y_path)).ids
        x_test.append(y_tokens[:1000])
        cond_test.append(x_tokens[:1000])
    return torch.tensor(x_test), torch.tensor(cond_test)
