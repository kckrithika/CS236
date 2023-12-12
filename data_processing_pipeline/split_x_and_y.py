import copy
import glob
import os
import time
import random
import multiprocessing

import pretty_midi
from split_into_2minute_chunks import split_drum_and_other_instruments

TEST_OUTPUT_DIR = '../processed_data/test'
TRAIN_OUTPUT_DIR = '../processed_data/train'


def get_midi_obj_from_instruments(instruments):
    new_midi = pretty_midi.PrettyMIDI()
    new_midi.instruments = copy.deepcopy(instruments)
    return new_midi


def get_x_and_y(file, is_test=False):
    filename = os.path.basename(file)
    if os.path.exists(f'{TRAIN_OUTPUT_DIR}/input/{filename}') and os.path.exists(f'{TRAIN_OUTPUT_DIR}/target/'
                                                                                 f'{filename}'):
        return
    start = time.time()
    print(f'Processing file {file}')
    midi_obj = pretty_midi.PrettyMIDI(file)
    drum_instruments, other_instruments = split_drum_and_other_instruments(midi_obj)
    y = get_midi_obj_from_instruments(other_instruments)
    x = get_midi_obj_from_instruments(drum_instruments)
    if is_test:
        y.write(f'{TEST_OUTPUT_DIR}/input/{filename}')
        x.write(f'{TEST_OUTPUT_DIR}/target/{filename}')
    else:
        y.write(f'{TRAIN_OUTPUT_DIR}/input/{filename}')
        x.write(f'{TRAIN_OUTPUT_DIR}/target/{filename}')
    print(f'Finished processing file {file} in {time.time() - start} seconds')


if __name__ == '__main__':
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    os.makedirs(f'{TEST_OUTPUT_DIR}/target', exist_ok=True)
    os.makedirs(f'{TRAIN_OUTPUT_DIR}/input', exist_ok=True)
    os.makedirs(f'{TRAIN_OUTPUT_DIR}/target', exist_ok=True)
    os.makedirs(f'{TEST_OUTPUT_DIR}/input', exist_ok=True)
    files = glob.glob('../processed_data/one_minute_chunks_filtered/*.mid')
    random.shuffle(files)
    test = files[:20]
    train = files[20:]
    with multiprocessing.Pool(processes=400) as p:
        tasks = [p.apply_async(get_x_and_y, (file,)) for file in train] + [p.apply_async(get_x_and_y, (file, True)) for file in test]
        for task in tasks:
            result = task.get()
