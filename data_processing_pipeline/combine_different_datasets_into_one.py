import glob
import shutil
import os

os.makedirs('../raw_data/all_midis_raw', exist_ok=True)
count = 0
for file in glob.glob('../raw_data/clean_midi/**/*.mid'):
    count += 1
    filename = f'{count}.mid'
    shutil.copy(file, f'../raw_data/all_midis_raw/clean_midi_{filename}')

count = 0
for file in glob.glob('../raw_data/Los-Angeles-MIDI-Dataset-Ver-4-0/**/*.mid'):
    count += 1
    filename = f'{count}.mid'
    shutil.copy(file, f'../raw_data/all_midis_raw/la_midi_{filename}')

print(len(glob.glob('../raw_data/all_midis_raw/*.mid')))