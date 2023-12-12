import copy
import glob
import os
import time
import tqdm
import sys
from multiprocessing import Pool

import pretty_midi

OUTPUT_DIR = '../processed_data/one_minute_chunks_filtered'


def filter_entity(start_time, end_time, entity):
    new_entity = []
    for item in entity:
        if hasattr(item, 'start'):
            if start_time <= item.start < end_time or start_time < item.end <= end_time:
                item.start = max(item.start - start_time, 0)
                item.end = min(item.end - start_time, end_time - start_time)
                new_entity.append(item)
        elif hasattr(item, 'time'):
            if start_time <= item.time < end_time:
                item.time = item.time - start_time
                new_entity.append(item)
    return new_entity


def get_midis_per_drum(obj):
    drum_instruments, other_instruments = split_drum_and_other_instruments(obj)
    drum_midis = []
    for instrument in drum_instruments:
        instrument.is_drum = True
        drum_midi = pretty_midi.PrettyMIDI()
        for other_instrument in other_instruments:
            drum_midi.instruments.append(copy.deepcopy(other_instrument))
        drum_midi.instruments.append(instrument)
        drum_midis.append(drum_midi)
    return drum_midis


def check_if_instrument_is_predominant(obj, instrument, threshold):
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    if not sorted_notes:
        return False
    gaps = sorted_notes[0].start
    for i in range(1, len(sorted_notes)):
        gaps += sorted_notes[i].start - sorted_notes[i-1].end
    if gaps > (threshold * obj.get_end_time()):
        return False
    return True


def split_drum_and_other_instruments(obj):
    return [instrument for instrument in obj.instruments if instrument.is_drum], \
           [instrument for instrument in obj.instruments if not instrument.is_drum]
    # if type == 'clean_midi':
    #     drum_instruments = [instrument for instrument in obj.instruments if instrument.program == 0 or instrument.is_drum
    #                         == True]
    #     other_instruments = [instrument for instrument in obj.instruments if instrument.program != 0 and
    #                          instrument.is_drum == False]
    # elif type == 'la_midi':
    #     drum_instruments = [instrument for instrument in obj.instruments if instrument.is_drum == True]
    #     other_instruments = [instrument for instrument in obj.instruments if instrument.is_drum == False]
    # else:
    #     drum_instruments = []
    #     other_instruments = []
    # return drum_instruments, other_instruments


def verify_drum_presence_and_remove_sporadic_instruments(obj):
    drum_instruments, other_instruments = split_drum_and_other_instruments(obj)
    assert(len(drum_instruments) == 1)
    drum_instrument = drum_instruments[0]
    if not check_if_instrument_is_predominant(obj, drum_instrument, 0.25):
        return None

    new_obj = pretty_midi.PrettyMIDI()
    new_obj.instruments.append(copy.deepcopy(drum_instrument))
    for instrument in other_instruments:
        if check_if_instrument_is_predominant(obj, instrument, 0.5):
            new_obj.instruments.append(copy.deepcopy(instrument))
    if len(new_obj.instruments) > 1:
        return new_obj
    return None


def get_chunks(obj, time_in_minutes):
    chunk_duration = time_in_minutes * 60
    total_time = obj.get_end_time()
    num_chunks = int(total_time / chunk_duration) + 1

    chunks = []
    # Split the MIDI obj into chunks
    for i in range(num_chunks):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, total_time)

        # Create a new PrettyMIDI object for the chunk
        chunk_midi = pretty_midi.PrettyMIDI()

        # Copy relevant notes, instruments, and other events to the chunk
        for instrument in obj.instruments:
            new_instrument = copy.deepcopy(instrument)

            new_instrument.notes = filter_entity(start_time, end_time, instrument.notes)
            new_instrument.control_changes = filter_entity(start_time, end_time, instrument.control_changes)
            new_instrument.pitch_bends = filter_entity(start_time, end_time, instrument.pitch_bends)
            chunk_midi.instruments.append(new_instrument)
        chunks.append(chunk_midi)
    return chunks


def get_midi_obj(file):
    return pretty_midi.PrettyMIDI(file)


def do_process_midi_file(file):
    filename = os.path.basename(file)[:-4]
    midi_obj = get_midi_obj(file)
    drum_instruments, _ = split_drum_and_other_instruments(midi_obj)
    if len(drum_instruments) != 1:
        print('More than 1 drum instruments used.', file=sys.stdout)
        return
    midi_chunks = get_chunks(midi_obj, time_in_minutes=0.5)
    chunk_count = 0
    for chunk in midi_chunks:
        per_drum_chunks = get_midis_per_drum(chunk)
        if len(per_drum_chunks) > 1:
            continue
        drum_count = 0
        for drum_chunk in per_drum_chunks:
            processed_chunk = verify_drum_presence_and_remove_sporadic_instruments(drum_chunk)
            if processed_chunk:
                processed_chunk.write(f'{OUTPUT_DIR}/{filename}_{chunk_count}_{drum_count}.mid')
            drum_count += 1
        chunk_count += 1


def process_midi_file(file):
    try:
        existing_files = glob.glob(f'{OUTPUT_DIR}/{os.path.basename(file)[:-4]}_*.mid')
        if len(existing_files) > 0:
            return
        print(f'Started processing {os.path.basename(file)}')
        start = time.time()
        do_process_midi_file(file)
        print(f'Finished processing {os.path.basename(file)}. Time taken: {time.time() - start} sec')
    except Exception as exc:
        print(exc)
        return


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    midi_files = sorted(glob.glob('../raw_data/all_midis_raw/*.mid'))
    with Pool(processes=400) as p:
        tasks = [p.apply_async(process_midi_file, (file,)) for file in midi_files]
        for task in tasks:
            task.get()
