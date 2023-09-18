import os
import numpy as np
from idcontrails.params import *
import matplotlib.pyplot as plt


list_chuncks = []
chunck_names = []
for i in range(0, NB_CHUNCKS):
    chunck_name = f'chunck_{i}'
    chunk_i_records = contrail_record_ids[(i*CHUNCK_SIZE):((i+1)*CHUNCK_SIZE)]
    chunck_names.append(chunck_name)
    list_chuncks.append(chunk_i_records)
chuncks_record_dict = dict(zip(chunck_names, list_chuncks))

def normalize_range(data, bounds):
        return (data - bounds[0]) / (bounds[1] - bounds[0])
def load_normalize_X_chunck(chunck, BASE_DIR, band_choice, N_TIMES_BEFORE):
    X_chunck = []
    for record_id in chuncks_record_dict[chunck]:
        # Building band paths
        record_first_band_path = os.path.join(BASE_DIR, record_id, band_choice[0])
        record_second_band_path = os.path.join(BASE_DIR, record_id, band_choice[1])
        record_third_band_path = os.path.join(BASE_DIR, record_id, band_choice[2])
        # Loading each band
        first_band = np.load(open(record_first_band_path, 'rb'))[:, :, N_TIMES_BEFORE]
        second_band = np.load(open(record_second_band_path, 'rb'))[:, :, N_TIMES_BEFORE]
        third_band = np.load(open(record_third_band_path, 'rb'))[:, :, N_TIMES_BEFORE]
        # Normalizing each band with its relevant bounds
        _T11_BOUNDS = (243, 303)
        _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
        _TDIFF_BOUNDS = (-4, 2)
        # Applying normalization functions
        normalized_r = normalize_range(third_band - second_band, _TDIFF_BOUNDS)
        normalized_g = normalize_range(second_band - first_band, _CLOUD_TOP_TDIFF_BOUNDS)
        normalized_b = normalize_range(second_band, _T11_BOUNDS)
        # Building a single record from all bands
        record = np.clip(np.stack([normalized_r, normalized_g, normalized_b], axis=2), 0, 1)
        # Appending chunck list
        X_chunck.append(record)
    # Building the chunck array
    X_chunck_array = np.stack(X_chunck, axis=0)
    return X_chunck_array

# Function to load a y chunck
def load_y_chunck(chunck, BASE_DIR, target_suffix):
    y_chunck = []
    for record_id in chuncks_record_dict[chunck]:
        # Building target paths and loading data
        target_path = os.path.join(BASE_DIR, record_id, target_suffix)
        target = np.load(open(target_path, 'rb'))
        # Appending chunck list
        y_chunck.append(target)
    # Building the chunck array
    y_chunck_array = np.stack(y_chunck, axis=0).astype(float)
    return y_chunck_array

def plot_history(history, title='', axs=None, exp_name=""):
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history.history['loss'], label='train' + exp_name)
    ax1.plot(history.history['val_loss'], label='val' + exp_name)
    ax1.set_title('loss')
    ax1.legend()
    ax2.plot(history.history['dice_metric'], label='train dice metric' + exp_name)
    ax2.plot(history.history['val_dice_metric'], label='val dice metric' + exp_name)
    ax2.set_title('Dice metric')
    ax2.legend()
    return (ax1, ax2)
