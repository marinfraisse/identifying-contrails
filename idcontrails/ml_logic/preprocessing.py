import os
import numpy as np
from idcontrails.params import *
import matplotlib.pyplot as plt

def create_list_samples_with_contrails():

    target_suffix = 'human_pixel_masks.npy'

    # Building list of record_ids
    record_ids = os.listdir(DATASET_SAMPLE_PATH)
    len_dataset = len(record_ids)
    print(f'The sample contains {len_dataset} observations.')


    # Keeping only observations that contains contrails
    contrail_record_ids = []
    len(contrail_record_ids)
    for record_id in record_ids:
        # Building target paths
        target_path = os.path.join(DATASET_SAMPLE_PATH, record_id, target_suffix)
        target = np.load(open(target_path, 'rb'))

        # Jumping over observations with no contrails
        if target.sum()==0:
            continue
        else:
            contrail_record_ids.append(record_id)
    print('-')
    print('-')
    print('-')
    print(f'The sample dataset contains {len(contrail_record_ids)} observations with contrails in them.')
    return contrail_record_ids

contrail_record_ids = create_list_samples_with_contrails()

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


def loading_single_array(index=0) :
    dataset_sample_path = DATASET_SAMPLE_PATH
    record_list = os.listdir(dataset_sample_path)
    record_id = record_list[index]

    # loading 3 bands required for normalization and the mask
    with open(os.path.join(dataset_sample_path, record_id, 'band_11.npy'), 'rb') as f:
        band11 = np.load(f)
    with open(os.path.join(dataset_sample_path, record_id, 'band_14.npy'), 'rb') as f:
        band14 = np.load(f)
    with open(os.path.join(dataset_sample_path, record_id, 'band_15.npy'), 'rb') as f:
        band15 = np.load(f)
    with open(os.path.join(dataset_sample_path, record_id, 'human_pixel_masks.npy'), 'rb') as f:
        output_mask = np.load(f)

    # normalizing the selected image to plot it in RGB ash
    _T11_BOUNDS = (243, 303)
    _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
    _TDIFF_BOUNDS = (-4, 2)

    r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band14, _T11_BOUNDS)
    return np.clip(np.stack([r, g, b], axis=2), 0, 1)[...,4]
