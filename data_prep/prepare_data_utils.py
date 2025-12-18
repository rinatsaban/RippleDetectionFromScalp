import os
import mne
mne.set_log_level('CRITICAL')
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

def load_raw_and_set_montage(patient, session, dataset_path, task_name='verbalWM'):
    session_path = os.path.join(dataset_path, patient, session)

    def get_raw_and_set_montage(scalp=True):
        folder_name = 'eeg' if scalp else 'ieeg'
        file_name = [f for f in os.listdir(os.path.join(session_path, folder_name)) if f.endswith('edf')][0]
        if scalp:
            fs = int(json.load(open(os.path.join(session_path, folder_name, f'{patient}_{session}_task-{task_name}_run-01_eeg.json')))['SamplingFrequency'])
            channels_data_path = os.path.join(session_path, folder_name,
                                             f'{patient}_{session}_task-{task_name}_run-01_channels.tsv')
            channels_data = pd.read_csv(channels_data_path, sep='\t')
            elec_names = channels_data['name'].tolist()
            montage = mne.channels.make_standard_montage('standard_1020')
            info = mne.create_info(elec_names, sfreq=fs, ch_types='eeg')
            raw = mne.io.read_raw_edf(os.path.join(session_path, folder_name, file_name), info, preload=True)
            elec_names = [elec for elec in elec_names if elec not in ['A1', 'A2']]
            raw.set_montage(montage, on_missing='ignore')
        else:
            fs = int(json.load(open(os.path.join(session_path, folder_name, f'{patient}_{session}_task-{task_name}_run-01_ieeg.json')))['SamplingFrequency'])
            montage_data_path = os.path.join(session_path, folder_name,
                                             f'{patient}_{session}_task-{task_name}_run-01_electrodes.tsv')
            montage_data = pd.read_csv(montage_data_path, sep='\t')
            elec_names = get_hippo_elec_names(montage_data)
            montage = mne.channels.make_dig_montage(
                ch_pos=dict(zip(montage_data['name'], montage_data[['x', 'y', 'z']].values)),
                )
            info = mne.create_info(montage_data['name'].tolist(), sfreq=fs, ch_types='seeg')
            raw = mne.io.read_raw_edf(os.path.join(session_path, folder_name, file_name), info, preload=True)
            raw.set_montage(montage)
        return raw, elec_names, fs

    raw_scalp, scalp_elec_names, scalp_fs = get_raw_and_set_montage(scalp=True)
    raw_ieeg, ieeg_elec_names, ieeg_fs = get_raw_and_set_montage(scalp=False)
    return raw_scalp, raw_ieeg, scalp_elec_names, ieeg_elec_names[0], ieeg_elec_names[1],ieeg_elec_names[2], ieeg_fs, scalp_fs


def get_annotations(patient, session, dataset_path, task_name):
    session_path = os.path.join(dataset_path, patient, session)
    annotations_path = os.path.join(session_path, 'ieeg', f'{patient}_{session}_task-{task_name}_run-01_events.tsv')
    annotations_df = pd.read_csv(annotations_path, sep='\t')
    annotations_df = annotations_df.loc[annotations_df['Artifact'] == 0]
    annotations_mne = mne.Annotations(onset=annotations_df['onset'].values,
                                  duration=annotations_df['duration'].values,
                                  description=annotations_df['Correct'].values)
    return annotations_mne, annotations_df

def get_hippo_elec_names(channels_data):
    hippo_elec_names = channels_data[channels_data['AnatomicalLocation'].str.contains('hippocampus', case=False, na=False)]['name'].tolist()
    hippo_elec_locations = channels_data[channels_data['AnatomicalLocation'].str.contains('hippocampus', case=False, na=False)]['AnatomicalLocation'].tolist()
    WM_elec_names = channels_data[channels_data['AnatomicalLocation']=='no_label_found']['name'].tolist()
    return [hippo_elec_names, hippo_elec_locations, WM_elec_names]

def return_refrenced_hippo_elec(mne_object, hippo_elec, type, WM_elec=None, mastoids_mean_signal=None):
    cur_mne_object = mne_object.copy()
    elec_idx = [cur_mne_object.ch_names.index(elec) for elec in hippo_elec]
    hippo_data = cur_mne_object.get_data(picks=hippo_elec)
    hippo_elec_final = hippo_elec.copy()
    if type == 'WM':
        WM_to_ref = []
        for elec in hippo_elec:
            relevant_wm = [wm_elec for wm_elec in WM_elec if wm_elec[:-1] == elec[:-1]]
            if len(relevant_wm) == 0:
                print(f'No WM electrode for {elec}')
                hippo_data = np.delete(hippo_data, hippo_elec.index(elec), axis=0)
                hippo_elec_final.remove(elec)
                elec_idx.remove(cur_mne_object.ch_names.index(elec))
                continue
            relevant_wm = sorted(relevant_wm, key=lambda x: x[-1], reverse=False)[0]
            WM_to_ref.append(relevant_wm)

        # since MNE don't support extracting the same electrode twice:
        unique_WM_elec = list(set(WM_to_ref))
        unique_WM_index = [unique_WM_elec.index(elec) for elec in WM_to_ref]
        final_ref_elec = [unique_WM_elec[i] for i in unique_WM_index]
        WM_data = cur_mne_object.get_data(picks=unique_WM_elec)
        WM_data = np.array([WM_data[i, :] for i in unique_WM_index])

        hippo_ref_data = hippo_data - WM_data
        # strings_to_print = [f'{hippo_elec_final[i]} - {WM_to_ref[i]}' for i in range(len(hippo_elec_final))]
        # print('WM reference: ', strings_to_print)

    elif type == 'BP':
        bp_idx = [curr_elec_idx+1 for curr_elec_idx in elec_idx]
        bp_elec = [cur_mne_object.ch_names[i] for i in bp_idx]
        bipolar_to_ref = []
        hippo_elec_final = hippo_elec.copy()
        for i, elec in enumerate(hippo_elec):
            if bp_elec[i][:-1] == elec[:-1]:
                bipolar_to_ref.append(bp_elec[i])
            else:
                print(f'No bipolar electrode for {elec}')
                hippo_data = np.delete(hippo_data, hippo_elec.index(elec), axis=0)
                hippo_elec_final.remove(elec)
                elec_idx.remove(cur_mne_object.ch_names.index(elec))

        unique_bp_elec = list(set(bipolar_to_ref))
        unique_bp_index = [unique_bp_elec.index(elec) for elec in bipolar_to_ref]
        final_ref_elec = [unique_bp_elec[i] for i in unique_bp_index]

        bp_data = cur_mne_object.get_data(picks=unique_bp_elec)
        bp_data = np.array([bp_data[i, :] for i in unique_bp_index])
        hippo_ref_data = hippo_data - bp_data
        # strings_to_print = [f'{hippo_elec[i]} - {bipolar_to_ref[i]}' for i in range(len(hippo_elec))]
        # print('BP reference: ', strings_to_print)

    elif type == 'mastoids+BP':
        hippo_ref_data = hippo_data - mastoids_mean_signal

        bp_idx = [curr_elec_idx+1 for curr_elec_idx in elec_idx]
        bp_elec = [cur_mne_object.ch_names[i] for i in bp_idx]
        bipolar_to_ref = []
        hippo_elec_final = hippo_elec.copy()
        for i, elec in enumerate(hippo_elec):
            if bp_elec[i][:-1] == elec[:-1]:
                bipolar_to_ref.append(bp_elec[i])
            else:
                print(f'No bipolar electrode for {elec}')
                hippo_data = np.delete(hippo_data, hippo_elec.index(elec), axis=0)
                hippo_elec_final.remove(elec)
                elec_idx.remove(cur_mne_object.ch_names.index(elec))

        unique_bp_elec = list(set(bipolar_to_ref))
        unique_bp_index = [unique_bp_elec.index(elec) for elec in bipolar_to_ref]
        final_ref_elec = [unique_bp_elec[i] for i in unique_bp_index]
        bp_data = cur_mne_object.get_data(picks=unique_bp_elec)
        bp_data = np.array([bp_data[i, :] for i in unique_bp_index])
        hippo_ref_data = hippo_ref_data - bp_data
        # strings_to_print = [f'{hippo_elec[i]} - {bipolar_to_ref[i]}' for i in range(len(hippo_elec))]
        # print('mastoids + BP reference: ', strings_to_print)

    # cur_mne_object = cur_mne_object.pick_channels(hippo_elec_final)
    # cur_mne_object._data = hippo_ref_data
    cur_mne_object._data[elec_idx] = hippo_ref_data
    print(hippo_elec_final)
    return cur_mne_object, hippo_elec_final, final_ref_elec

def return_refrenced_scalp_elec(mne_object):
    mne_object.set_eeg_reference(ref_channels=["A1", "A2"])
    return mne_object

def save_figures(raw_scalp, scalp_elec_names, raw_ieeg, hippo_elec_names, patient, session, figures_path):
    # plot spectrum and save
    fig, ax = plt.subplots()
    raw_scalp.plot_psd(picks=scalp_elec_names, ax=ax, show=False)
    ax.set_title(f'{patient}-{session} Scalp PSD', fontsize=14)
    fig.savefig(os.path.join(figures_path, patient, session, f'Scalp_PSD.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    raw_ieeg.plot_psd(fmin=0, fmax=150, picks=hippo_elec_names, ax=ax, show=False)
    line_handles = ax.get_lines()
    ax.legend(handles=line_handles[2:], labels=hippo_elec_names, fontsize=10, title_fontsize=12,
              loc='lower left', ncol=2)
    ax.set_title(f'Hippocampus PSD', fontsize=14, fontweight='bold')
    fig.savefig(os.path.join(figures_path, patient, session, f'Hippocampus_PSD.png'))

    plt.close()

    # for hippo_elec in hippo_elec_names:
    #     data = raw_ieeg.copy().pick_channels([hippo_elec]).get_data()
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=raw_ieeg.times, y=data[0], mode='lines'))
    #     fig.update_layout(title=f'{patient}-{session}-{hippo_elec} Raw Data',
    #                       xaxis_title='Time (s)',
    #                       yaxis_title='Amplitude')
    #     fig.write_image(os.path.join(figures_path, patient, session, f'{hippo_elec}_raw_data.png'))
    #
    # for scalp_elec in scalp_elec_names:
    #     data = raw_scalp.copy().pick_channels([scalp_elec]).get_data()
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=raw_scalp.times, y=data[0], mode='lines'))
    #     fig.update_layout(title=f'{patient}-{session}-{scalp_elec} Raw Data',
    #                       xaxis_title='Time (s)',
    #                       yaxis_title='Amplitude')
    #     fig.write_image(os.path.join(figures_path, patient, session, f'{scalp_elec}_raw_data.png'))

def return_trails_as_epochs_object(raw_ieeg, events, hippo_elec_names):
    tmin = 0
    tmax = 8-0.001
    raw_ieeg.set_annotations(events)
    ieeg_events, ieeg_event_id = mne.events_from_annotations(raw_ieeg)
    epochs_object = mne.Epochs(raw_ieeg, events=ieeg_events, event_id=ieeg_event_id, tmin=tmin, tmax=tmax,
                              baseline=None, preload=True)
    hippo_data = epochs_object.get_data(picks=hippo_elec_names)
    noisy_trails_indexes = remove_noisy_trails(hippo_data)
    if len(noisy_trails_indexes) > 0:
        epochs_object = epochs_object.drop(noisy_trails_indexes)
        print(f'{len(noisy_trails_indexes)} were excluded due to noise')
    good_epochs_indices = epochs_object.selection
    good_epochs_onsets = events.onset[good_epochs_indices]
    return epochs_object, good_epochs_onsets

def remove_noisy_trails(hippo_data, w=2.3):
    noisy_trails_indexes = []
    for elec_index in range(hippo_data.shape[1]):
        cur_elec_data = hippo_data[:, elec_index, :]
        means_vector = np.mean(cur_elec_data, axis=1)
        Q1 = np.percentile(means_vector, 25)
        Q3 = np.percentile(means_vector, 75)
        th = Q3 + w * (Q3 - Q1)
        idx = np.where(means_vector > th)[0]
        if len(idx)>0:
            noisy_trails_indexes.extend(idx)
    return list(set(noisy_trails_indexes))







