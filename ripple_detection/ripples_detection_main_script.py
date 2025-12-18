import os
import pandas as pd
from tqdm import tqdm
import mne
mne.set_log_level('CRITICAL')

import numpy as np
import joblib
import pickle
from functions_to_run_ripples_detection_algo_hybrid import *

projects_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
dataset_path = 'D:/ds004752'
methods_results_path = os.path.join(dataset_path, 'for_writing', 'methods_comparison_results')
mne_data_path = os.path.join(dataset_path, 'for_writing', 'mne_data')

patient_df = pd.read_csv(os.path.join(dataset_path, 'participants.tsv'), sep='\t')
patients = patient_df['participant_id'].tolist()
task_name = 'verbalWM'
methods_to_run = [
    'norman',
    'vaz',
    'staresina',
    'skelin',
    'henin',
    'sakon_kahana',
    'charupanit',
    # 'frauscher',
]
functions = [
run_norman_method,
run_vaz_method,
run_staresina_method,
run_skelin_method,
run_henin_method,
run_sakon_kahana_method,
run_charupanit_method,
# run_frauscher_method #TODO: run later? maybe we can skip it since it takes a lot of time
]
modes = [
    # 'full_recording',
    'trails',
]

trails_results = pd.DataFrame(columns=['patient', 'session', 'electrode', 'start', 'end','peak','confidence'])
for i, method in enumerate(methods_to_run):
    start = time()
    all_trails_results = []

    for patient in patients:
        # trails_results[patient] = {}
        print(f'Patient: {patient}')
        sessions = [f for f in os.listdir(os.path.join(dataset_path, patient)) if f.startswith('ses')]
        for session in tqdm(sessions):
            # trails_results[patient][session] = {}
            print(f'Session: {session}')
            #load fif files from mne_data_path
            if not os.path.exists(os.path.join(mne_data_path, patient, f'{patient}_{session}_WM_referenced_raw_ieeg.fif')):
                continue
            # WM_referenced_epochs_ieeg = mne.read_epochs(os.path.join(mne_data_path, patient, f'{patient}_{session}_hippo_data_wm-epo.fif'))
            # bp_referenced_epochs_ieeg = mne.read_epochs(os.path.join(mne_data_path, patient, f'{patient}_{session}_hippo_data_bp-epo.fif'))
            # mastoid_bp_referenced_epochs_ieeg = mne.read_epochs(os.path.join(mne_data_path, patient, f'{patient}_{session}_hippo_data_mastoids_bp-epo.fif'))

            hippo_data_wm = mne.io.read_raw(os.path.join(mne_data_path, patient, f'{patient}_{session}_WM_referenced_raw_ieeg.fif'))
            hippo_data_bp = mne.io.read_raw(os.path.join(mne_data_path, patient, f'{patient}_{session}_bp_referenced_raw_ieeg.fif'))
            hippo_mastoids_bp = mne.io.read_raw(os.path.join(mne_data_path, patient, f'{patient}_{session}_mastoids_bp_referenced_raw_ieeg.fif'))

            start_of_trails_WM = np.load(os.path.join(mne_data_path, patient, f'{patient}_{session}_start_of_trails_WM.npy'))
            start_of_trails_BP = np.load(os.path.join(mne_data_path, patient, f'{patient}_{session}_start_of_trails_BP.npy'))
            start_of_trails_BP_mas = np.load(os.path.join(mne_data_path, patient, f'{patient}_{session}_start_of_trails_BP_mas.npy'))

            hippo_elec_names_wm = joblib.load(os.path.join(mne_data_path, patient, f'{patient}_{session}_hippo_elec_names_wm.joblib'))
            hippo_elec_names_bp = joblib.load(os.path.join(mne_data_path, patient, f'{patient}_{session}_hippo_elec_names_bp.joblib'))
            hippo_elec_names_bp_mas = joblib.load(os.path.join(mne_data_path, patient, f'{patient}_{session}_hippo_elec_names_bp_mas.joblib'))

            trails_results = functions[i](patient, session, hippo_data_wm, hippo_data_bp, hippo_mastoids_bp, start_of_trails_WM, start_of_trails_BP, start_of_trails_BP_mas, hippo_elec_names_wm, hippo_elec_names_bp, hippo_elec_names_bp_mas)
            all_trails_results.append(trails_results)
    end = time()
    print(f'{method}: {(end-start)/60} min')
    all_trails_results = pd.concat(all_trails_results, axis=0, ignore_index=True)
    print(f'{method} {len(all_trails_results)} ripples')
    if method=='norman':
        method +='_bp'
    joblib.dump(all_trails_results, os.path.join(methods_results_path, f'all_trails_w_peak_all_patients_for_writing_{method}.pkl'))
    # save
# trails_results.to_csv(os.path.join(figures_path, f'all_trails_w_peak_all_patients.csv'), index=False)
# joblib.dump(trails_results, os.path.join(figures_path, f'all_trails_w_peak_all_patients_for_writing.pkl'))