import os
from collections import defaultdict
import joblib
import pandas as pd

import numpy as np
from data_prep.prepare_data_utils import load_raw_and_set_montage, return_refrenced_scalp_elec, \
    save_figures, return_refrenced_hippo_elec, get_annotations, return_trails_as_epochs_object

################################ ds004752 dataset ################################
"""
This script is used to prepare the data from the ds004752 dataset.
The dataset is organized in the following way:
- The main folder contains a participants.tsv file that contains the participants' ids.
- Each participant has a folder with the participant's id.
- Each participant's folder contains a folder for each session.
- Each session folder contains the following files:
    - A .edf file that contains the scalp EEG data.
    - A .edf file that contains the intracranial EEG data.
    - A .tsv file that contains the events.
    - A .tsv file that contains the annotations.
"""

plot_data = False
save_data = True
projects_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
dataset_path = ['D:/ds004752', os.path.join(projects_path, 'ds004752-download')][0]
figures_path = os.path.join(dataset_path, 'for_writing', 'figures',)
mne_data_path = os.path.join(dataset_path, 'for_writing', 'mne_data')

patient_df = pd.read_csv(os.path.join(dataset_path, 'participants.tsv'), sep='\t')
patients = patient_df['participant_id'].tolist()
task_name = 'verbalWM'

patient_df_final = pd.DataFrame(columns=['patient', 'session', 'hippo_left', 'hippo_right', 'reference_elecs_left', 'reference_elecs_right', 'hippo_fs', 'scalp', 'scalp_fs'])
all_electrodes = defaultdict(dict)
for patient in patients:
    print('----------', patient, '----------')
    all_electrodes[patient] = {}
    all_scalp_electrodes_patient = []
    all_hippo_electrodes_patient = []
    sessions = [f for f in os.listdir(os.path.join(dataset_path, patient)) if f.startswith('ses')]
    for session in sessions:
        print('-----', session, '-----')
        raw_scalp, raw_ieeg, scalp_elec_names, hippo_elec_names, hippo_elec_location, WM_elec_names, ieeg_fs, scalp_fs = load_raw_and_set_montage(patient, session, dataset_path, task_name)

        right_elecs = [elec for elec in hippo_elec_names if elec[-2] == 'R']
        left_elecs = [elec for elec in hippo_elec_names if elec[-2] == 'L']
        raw_scalp.filter(l_freq=0.1, h_freq=None, verbose=False)  # detrending
        raw_scalp.notch_filter([50], notch_widths=1.0, method='iir', trans_bandwidth=1, filter_length='auto',
                               phase='zero', verbose=False)  # powerline noise removal
        ref_scalp = return_refrenced_scalp_elec(raw_scalp)


        if scalp_fs > 200:
            ref_scalp.filter(l_freq=None, h_freq=100, verbose=False)



        if len(hippo_elec_names) == 0:
            continue
        WM_referenced_raw_ieeg, hippo_elec_names_wm, final_ref_elec_wm = return_refrenced_hippo_elec(raw_ieeg, hippo_elec_names, type='WM', WM_elec=WM_elec_names)
        bp_referenced_raw_ieeg, hippo_elec_names_bp, final_ref_elec_bp = return_refrenced_hippo_elec(raw_ieeg, hippo_elec_names, type='BP')
        mastoids_bp_referenced_raw_ieeg, hippo_elec_names_bp_mas, final_ref_elec_mas_bp = return_refrenced_hippo_elec(raw_ieeg, hippo_elec_names,
                                                                      type='mastoids+BP',
                                                                      mastoids_mean_signal=raw_scalp.copy().resample(
                                                                          ieeg_fs).get_data(
                                                                          picks=['A1', 'A2']).mean(axis=0),
                                                                      WM_elec=WM_elec_names)

        events, annotation_df = get_annotations(patient, session, dataset_path, task_name)
        hippo_data_wm, start_of_trails_WM = return_trails_as_epochs_object(WM_referenced_raw_ieeg, events,
                                                                           hippo_elec_names_wm)
        hippo_data_bp, start_of_trails_BP = return_trails_as_epochs_object(bp_referenced_raw_ieeg, events,
                                                                           hippo_elec_names_bp)
        hippo_data_mastoids_bp, start_of_trails_BP_mas = return_trails_as_epochs_object(mastoids_bp_referenced_raw_ieeg,
                                                                                        events, hippo_elec_names_bp_mas)

        final_ref_elec_R = [elec for elec in final_ref_elec_wm if elec[-2] == 'R']
        final_ref_elec_L = [elec for elec in final_ref_elec_wm if elec[-2] == 'L']

        for_df = {'patient': patient, 'session': session, 'hippo_left': [left_elecs],
                  'hippo_right':[right_elecs],  'reference_elecs_left_wm': [[elec for elec in final_ref_elec_wm if elec[-2] == 'L']],
                  'reference_elecs_left_bp': [[elec for elec in final_ref_elec_bp if elec[-2] == 'L']],
                  'reference_elecs_left_mas_bp': [[elec for elec in final_ref_elec_mas_bp if elec[-2] == 'L']],
                  'reference_elecs_right_wm': [[elec for elec in final_ref_elec_wm if elec[-2] == 'R']],
                  'reference_elecs_right_bp': [[elec for elec in final_ref_elec_bp if elec[-2] == 'R']],
                  'reference_elecs_right_mas_bp': [[elec for elec in final_ref_elec_mas_bp if elec[-2] == 'R']],
                  'hippo_fs': ieeg_fs, 'scalp': [scalp_elec_names], 'scalp_fs': scalp_fs}
        patient_df_final = pd.concat([patient_df_final, pd.DataFrame(for_df)], ignore_index=True)


        if plot_data:
            if not os.path.exists(os.path.join(figures_path, patient, session)):
                os.makedirs(os.path.join(figures_path, patient, session))
            scalp_elec_names= ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
            save_figures(raw_scalp, scalp_elec_names, bp_referenced_raw_ieeg, hippo_elec_names, patient, session, figures_path)

        if save_data:
            if not os.path.exists(os.path.join(mne_data_path, patient)):
                os.makedirs(os.path.join(mne_data_path, patient))
            WM_referenced_raw_ieeg.save(os.path.join(mne_data_path, patient, f'{patient}_{session}_WM_referenced_raw_ieeg.fif'), overwrite=True)
            bp_referenced_raw_ieeg.save(os.path.join(mne_data_path, patient, f'{patient}_{session}_bp_referenced_raw_ieeg.fif'), overwrite=True)
            mastoids_bp_referenced_raw_ieeg.save(os.path.join(mne_data_path, patient, f'{patient}_{session}_mastoids_bp_referenced_raw_ieeg.fif'), overwrite=True)
            ref_scalp.save(os.path.join(mne_data_path, patient, f'{patient}_{session}_scalp_ref_eeg.fif'), overwrite=True)

            hippo_data_wm.save(os.path.join(mne_data_path, patient, f'{patient}_{session}_hippo_data_wm-epo.fif'), overwrite=True)
            hippo_data_bp.save(os.path.join(mne_data_path, patient, f'{patient}_{session}_hippo_data_bp-epo.fif'), overwrite=True)
            hippo_data_mastoids_bp.save(os.path.join(mne_data_path, patient, f'{patient}_{session}_hippo_data_mastoids_bp-epo.fif'), overwrite=True)

            np.save(os.path.join(mne_data_path, patient, f'{patient}_{session}_start_of_trails_WM.npy'), start_of_trails_WM)
            np.save(os.path.join(mne_data_path, patient, f'{patient}_{session}_start_of_trails_BP.npy'), start_of_trails_BP)
            np.save(os.path.join(mne_data_path, patient, f'{patient}_{session}_start_of_trails_BP_mas.npy'), start_of_trails_BP_mas)

            joblib.dump(hippo_elec_names_wm, os.path.join(mne_data_path, patient, f'{patient}_{session}_hippo_elec_names_wm.joblib'))
            joblib.dump(hippo_elec_names_bp, os.path.join(mne_data_path, patient, f'{patient}_{session}_hippo_elec_names_bp.joblib'))
            joblib.dump(hippo_elec_names_bp_mas, os.path.join(mne_data_path, patient, f'{patient}_{session}_hippo_elec_names_bp_mas.joblib'))

        patient_df_final.to_csv(os.path.join(mne_data_path, 'patients_details_final_df.csv'))

#############################################################################################################################




############################### Ichilov dataset ################################
# TODO!!
