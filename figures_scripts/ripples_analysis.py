from collections import defaultdict

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import pickle
import mne
import numpy as np
import scipy

from functions_for_analysis import (calculate_and_plot_iou, calculate_and_plot_rarm, calculate_agreement_within_session,
                                    plot_erp_per_patient_per_method,
                                    n_ripples_statistics, ripple_duration_analysis, plot_frequency_analysis,
                                    calc_corr_results, calculate_and_plot_ripples_agreement_histogram,
                                    plot_corr_results)

window_length = 150e-3  # 150 ms

projects_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
dataset_path = 'D:/ds004752'
figures_path = os.path.join(dataset_path, 'for_writing', 'figures')
mne_data_path = os.path.join(dataset_path, 'for_writing', 'mne_data')
methods_results_path = os.path.join(dataset_path, 'for_writing', 'methods_comparison_results')


patient_df = pd.read_csv(os.path.join(dataset_path, 'participants.tsv'), sep='\t')
patients = patient_df['participant_id'].tolist()
task_name = 'verbalWM'
methods_to_run = [
    'norman',
    'norman_bp',
    'henin',
    'vaz',
    'staresina',
    'skelin',
    'sakon_kahana',
    'charupanit',
    # 'frauscher',
]
modes = [
    # 'full_recording',
    'trails',
]


all_trails_results = []
for method in methods_to_run:
    method_df = joblib.load(os.path.join(methods_results_path, f'all_trails_w_peak_all_patients_for_writing_{method}.pkl'))
    method_df['method'] = method
    all_trails_results.append(method_df)

all_trails_results_df = pd.concat(all_trails_results)


calculate_and_plot_iou(all_trails_results_df)
calculate_and_plot_ripples_agreement_histogram(all_trails_results_df)
calculate_and_plot_rarm(all_trails_results_df)
n_ripples_statistics(all_trails_results_df, methods_to_run)
ripple_duration_analysis(all_trails_results_df)

calculate_agreement_within_session(all_trails_results_df) # not so relevant in my opinion...

second_part = False
if second_part:
    if os.path.exists(os.path.join(figures_path, 'hippo_signals_and_psd_w_peak.pkl')):
        with open(os.path.join(figures_path, 'hippo_signals_and_psd_w_peak.pkl'), 'rb') as f:
            hippo_signals_and_psd = pickle.load(f)
        # all_trails_results_df_with_freq_analysis = pd.read_csv(os.path.join(figures_path, 'all_trails_results_with_freq_analysis_w_peak.csv'))
    else:
        freq_range = (70, 250)
        hippo_signals_and_psd = {}
        for patient in patients:
            hippo_signals_and_psd[patient] = {}
            patient_df = all_trails_results_df[all_trails_results_df['patient'] == patient]
            for session in patient_df['session'].unique():
                hippo_signals_and_psd[patient][session] = {}
                session_df = patient_df[patient_df['session'] == session]

                ieeg_signal = mne.io.read_raw_fif(os.path.join(mne_data_path, patient, f'{patient}_{session}_WM_referenced_raw_ieeg.fif'), preload=True)
                for method in session_df['method'].unique():
                    hippo_signals_and_psd[patient][session][method] = {}
                    method_df = session_df[session_df['method'] == method]
                    for electrode in method_df['electrode'].unique():
                        filtered_ieeg_signal = mne.filter.filter_data(ieeg_signal.get_data(picks=electrode), sfreq=ieeg_signal.info['sfreq'], l_freq=freq_range[0], h_freq=freq_range[1])
                        electrode_df = method_df[method_df['electrode'] == electrode]
                        hippo_signals_and_psd[patient][session][method][electrode] = defaultdict(list)
                        for i, row in electrode_df.iterrows():
                            start = row['start']
                            end = row['end']
                            start_idx = np.argmin(np.abs(ieeg_signal.times - start))
                            end_idx = np.argmin(np.abs(ieeg_signal.times - end))
                            data = ieeg_signal.get_data(start=start_idx, stop=end_idx, picks=electrode)[0,:]
                            hippo_signals_and_psd[patient][session][method][electrode]['data'].append(data)

                            filtered_data = filtered_ieeg_signal[0, start_idx:end_idx]
                            hippo_signals_and_psd[patient][session][method][electrode]['filtered_data'].append(filtered_data)

                            peak = row['peak']
                            peak_idx = np.argmin(np.abs(ieeg_signal.times - peak))
                            if peak_idx - int(window_length * ieeg_signal.info['sfreq'] // 2) < 0 or peak_idx + int(window_length * ieeg_signal.info['sfreq'] // 2) > len(ieeg_signal.times):
                                continue
                            peak_data = ieeg_signal.get_data(start=peak_idx - int(window_length * ieeg_signal.info['sfreq'] // 2),
                                                             stop=peak_idx + int(window_length * ieeg_signal.info['sfreq'] // 2), picks=electrode)[0,:]
                            hippo_signals_and_psd[patient][session][method][electrode]['window_data'].append(peak_data)
                            filtered_peak_data = filtered_ieeg_signal[0, peak_idx - int(window_length * ieeg_signal.info['sfreq'] // 2):
                                                                         peak_idx + int(window_length * ieeg_signal.info['sfreq'] // 2)]
                            hippo_signals_and_psd[patient][session][method][electrode]['filtered_window_data'].append(filtered_peak_data)

                            # find main frequency within the range 70-250 Hz
                            f, Pxx = scipy.signal.periodogram(data, fs=ieeg_signal.info['sfreq'])
                            freq_indices = np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]
                            max_f = f[freq_indices[np.argmax(Pxx[freq_indices])]]
                            hippo_signals_and_psd[patient][session][method][electrode]['pxx'].append(Pxx)

                            # Compute the total power in the range 70-250 Hz
                            f, t_spec, Sxx = scipy.signal.spectrogram(data, fs=ieeg_signal.info['sfreq'], nperseg=16, noverlap=8, nfft=256)
                            freq_indices = np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]
                            total_power_ripple_band = 10 * np.log10(np.sum(Sxx[freq_indices, :]))

                            all_trails_results_df.loc[(all_trails_results_df['patient'] == patient) & (all_trails_results_df['session'] == session) & (all_trails_results_df['method'] == method) & (all_trails_results_df['electrode'] == electrode) & (all_trails_results_df['start'] == start) & (all_trails_results_df['end'] == end), 'main_frequency'] = max_f
                            all_trails_results_df.loc[(all_trails_results_df['patient'] == patient) & (all_trails_results_df['session'] == session) & (all_trails_results_df['method'] == method) & (all_trails_results_df['electrode'] == electrode) & (all_trails_results_df['start'] == start) & (all_trails_results_df['end'] == end), 'total_power_ripple_band'] = total_power_ripple_band

    all_trails_results_df_with_freq_analysis = all_trails_results_df.copy()
    all_trails_results_df_with_freq_analysis.to_csv(os.path.join(figures_path, 'all_trails_results_with_freq_analysis_w_peak.csv'), index=False)
    signals_path = os.path.join(figures_path, 'hippo_signals_and_psd_w_peak.pkl')
    with open(signals_path, 'wb') as f:
        pickle.dump(hippo_signals_and_psd, f)

    plot_frequency_analysis(all_trails_results_df_with_freq_analysis)
    plot_erp_per_patient_per_method(hippo_signals_and_psd, methods_to_run)