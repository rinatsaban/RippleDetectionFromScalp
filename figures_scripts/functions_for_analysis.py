import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from fastdtw import fastdtw
from scipy.signal import find_peaks
from collections import defaultdict
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine, euclidean

from scipy.signal import correlate
from sklearn.metrics import mutual_info_score
from tqdm import tqdm

dataset_path = 'D:/ds004752'
figures_path = os.path.join(dataset_path, 'for_writing', 'figures')
mne_data_path = os.path.join(dataset_path, 'for_writing', 'mne_data')
methods_results_path = os.path.join(dataset_path, 'for_writing', 'methods_comparison_results')

tab20_colors = list(plt.cm.tab20.colors)
tab20b_colors = list(plt.cm.tab20b.colors)
combined_palette = tab20_colors + tab20b_colors


def computeRippleActivityReliabilityMetric(ripple_times, n_electrodes, a=0.25, b=0.5, penalty_weight = 0.1):
    """
    Calculate the Ripple Activity Reliability Metric (RARM).

    Parameters:
    ripple_times (np.ndarray): A numpy array of shape (n, 2) where each row represents [start_time, end_time] of a ripple.
    phase_intervals (dict): A dictionary defining the time intervals for each phase.
                           The keys are phase names ('encoding', 'recall', 'maintenance', 'fixation'),
                           and the values are lists [start, end] indicating the duration of each phase in a cycle.

    Returns:
    float: The final metric combining consistency and weighted error.
    """
    # Step 1: Assign ripples to phases
    phase_intervals = {
        'encoding': [1, 3],
        'recall': [6, 8],
        'maintenance': [3, 6],
        'fixation': [0, 1],
    }
    phase_durations = {phase: end - start for phase, (start, end) in phase_intervals.items()}
    trail_dur = sum(phase_durations.values())

    # Determine the phase for each ripple
    ripple_counts = {phase: 0 for phase in phase_intervals.keys()}
    trails_ids = []
    for ripple_start, ripple_end in ripple_times:
        # Map ripple time to the corresponding phase
        time_in_cycle = ripple_start % trail_dur
        trail_ids = ripple_start//trail_dur
        for phase, (start, end) in phase_intervals.items():
            if start <= time_in_cycle < end:
                ripple_counts[phase] += 1
                trails_ids.append(trail_ids)
                break

    n_trails = len(set(trails_ids))
    r_encoding = ripple_counts['encoding'] / (n_electrodes*n_trails*phase_durations['encoding'])
    r_recall = ripple_counts['recall'] / (n_electrodes*n_trails*phase_durations['recall'])
    r_maintenance = ripple_counts['maintenance'] / (n_electrodes*n_trails*phase_durations['maintenance'])
    r_fixation = ripple_counts['fixation'] / (n_electrodes*n_trails*phase_durations['fixation'])

    final_score = (r_encoding + r_recall) / (r_encoding + r_recall + a*r_maintenance + b*r_fixation)

    return final_score

def compute_iou_for_two_methods(two_methods_patient_df, method1, method2):
    all_union = 0
    all_intersection = 0
    session_elec_df = two_methods_patient_df.groupby(['session', 'electrode'])
    for (session, elec), data in session_elec_df:
        detections1 = data[data['method'] == method1][['start', 'end']].values
        detections2 = data[data['method'] == method2][['start', 'end']].values
        intersection, union, _ = compute_iou(detections1, detections2)
        all_union += union
        all_intersection += intersection
    return all_intersection / all_union if all_union > 0 else 0

def compute_iou(detections1, detections2, time_resolution=0.001):
    if detections1.shape[0] == 0 and detections2.shape[0] == 0:
        return 0, 0, 0
    elif detections1.shape[0] == 0 or detections2.shape[0] == 0:
        intersection = 0
        union = np.sum(detections1[:, 1] - detections1[:, 0])//time_resolution if detections2.shape[0] == 0 else (
                np.sum(detections2[:, 1] - detections2[:, 0])//time_resolution)
        return intersection, union, 0

    min_time = min(detections1[0][0], detections2[0][0])
    max_time = max(detections1[-1][1], detections2[-1][1])
    common_time_range = np.arange(min_time, max_time, time_resolution)

    events_vector1 = create_events_vector(detections1, common_time_range)
    events_vector2 = create_events_vector(detections2, common_time_range)

    # Compute intersection and union
    intersection = np.sum(np.logical_and(events_vector1, events_vector2))
    union = np.sum(np.logical_or(events_vector1, events_vector2))

    # Compute IoU
    overall_iou = intersection / union if union > 0 else 0
    return intersection, union, overall_iou


def create_events_vector(intervals, common_time_range):
    """
    Convert intervals into a binary time vector.
    Each time step corresponds to a `time_resolution` unit.
    """
    if len(intervals) == 0:
        return np.zeros_like(common_time_range)

    # Initialize a binary vector
    events_vector = np.zeros(len(common_time_range), dtype=int)

    # Mark 1 for the presence of a ripple
    for start, end in intervals:
        start_idx = np.searchsorted(common_time_range, start, side='left')
        end_idx = np.searchsorted(common_time_range, end, side='right')
        events_vector[start_idx:end_idx] = 1

    return events_vector


def calculate_and_plot_rarm(data):
    rarm_results = []
    methods = data['method'].unique()
    for method in methods:
        result_dict = {'method': method}
        methods_df = data[data['method'] == method]
        grouped = methods_df.groupby('patient')
        for (patient), group in grouped:
            detections = group[group['method'] == method][['start', 'end']].values
            n_electrodes = len(group[group['method'] == method]['electrode'].unique())
            RARM_Score = computeRippleActivityReliabilityMetric(detections, n_electrodes)
            result_dict[patient] = RARM_Score
        rarm_results.append(result_dict)

    rarm_results_df = pd.DataFrame(rarm_results)
    rarm_results_df.fillna(0, inplace=True)
    rarm_results_df['mean_rarm'] = rarm_results_df.drop(columns=['method'], axis=1).mean(axis=1)
    rarm_results_df.sort_values('mean_rarm', ascending=False, inplace=True)
    rarm_results_df.to_csv(os.path.join(figures_path, 'rarm_results_df.csv'), index=False)

    # Display patterns for each category
    plt.figure(figsize=(8, 8))
    for (i, method_df), color in zip(rarm_results_df.iterrows(), combined_palette):
        plt.plot(method_df.index[1:-1], method_df.values[1:-1],
                 label=f"{method_df['method']} - Mean: {method_df['mean_rarm']:.2f}",
                 color=color, marker='o')
    plt.title(f'RATAM by Patient')
    plt.xlabel('Patient')
    plt.ylabel('Average RATAM')
    plt.xticks(rotation=45, fontsize=8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'RATAM_by_patient_and_method_pairs.png'))
    plt.close()

    # Create a violin plot
    df_melted = rarm_results_df.melt(id_vars=["method"],
                                  value_vars=[col for col in rarm_results_df.columns if "sub-" in col],
                                  var_name="Subject", value_name="RARM")
    df_melted.fillna(0, inplace=True)
    plt.figure(figsize=(12, 6))

    # Create the box plot
    ax = sns.boxplot(data=df_melted, x="RARM", y="method", palette=combined_palette)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

    # Customize the plot
    plt.title("RATAM Distribution by Method", fontdict={"fontsize": 16})
    plt.xlabel("RATAM")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, "RATAM_by_method_pairs_boxplot.png"))
    plt.close()


def calculate_and_plot_iou(data):
    agreement_results = []
    methods = data['method'].unique()
    method_pairs = list(combinations(methods, 2))
    for method1, method2 in method_pairs:
        pairs_dict = {'method1': method1, 'method2': method2}
        methods_df = data[(data['method'] == method1) | (data['method'] == method2)]
        grouped = methods_df.groupby('patient')
        for (patient), group in grouped:
            iou = compute_iou_for_two_methods(group, method1, method2)
            pairs_dict[patient] = iou
        agreement_results.append(pairs_dict)

    # Convert results to a DataFrame
    agreement_df = pd.DataFrame(agreement_results)
    agreement_df['pair_mean_iou'] = agreement_df.drop(columns=['method1', 'method2'], axis=1).mean(axis=1)
    agreement_df.sort_values('pair_mean_iou', ascending=False, inplace=True)
    agreement_df.to_csv(os.path.join(figures_path, 'agreement_df.csv'), index=False)

    # Display patterns for each category
    plt.figure(figsize=(8, 8))
    for (i, pair_df), color in zip(agreement_df.iterrows(), combined_palette):
        plt.plot(pair_df.index[2:-1], pair_df.values[2:-1],
                    label=f"{pair_df['method1']} vs {pair_df['method2']} - Mean: {pair_df['pair_mean_iou']:.2f}",
                    color=color, marker='o')
    plt.title(f'IoU by Patient and Method Pairs')
    plt.xlabel('Patient')
    plt.ylabel('Average IoU')
    plt.xticks(rotation=45, fontsize=8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'IoU_by_patient_and_method_pairs.png'))
    plt.close()


    # Create a violin plot
    df_melted = agreement_df.melt(id_vars=["method1", "method2"], value_vars=[col for col in agreement_df.columns if "sub-" in col],
                        var_name="Subject", value_name="IoU")
    df_melted["Method Pair"] = df_melted["method1"] + " vs " + df_melted["method2"]
    plt.figure(figsize=(12, 6))

    # Create the box plot
    ax = sns.boxplot(data=df_melted, x="IoU", y="Method Pair", palette=combined_palette)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

    # Customize the plot
    plt.title("IoU Distribution by Method Pairs", fontdict={"fontsize": 16})
    plt.xlabel("IoU")
    plt.ylabel("Method Pair")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, "iou_by_method_pairs_boxplot.png"))
    plt.close()



def calculate_agreement_within_session(data):
    # The purpose is to check consistency of the detections within the same session and method on different electrodes
    grouped = data.groupby(['patient', 'session', 'method'])
    agreement_results = []
    for (patient, session, method), group in grouped:
        electrodes = group['electrode'].unique()
        electrode_pairs = list(combinations(electrodes, 2))
        session_iou = []
        for electrode1, electrode2 in electrode_pairs:
            detections1 = group[group['electrode'] == electrode1][['start', 'end']].values
            detections2 = group[group['electrode'] == electrode2][['start', 'end']].values
            iou = compute_iou(detections1, detections2)
            session_iou.append(iou)
        if len(session_iou) > 0:
            agreement_results.append({'session': session, 'method': method, 'patient': patient, 'mean_iou': np.mean(np.array(session_iou))})
    agreement_df = pd.DataFrame(agreement_results)
    plt.figure(figsize=(8, 6))
    method_means = agreement_df.groupby('method')['mean_iou'].mean().sort_values(ascending=False)
    sorted_methods = method_means.index
    for method, color in zip(sorted_methods, combined_palette):
        method_df = agreement_df[agreement_df['method'] == method]
        avg_iou_by_patient = method_df.groupby('patient')['mean_iou'].mean()
        plt.plot(avg_iou_by_patient.index, avg_iou_by_patient.values, label=f'{method} - Mean: {avg_iou_by_patient.mean():.2f}', color=color, marker='o')
    plt.title('Mean IoU between electrodes by Patient and Method')
    plt.xlabel('Patient')
    plt.ylabel('Average IoU')
    plt.xticks(rotation=45, fontsize=8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'mean_iou_between_electrodes_by_patient_and_method.png'))
    plt.close()

    # violin plot
    plt.figure(figsize=(12, 6))
    iou_data = []
    labels = []
    method_means = agreement_df.groupby('method')['mean_iou'].mean().sort_values(ascending=False)
    sorted_methods = method_means.index
    for method in sorted_methods:
        group = agreement_df[agreement_df['method'] == method]
        values = group['mean_iou'].dropna()
        iou_data.extend(values)
        labels.extend([method] * len(values)
    )
    violin_df = pd.DataFrame({'Mean IoU': iou_data, 'Method': labels})
    sns.violinplot(
        data=violin_df,
        x='Mean IoU',
        y='Method',
        cut=0,
        inner='quart',
        palette=combined_palette,
        order=sorted_methods,
        hue='Method',
        legend=False
    )
    plt.title('Mean IoU Distribution between electrodes by Method')
    plt.xlabel('Mean IoU')
    plt.ylabel('Method')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'mean_iou_between_electrodes_by_method_violinplot.png'))
    plt.close()


def n_ripples_statistics(data_df, methods_to_run):
    processed_df = []
    grouped = data_df.groupby(['patient', 'session', 'electrode'])
    for group, data in grouped:
        methods = data['method'].unique()
        detections = [len(data[data['method'] == method]) for method in methods]
        processed_df.append({
            'patient': group[0],
            'session': group[1],
            'electrode': group[2],
            **{method: detection for method, detection in zip(methods, detections)}
        })
    processed_df = pd.DataFrame(processed_df)
    processed_df.fillna(0, inplace=True)
    # processed_df.to_csv(os.path.join(figures_path, 'zero_ripples.csv'), index=False)

    grouped = processed_df.groupby(['patient', 'session'])
    df_list = []
    for group, data in grouped:
        for method in methods_to_run:
            df_list.append({
                'patient': group[0],
                'session': group[1],
                'method': method,
                'mean': data[method].mean(),
                'std': data[method].std(),
            })

    df = pd.DataFrame(df_list)

    # Create a single figure with one subplot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare the data for side-by-side bar plots
    df_melted = df.melt(id_vars=['method'], value_vars=['mean', 'std'], var_name='Metric', value_name='Value')

    # Plot the side-by-side vertical bar plot
    sns.barplot(
        data=df_melted,
        x='method',
        y='Value',
        hue='Metric',
        palette=combined_palette,
        ax=ax
    )

    # Add values above the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=10, padding=3)

    # Customize the plot
    ax.set_title('Mean and Standard Deviation of number of Ripples within one session')
    ax.set_xlabel('Method')
    ax.set_ylabel('Value')
    ax.legend(title='Metric')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'n_ripples_vertical_barplots_with_values.png'))
    plt.close()

    # Create a violin plot
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.violinplot(
        data=df_melted,
        x='method',
        y='Value',
        hue='Metric',
        inner='quart',
        split=True,  # This parameter is set to True to show side-by-side violins
        palette=combined_palette,  # Adjust the palette as needed
        ax=ax
    )

    # Customize the plot
    ax.set_title('Mean and Standard Deviation of Ripples within one Session')
    ax.set_xlabel('Method')
    ax.set_ylabel('Value')
    ax.legend(title='Metric')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'n_ripples_violin_plot_with_values.png'))
    plt.close()

def ripple_duration_analysis(data_df):
    data_df['duration'] = (data_df['end'] - data_df['start'])*1000

    # Calculate mean and std for each method
    stats_df = data_df.groupby('method')['duration'].agg(['mean', 'std']).reset_index()
    stats_df['label'] = stats_df.apply(lambda x: f"{x['method']} ({x['mean']:.1f}±{x['std']:.1f})", axis=1)

    # Merge the new labels back into the original dataframe
    data_df = data_df.merge(stats_df[['method', 'label']], on='method', how='left')

    # Create violin plot with the formatted labels
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=data_df, x='duration', y='label', cut=0, inner='quart', palette='viridis', hue='label', legend=False)

    plt.title('Ripple Duration Distribution by Method')
    plt.xlabel('Duration (ms)')
    plt.ylabel('Method')
    plt.xlim(0, 500)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'ripple_duration_by_method_violinplot_500.png'))
    plt.close()

def plot_frequency_analysis(all_trails_results_df_with_freq_analysis):
    groups = all_trails_results_df_with_freq_analysis.groupby('method')

    # violin plot of main frequency and power in the range 70-180 Hz
    plt.figure(figsize=(12, 6))
    freq_data = []
    power_data = []
    labels = []
    for (method), data in groups:
        freq_data.extend(data['main_frequency'])
        power_data.extend(data['total_power_ripple_band'])
        mean_freq = data['main_frequency'].mean()
        std_freq = data['main_frequency'].std()
        labels.extend([f"{method} ({mean_freq:.1f}±{std_freq:.1f} Hz)"] * len(data))
        # labels.extend(method * len(data))
    violin_df = pd.DataFrame({'Main Frequency': freq_data, 'Total Power': power_data, 'Method': labels})

    sns.violinplot(
        data=violin_df,
        x='Main Frequency',
        y='Method',
        # cut=250,
        inner='quart',
        palette=combined_palette,
        hue='Method',
        legend=False
    )
    plt.xlim([0, 300])
    # add dashed lines in x = 70 and x = 180
    plt.axvline(70, color='black', linestyle='--')
    plt.axvline(250, color='black', linestyle='--')
    plt.title('Main Frequency within the Ripple Band by Method')
    plt.xlabel('Main Frequency [Hz]')
    plt.ylabel('Method')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'main_frequency_by_method_violinplot.png'))
    plt.close()

    violin_df['Method'] = violin_df['Method'].apply(lambda x: x.split(' ')[0])
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=violin_df,
        x='Total Power',
        y='Method',
        # cut=0,
        inner='quart',
        palette=combined_palette,
        hue='Method',
        legend=False
    )
    plt.title('Total Power Distribution within the Ripple band by Method')
    plt.xlabel('Total Power (dB)')
    plt.ylabel('Method')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'total_power_by_method_violinplot.png'))
    plt.close()


def get_peak_index(filtered_ieeg_signal, data, start_idx, end_idx):
    filtered_data = filtered_ieeg_signal[0, start_idx:end_idx+1]
    all_pks, _ = find_peaks(filtered_data)
    if len(all_pks) == 0:
        return start_idx + np.argmax(filtered_data)
    peak = all_pks[np.argmax(filtered_data[all_pks])]
    min_indices, _ = find_peaks(-data)
    if len(min_indices) == 0:
        return peak + start_idx
    peak_idx = min_indices[np.argmin(np.abs(min_indices - peak))]
    return peak_idx + start_idx


def calc_corr_results(hippo_signals_and_psd, data_type):
    results = {}
    for patient in hippo_signals_and_psd.keys():
        results[patient] = {}
        all_patient_results = defaultdict(list)

        for session in hippo_signals_and_psd[patient].keys():
            for method in hippo_signals_and_psd[patient][session].keys():
                for electrode in hippo_signals_and_psd[patient][session][method].keys():
                    all_patient_results[method].extend(hippo_signals_and_psd[patient][session][method][electrode][data_type])

        for method in tqdm(all_patient_results.keys()):
            similarity_metrics = []
            for i in range(len(all_patient_results[method])):
                for j in range(i + 1, len(all_patient_results[method])):
                    x = all_patient_results[method][i]
                    y = all_patient_results[method][j]
                    curr_results = []

                    curr_results.append(cross_correlation(x, y))
                    curr_results.append(get_mi_score(x, y))
                    curr_results.append(get_dtw_distance(x, y))

                    if 'window' in data_type:
                        curr_results.append(1 - cosine(x, y))
                        curr_results.append(pearsonr(x, y)[0])
                    similarity_metrics.append(curr_results)
            results[patient][method] = [np.array(similarity_metrics).mean(axis=0), np.array(similarity_metrics).std(axis=0)]
    return results

def get_mi_score(x, y):
    hist1, _ = np.histogram(x, bins=10)
    hist2, _ = np.histogram(y, bins=10)
    return mutual_info_score(hist1, hist2)

def get_dtw_distance(x, y):
    z_x = (x - np.mean(x)) / np.std(x)
    z_y = (y - np.mean(y)) / np.std(y)
    return fastdtw(z_x.T, z_y.T, dist=2)[0]
def cross_correlation(signal1, signal2):
    return np.max(correlate(signal1 - np.mean(signal1), signal2 - np.mean(signal2), mode='full') / (np.std(signal1) * np.std(signal2) * len(signal1)))

# def dtw_align(signal1, signal2):
#     alignment = dtw.warping_path(signal1, signal2)
#     aligned_signal1 = np.array([signal1[i] for i, j in alignment])
#     aligned_signal2 = np.array([signal2[j] for i, j in alignment])
#     return aligned_signal1, aligned_signal2


def calculate_and_plot_ripples_agreement_histogram(data, intersection_threshold = 0.01):
    # Sort the dataframe by patient, session, electrode, and start time
    df_sorted = data.sort_values(by=['patient', 'session', 'electrode', 'start']).reset_index(drop=True)

    # Initialize a column for ripple groups
    df_sorted['ripple_group'] = np.nan

    # Dictionary to track ongoing groups
    ripple_groups = {}
    group_id = 0

    # Iterate over the sorted ripples
    for index, row in df_sorted.iterrows():
        key = (row['patient'], row['session'], row['electrode'])

        if key not in ripple_groups:
            ripple_groups[key] = []  # Store (start, end, group_id)

        assigned = False
        start_time, end_time = row['start'], row['end']

        # Check if the ripple intersects with any existing group in the same patient/session/electrode
        for i, (existing_start, existing_end, existing_group) in enumerate(ripple_groups[key]):
            if max(start_time, existing_start) <= min(end_time, existing_end) - intersection_threshold:
                df_sorted.at[index, 'ripple_group'] = existing_group
                # Expand group range
                ripple_groups[key][i] = (min(start_time, existing_start), max(end_time, existing_end), existing_group)
                assigned = True
                break

        # If not assigned to a group, create a new one
        if not assigned:
            group_id += 1
            df_sorted.at[index, 'ripple_group'] = group_id
            ripple_groups[key].append((start_time, end_time, group_id))

    # Count unique methods detecting each ripple group
    ripple_counts = df_sorted.groupby(['patient', 'session', 'electrode', 'ripple_group'])[
        'method'].nunique().reset_index()

    # Rename the count column
    ripple_counts.rename(columns={'method': 'num_methods_detected'}, inplace=True)

    # Create a histogram with labels centered on the x-axis bins, starting from 1
    plt.figure(figsize=(8, 5))
    bins = range(1, ripple_counts['num_methods_detected'].max() + 2)
    hist_values, bin_edges, _ = plt.hist(ripple_counts['num_methods_detected'], bins=bins, edgecolor='black', alpha=0.7)

    # Add value labels on top of each bin, centered
    for i in range(len(hist_values)):
        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2  # Center the label
        plt.text(bin_center, hist_values[i] + 0.5, str(int(hist_values[i])), ha='center', fontsize=10)

    # Adjust x-ticks to be centered on bins, starting from 1
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    plt.xticks(bin_centers, labels=[str(int(x)) for x in range(1, len(bin_centers) + 1)])  # Ensure labels start from 1

    # Labels and title
    plt.xlabel("Number of Methods Detected the Ripples")
    plt.ylabel("Number of Ripples")
    plt.title("Histogram of Number of Methods Detecting the Same Ripples")
    plt.draw()
    plt.savefig(os.path.join(figures_path, 'ripple_detection_histogram.png'))
    plt.close()


def plot_erp_per_patient_per_method(hippo_signals_and_psd, methods_to_run):
    loc_dict = {f'sub-{i + 1:02d}' if i < 9 else f'sub-{i + 1}': i for i in range(12)}
    loc_dict.update({f'sub-{i + 1}': i-1 for i in range(13,15)})
    for i, method in enumerate(methods_to_run):
        loc_dict[method] = i

    sub_dict_raw = {}
    sub_dict_filtered = {}
    for sub in hippo_signals_and_psd.keys():
        all_sub_window_data = defaultdict(list)
        all_sub_filtered_window_data = defaultdict(list)
        for ses in hippo_signals_and_psd[sub].keys():
            for method in hippo_signals_and_psd[sub][ses].keys():
                for electrode in hippo_signals_and_psd[sub][ses][method].keys():
                    all_sub_window_data[method].extend(hippo_signals_and_psd[sub][ses][method][electrode]['window_data'])
                    all_sub_filtered_window_data[method].extend(
                        hippo_signals_and_psd[sub][ses][method][electrode]['filtered_window_data'])

        raw_erp_per_method = {}
        for method in all_sub_window_data.keys():
            all_sub_window_data[method] = [data for data in all_sub_window_data[method] if len(data) in [300,600, 614]]
            raw_erp_per_method[method] = [np.mean(all_sub_window_data[method], axis=0), len(all_sub_window_data[method])]

        filtered_erp_per_method = {}
        for method in all_sub_filtered_window_data.keys():
            all_sub_filtered_window_data[method] = [data for data in all_sub_filtered_window_data[method] if len(data) in [300,600, 614]]
            filtered_erp_per_method[method] = [np.mean(all_sub_filtered_window_data[method], axis=0), len(all_sub_filtered_window_data[method])]

        sub_dict_raw[sub] = raw_erp_per_method
        sub_dict_filtered[sub] = filtered_erp_per_method
    # create figure for each of the dict, where each subplot is a method-patient
    for sub_dict, data_type in zip([sub_dict_raw, sub_dict_filtered], ['raw', 'filtered']):
        fig, axs = plt.subplots(14, len(methods_to_run), figsize=(40, 75)) #figsize=(40, 75) good for 15
        for sub_idx, sub in enumerate(sub_dict.keys()):
            for i, (method, (erp, n)) in enumerate(sub_dict[sub].items()):
                row = loc_dict[sub]
                col = loc_dict[method]
                axs[row, col].plot(erp)
                # axs[row, col].set_title(f'{sub} - {method} - {data_type}', fontsize=16)
                axs[row, col].set_title(f'{sub} - {method} - {data_type} (n={n})',
                                        fontsize=16)
        plt.suptitle(f'{data_type} ERP per Patient per Method', fontsize=18, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, f'{data_type}_erp_per_patient_per_method.png'))
        plt.close()

def plot_corr_results(results, data_type):
    rows = []
    for patient, methods in results.items():
        for method, (mean, std) in methods.items():
            if not isinstance(mean, np.float64):
                # Create a dictionary for each row
                if 'window' in data_type:
                    row = {
                        'patient': patient,
                        'method': method,
                        'cross_correlation_mean': mean[0],
                        'cross_correlation_std': std[0],
                        'MI_mean': mean[1],
                        'MI_std': std[1],
                        'cosine_similarity_mean': mean[3],
                        'cosine_similarity_std': std[3],
                        'pearson_correlation_mean': mean[4],
                        'pearson_correlation_std': std[4],
                    }
                else:
                    row = {
                        'patient': patient,
                        'method': method,
                        'cross_correlation_mean': mean[0],
                        'cross_correlation_std': std[0],
                        'MI_mean': mean[1],
                        'MI_std': std[1],}
                # Append the row to the list
                rows.append(row)
    results_df = pd.DataFrame(rows)

    # create bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    df_melted = results_df.melt(id_vars=['method'], value_vars=[col for col in results_df.columns if "mean" in col],
                                var_name="Metric", value_name="Value")

    sns.barplot(data=df_melted, x="method", y="Value", hue="Metric", palette=combined_palette, ax=ax)
    title = f"Mean Similarity between Ripples by Method" if 'filtered' not in data_type else f"Mean Similarity between Filtered Ripples by Method"
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Method")
    ax.set_xticklabels(results_df['method'].unique(), rotation=45, ha='right')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Cross Correlation', 'Mutual Information', 'Cosine Similarity', 'Pearson Correlation'],
              title='Metric')
    plt.savefig(os.path.join(figures_path, f"mean_similarity_between_ripples_by_method_{data_type}.png"))



