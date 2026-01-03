import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
events_files_path = '/Users/rinatsaban/Library/CloudStorage/OneDrive-mail.tau.ac.il/TAU/master/events'
start_of_trails_path = '/Users/rinatsaban/Library/CloudStorage/OneDrive-mail.tau.ac.il/TAU/master/start_of_trails'
figures_path = '/Users/rinatsaban/Library/CloudStorage/OneDrive-mail.tau.ac.il/TAU/master/docs for writing/figures'

tab20_colors = list(plt.cm.tab20.colors)
tab20b_colors = list(plt.cm.tab20b.colors)
combined_palette = tab20_colors + tab20b_colors


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- 1. Accuracy per patient ----
    acc_patient = (
        df.groupby('patient')['Correct']
          .mean()
          .rename('accuracy_all')
    )

    # ---- 2. Accuracy per patient × set size ----
    acc_setsize = (
        df.groupby(['patient', 'SetSize'])['Correct']
          .mean()
          .unstack()  # columns: set sizes
          .add_prefix('acc_set_')
    )

    # ---- 3. Cowan’s K (if TrialType exists) ----
    def compute_k(sub):
        k_vals = {}
        for s, g in sub.groupby('SetSize'):
            hits = g.loc[g['Match'] == 'IN',  'Correct'].mean()
            cr   = g.loc[g['Match'] == 'OUT', 'Correct'].mean()
            k_vals[f'K_{s}'] = (hits + cr - 1) * s
        # average K for high load (6 & 8) if available
        high = [v for k, v in k_vals.items() if k in ['K_6', 'K_8']]
        if high:
            k_vals['K_highload_mean'] = sum(high) / len(high)
        return pd.Series(k_vals)

    if 'Match' in df.columns:
        k_df = df.groupby('patient').apply(compute_k)
    else:
        k_df = pd.DataFrame(index=acc_patient.index)

    # ---- 4. Mean RT for correct trials (optional) ----
    if 'ResponseTime' in df.columns:
        rt = (
            df[df['Correct'] == 1]
            .groupby('patient')['ResponseTime']
            .mean()
            .rename('mean_RT_correct')
        )
    else:
        rt = pd.Series(index=acc_patient.index, name='mean_RT_correct')

    # ---- Combine everything ----
    final = pd.concat([acc_patient, acc_setsize, k_df, rt], axis=1).reset_index()

    return final

def plot_patient_tests(df, save_fname='patient_tests_by_patient.png', figsize=(10, 6), columns=[]):
    """
    Plot each column (test/method) from a dataframe where each row is a patient.
    - df: pandas.DataFrame with rows = patients (index or first column) and columns = test results
    - save_fname: file name to save under `figures_path`
    """
    # Ensure index are strings for plotting ticks
    x_labels = df.index.astype(str).tolist()

    plt.figure(figsize=figsize)
    for col, color in zip(columns, combined_palette):
        vals = df[col].values.astype(float)
        plt.plot(x_labels, vals, marker='o', color=color,
                 label=f"{col} - Mean: {np.nanmean(vals):.2f}")

    plt.title('Test results by Patient')
    plt.xlabel('Patient')
    plt.ylabel('Score')
    plt.xticks(rotation=45, fontsize=8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()

    # Save to `figures_path`
    plt.savefig(os.path.join(figures_path, save_fname))
    plt.close()


def plot_hist_per_window_size(df, method, start_of_trails_path, figures_path, window_sizes_msec=[200, 300, 500],
                              hop_size_msec=100):
    """
    Create histograms of number of ripples per window size, aggregated across electrodes.
    Ripples with at least 10ms overlap are considered the same ripple.
    """
    n_windows = len(window_sizes_msec)
    fig, axes = plt.subplots(1, n_windows, figsize=(n_windows * 6, 5))

    for idx, window_size_msec in enumerate(window_sizes_msec):
        window_size_sec = window_size_msec / 1000.0
        hop_size_sec = hop_size_msec / 1000.0

        all_counts = []

        for (patient_id, session_id), group_idx in df.groupby(["patient", "session"]).groups.items():
            g = df.loc[group_idx]

            npy_path = os.path.join(
                start_of_trails_path,
                f"{patient_id}_{session_id}_start_of_trails_BP.npy"
            )

            try:
                trial_starts = np.load(npy_path)
            except FileNotFoundError:
                continue

            trial_starts = np.asarray(trial_starts, dtype=float)

            if trial_starts.size == 0:
                continue

            trial_starts.sort()

            n_trials = len(trial_starts)
            recording_sec = n_trials * 8.0

            n_windows_count = int(np.floor((recording_sec - window_size_sec) / hop_size_sec)) + 1

            # Store ripple intervals per window
            ripples_per_window = [[] for _ in range(n_windows_count)]

            ripple_starts = g["start"].to_numpy(dtype=float)
            ripple_ends = g["end"].to_numpy(dtype=float)

            # Assign each ripple to windows it overlaps with
            for start, end in zip(ripple_starts, ripple_ends):
                # Check all windows this ripple might overlap with
                for window_idx in range(n_windows_count):
                    window_start = window_idx * hop_size_sec
                    window_end = window_start + window_size_sec

                    # Check if ripple overlaps with this window
                    overlap_start = max(window_start, start)
                    overlap_end = min(window_end, end)

                    if overlap_end > overlap_start:  # There is overlap
                        ripples_per_window[window_idx].append((start, end))

            # Count unique ripples per window (merge if overlap >= 10ms)
            counts_per_window = []
            for ripple_intervals in ripples_per_window:
                if len(ripple_intervals) == 0:
                    counts_per_window.append(0)
                    continue

                # Sort by start time
                sorted_ripples = sorted(ripple_intervals, key=lambda x: x[0])

                # Merge overlapping ripples (overlap >= 10ms = 0.01 sec)
                merged = [sorted_ripples[0]]
                for curr_start, curr_end in sorted_ripples[1:]:
                    last_start, last_end = merged[-1]
                    distance = curr_start - last_end

                    if distance <= 0.01:  # 10ms distance
                        # Merge: extend the last interval
                        merged[-1] = (last_start, max(last_end, curr_end))
                    else:
                        # No significant overlap: add as new ripple
                        merged.append((curr_start, curr_end))

                counts_per_window.append(len(merged))

            all_counts.extend(counts_per_window)

        # Plot histogram for this window size
        ax = axes[idx]
        if len(all_counts) > 0:
            max_count = max(all_counts)
            n, bins, patches = ax.hist(all_counts, bins=range(0, max_count + 2), align='left',
                                       color='skyblue', edgecolor='black')

            # Add count labels on top of bars
            for i, (count, patch) in enumerate(zip(n, patches)):
                if count > 0:  # Only label non-zero bars
                    height = patch.get_height()
                    ax.text(patch.get_x() + patch.get_width() / 2., height,
                            f'{int(count)}',
                            ha='center', va='bottom', fontsize=9)

            ax.set_xticks(range(0, max_count + 1))
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)

        ax.set_title(f'{window_size_msec}ms Window', fontsize=13)
        ax.set_xlabel('Number of Ripples', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=11)

    fig.suptitle(f'Ripple Count Distribution by Window Size - Method: {method}',
                 fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(figures_path, f'ripple_count_hist_{method}_combined.png'))
    plt.close()








def compute_ripple_rate_per_electrode_session(
    df: pd.DataFrame,
    start_of_trails_path: str,
    method: str,
    trial_duration_sec: float = 8.0,
) -> pd.DataFrame:
    """
    df: DataFrame with columns:
        ["ripple_start", "peak", "ripple_end", "patient", "electrode", "session"]
        'peak' must be in seconds in the same time base as trial starts.
    start_of_trails_path: directory containing npy files named
        "{patient_id}_{session_id}_start_of_trails_BP.npy"
    trial_duration_sec: duration of each trial in seconds (default 8s).

    Returns:
        DataFrame with one row per (patient, session, electrode):
        ["patient", "session", "electrode", "n_ripples",
         "recording_sec", "rate_per_sec"]
    """

    df = df.copy()

    # ---- 1. Mark ripples that fall in valid trials ----
    valid_masks = []
    recording_sec_by_ps = {}  # (patient, session) -> total valid recording seconds

    for (patient_id, session_id), idx in df.groupby(["patient", "session"]).groups.items():
        g = df.loc[idx]

        if method not in ['norman', 'staresina']:
            npy_path = os.path.join(
                start_of_trails_path,
                f"{patient_id}_{session_id}_start_of_trails_BP.npy"
            )
        elif method == 'norman':
            npy_path = os.path.join(
                start_of_trails_path,
                f"{patient_id}_{session_id}_start_of_trails_WM.npy"
            )
        elif method == 'staresina':
            npy_path = os.path.join(
                start_of_trails_path,
                f"{patient_id}_{session_id}_start_of_trails_BP_mas.npy"
            )

        try:
            trial_starts = np.load(npy_path)
        except FileNotFoundError:
            # No trial info for this patient-session
            valid_masks.append(pd.Series(False, index=idx))
            recording_sec_by_ps[(patient_id, session_id)] = 0.0
            continue

        trial_starts = np.asarray(trial_starts, dtype=float)

        if trial_starts.size == 0:
            valid_masks.append(pd.Series(False, index=idx))
            recording_sec_by_ps[(patient_id, session_id)] = 0.0
            continue

        trial_starts.sort()
        recording_sec_by_ps[(patient_id, session_id)] = (
            len(trial_starts) * trial_duration_sec
        )

        # ripple peak times (seconds)
        t = g["peak"].to_numpy(dtype=float)

        # For each ripple time t, find the last trial_start <= t
        pos = np.searchsorted(trial_starts, t, side="right") - 1

        in_window = np.zeros_like(t, dtype=bool)
        valid_pos = pos >= 0
        if valid_pos.any():
            ts = trial_starts[pos[valid_pos]]
            in_window[valid_pos] = (
                (t[valid_pos] >= ts) &
                (t[valid_pos] < ts + trial_duration_sec)
            )

        valid_masks.append(pd.Series(in_window, index=idx))

    valid_mask = pd.concat(valid_masks).sort_index()
    df_valid = df[valid_mask].copy()

    # If no valid ripples at all, return empty with expected columns
    if df_valid.empty:
        return pd.DataFrame(
            columns=["patient", "session", "electrode",
                     "n_ripples", "recording_sec", "rate_per_min"]
        )

    # ---- 2. Map recording duration per patient-session ----
    df_valid["ps_key"] = list(zip(df_valid["patient"], df_valid["session"]))
    df_valid["recording_sec"] = df_valid["ps_key"].map(recording_sec_by_ps)

    # ---- 3. Aggregate per (patient, session, electrode) ----
    agg = (
        df_valid
        .groupby(["patient", "session", "electrode"], as_index=False)
        .agg(
            n_ripples=("peak", "size"),
            recording_sec=("recording_sec", "first"),
        )
    )

    # Guard against zero recording_sec (should be rare if any valid ripple exists)
    agg = agg[agg["recording_sec"] > 0].copy()

    # ---- 4. Compute ripple rate (events per minute) ----
    agg["rate_per_sec"] = agg["n_ripples"] / (agg["recording_sec"])
    return agg

def compute_patient_level_rate_per_second(agg: pd.DataFrame) -> pd.DataFrame:
    """
    agg must contain:
    - patient
    - n_ripples
    - recording_sec  (in seconds)
    """

    patient_rate = (
        agg.groupby("patient")
        .apply(lambda g: pd.Series({
            "total_ripples": g["n_ripples"].sum(),
            "total_recording_sec": g["recording_sec"].sum(),
            "rate_per_sec": g["n_ripples"].sum() / g["recording_sec"].sum(),
            "n_electrodes_used": g["electrode"].nunique(),
            "n_sessions_used": g["session"].nunique(),
        }))
        .reset_index()
    )

    return patient_rate


def compute_patient_level_rate_with_sides(agg: pd.DataFrame) -> pd.DataFrame:
    """
    agg columns required:
      - patient
      - electrode_side  ('L' or 'R')
      - n_ripples
      - recording_sec   (in seconds)

    Returns one row per patient with:
      - total_ripples
      - total_recording_sec
      - rate_overall_per_sec
      - rate_L_per_sec
      - rate_R_per_sec
      - n_electrodes_used
      - n_sessions_used
    """

    def per_patient(g: pd.DataFrame) -> pd.Series:
        # Overall
        total_ripples = g["n_ripples"].sum()
        total_rec_sec = g["recording_sec"].sum()
        rate_overall = (
            total_ripples / total_rec_sec if total_rec_sec > 0 else np.nan
        )

        # Left side
        gL = g[g["electrode_side"] == "L"]
        total_ripples_L = gL["n_ripples"].sum()
        total_rec_sec_L = gL["recording_sec"].sum()
        rate_L = (
            total_ripples_L / total_rec_sec_L
            if total_rec_sec_L > 0 else np.nan
        )

        # Right side
        gR = g[g["electrode_side"] == "R"]
        total_ripples_R = gR["n_ripples"].sum()
        total_rec_sec_R = gR["recording_sec"].sum()
        rate_R = (
            total_ripples_R / total_rec_sec_R
            if total_rec_sec_R > 0 else np.nan
        )

        return pd.Series({
            "total_ripples": total_ripples,
            "total_recording_sec": total_rec_sec,
            "rate_overall_per_sec": rate_overall,
            "rate_L_per_sec": rate_L,
            "rate_R_per_sec": rate_R,
            "n_electrodes_used": g["electrode"].nunique(),
            "n_sessions_used": g["session"].nunique(),
        })
    if len(agg)>0:
        patient_rate = (
            agg
            .groupby("patient")
            .apply(per_patient)
            .reset_index()
        )
    else:
        patient_rate = pd.DataFrame(
            columns=[
                "patient",
                "total_ripples",
                "total_recording_sec",
                "rate_overall_per_sec",
                "rate_L_per_sec",
                "rate_R_per_sec",
                "n_electrodes_used",
                "n_sessions_used",
            ]
        )

    return patient_rate


def plot_ripple_rate_by_method_boxplot(all_methods_all_patients_level_rates, methods_to_run, figures_path):
    # Pivot to wide format
    wide = (
        all_methods_all_patients_level_rates
        .pivot(index="method", columns="patient", values="rate_overall_per_sec")
        .reset_index()
    )

    # Melt to long format
    df_melted = wide.melt(
        id_vars=["method"],
        value_vars=[c for c in wide.columns if c != "method"],
        var_name="Subject",
        value_name="RippleRate"
    )

    df_melted["RippleRate"] = df_melted["RippleRate"].fillna(0)

    plt.figure(figsize=(12, 6))

    ax = sns.boxplot(
        data=df_melted,
        x="RippleRate",
        y="method",
        palette=combined_palette,  # assumes you defined this
        order=methods_to_run
    )

    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

    plt.title("Ripple-rate Distribution by Detection Method", fontdict={"fontsize": 16})
    plt.xlabel("Ripple rate (events/sec)")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, "ripple_rate_by_method_boxplot.png"))
    plt.close()

def plot_ripple_rate_L_R_boxplot(all_methods_all_patients_level_rates, methods_to_run, figures_path):
    df = all_methods_all_patients_level_rates.copy()
    # Keep only rows with relevant metrics (avoid KeyError if some metrics missing)
    for col in ['rate_R_per_sec', 'rate_L_per_sec']:
        if col not in df.columns:
            df[col] = np.nan

    long = pd.melt(
        df,
        id_vars=['method', 'patient'],
        value_vars=['rate_R_per_sec', 'rate_L_per_sec'],
        var_name='metric',
        value_name='value'
    )

    # Drop missing values so boxplot isn't cluttered with NaNs
    long = long.dropna(subset=['value'])

    # Map metric keys to nicer labels
    metric_labels = {
        'rate_R_per_sec': 'R rate (events/sec)',
        'rate_L_per_sec': 'L rate (events/sec)',
    }
    long['metric_label'] = long['metric'].map(metric_labels)

    plt.figure(figsize=(max(10, len(methods_to_run) * 1.8), 7))

    # Use hue to get three boxes per method (grouped)
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
    hue_order = ['R rate (events/sec)', 'L rate (events/sec)']
    ax = sns.boxplot(
        data=long,
        x='value',
        y='method',
        hue='metric_label',
        order=methods_to_run,
        hue_order=hue_order,
        palette=palette
    )

    # Styling
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    plt.title("R / L Ripple-rate by Detection Method", fontsize=16)
    plt.xlabel("Value", fontsize=13)
    plt.ylabel("Method", fontsize=13)
    plt.legend(title='', fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, "ripple_rate_L_R_by_method_boxplot.png"))
    plt.close()




def plot_ripple_rate_vs_task_score(all_methods_all_patients_level_rates, final, figures_path, size_comparison=False):
    merged = all_methods_all_patients_level_rates.merge(final, on='patient', how='left')
    # create a subplot grid for each method
    methods = merged['method'].unique()
    n_methods = len(methods)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    # smaller per-subplot size but overall readable fonts: ~5x4 per subplot
    fig = plt.figure(figsize=(n_cols * 5, max(4, n_rows * 4)))

    for i, method in enumerate(methods):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        data = merged[merged['method'] == method]

        # extract x and y and drop NaNs
        if size_comparison:
            x = (data['rate_L_per_sec'] - data['rate_R_per_sec']).to_numpy(dtype=float)
        else:
            x = data['rate_overall_per_sec'].to_numpy(dtype=float)
        y = data['accuracy_all'].to_numpy(dtype=float)
        valid = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid]
        y = y[valid]

        if x.size == 0:
            ax.set_title(f'No data - {method}', fontsize=12)
            continue

        # Plot original scatter (do not center so axes keep original values)
        sns.scatterplot(x=x, y=y, s=70, edgecolor='k', linewidth=0.5, ax=ax)

        # fit line on original data (y = m * x + c)
        if x.size >= 2:
            m, c = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 200)
            y_line = m * x_line + c
            ax.plot(x_line, y_line, color='red', linestyle='--', linewidth=2.2, label=f'fit: m={m:.3f}')
            # annotate slope with larger font in axes coordinates
            ax.legend(loc='best', fontsize=10)
        else:
            ax.annotate('insufficient points for fit', xy=(0.05, 0.95), xycoords='axes fraction',
                         ha='left', va='top', fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.7))

        # keep original scales, increase font sizes
        ax.set_title(f'{method}', fontsize=13)
        if size_comparison:
            ax.set_xlabel('Ripple Rate L - R (events/sec)', fontsize=12)
        else:
            ax.set_xlabel('Ripple Rate (events/sec)', fontsize=12)
        ax.set_ylabel('Task Accuracy', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=11)

    # make sure the suptitle is visible by reserving top margin
    if size_comparison:
        fig.suptitle('Ripple Rate L - R vs. Task Accuracy by Detection Method', fontsize=18, y=0.95)
    else:
        fig.suptitle('Ripple Rate vs. Task Accuracy by Detection Method', fontsize=18, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if size_comparison:
        plt.savefig(os.path.join(figures_path, 'ripple_rate_L_minus_R_vs_task_accuracy_by_method.png'))
    else:
        plt.savefig(os.path.join(figures_path, 'ripple_rate_vs_task_accuracy_by_method.png'))
    plt.close()
