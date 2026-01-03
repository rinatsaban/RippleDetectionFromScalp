from functools import partial
import os
import numpy as np
from scipy.signal import remez, firwin, cheby1, filtfilt, hilbert, kaiserord, find_peaks, butter, firls, convolve
from scipy.stats import lognorm, gamma, skew, kurtosis, percentileofscore
from scipy.ndimage import label, binary_opening
import pandas as pd
import matplotlib.pyplot as plt
from general.configs import *
from scipy.stats import median_abs_deviation

projects_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
dataset_path = os.path.join(projects_path, 'ds004752-download')
figures_path = os.path.join(dataset_path, 'figures')

class RipplesDetection:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def ripple_detection(self):
        raise NotImplementedError("Subclasses must implement this method")

    def filter_signal(self, signal, band, N, filter_type, fs=500, btype=None):
        if filter_type == 'hamming':
            filter = firwin(N + 1, band, pass_zero=False, window='hamming', fs=fs)
            filtered_signal = filtfilt(filter, 1, signal.astype(float))
        elif filter_type == 'butter':
            b, a = butter(N=N, Wn=band, btype='bandpass', fs=fs, output='ba')
            filtered_signal = filtfilt(b, a, signal.astype(float))
        elif filter_type == 'chebyshev':
            if btype == 'band':
                b, a = cheby1(N=N, rp=0.5, Wn=[band[0] / (fs / 2), band[1] / (fs / 2)], btype=btype, analog=False)
            elif btype == 'high':
                b, a = cheby1(N=N, rp=0.5, Wn=band[0] / (fs / 2), btype=btype, analog=False)
            elif btype == 'low':
                b, a = cheby1(N=N, rp=0.5, Wn=band[0] / (fs / 2), btype=btype, analog=False)
            filtered_signal = filtfilt(b, a, signal.astype(float))

        return filtered_signal

    def IED_detection(self, hp_channel, filter_partial_function, std_threshold, fs, method):
        filtered_signal = filter_partial_function(hp_channel)
        absolute_differences = np.abs(np.diff(filtered_signal))

        if method == 'vaz':
            abs_filtered_signal = np.abs(filtered_signal)
            threshold_1 = np.mean(abs_filtered_signal) + std_threshold * np.std(abs_filtered_signal)
            threshold_2 = np.mean(absolute_differences) + std_threshold * np.std(absolute_differences)
            signal_1 = filtered_signal
            signal_2 = absolute_differences

        elif method == 'skelin':
            upper_envelope = np.abs(hilbert(filtered_signal))
            threshold_1 = np.mean(upper_envelope) + std_threshold * np.std(upper_envelope)
            threshold_2 = np.mean(absolute_differences) + std_threshold * np.std(absolute_differences)
            signal_1 = upper_envelope
            signal_2 = absolute_differences

        start_indices_1, end_indices_1 = self.events_exceed_th(signal_1, threshold_1)
        start_indices_2, end_indices_2 = self.events_exceed_th(signal_2, threshold_2)
        start_indices = np.concatenate([start_indices_1, start_indices_2])
        end_indices = np.concatenate([end_indices_1, end_indices_2])

        start_times = start_indices / fs
        end_times = end_indices / fs
        return start_times, end_times

    def z_score(self, signal):
        return (signal - np.mean(signal)) / np.std(signal)

    def events_exceed_th(self, signal, th):
        event_detected = signal > th
        start_indices = np.where(np.diff(event_detected.astype(int)) == 1)[0]
        end_indices = np.where(np.diff(event_detected.astype(int)) == -1)[0]
        if len(start_indices) == 0 or len(end_indices) == 0:
            return np.array([]), np.array([])
        if end_indices[0] < start_indices[0]:
            end_indices = end_indices[1:]
        if len(start_indices) > len(end_indices):
            start_indices = start_indices[:-1]
        assert len(start_indices) == len(end_indices)
        return start_indices, end_indices

    def log(self, msg):
        if self.verbose:
            print(msg)

    def plot_response(self, w, h, title):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(w, 20 * np.log10(np.abs(h)))
        print(w[np.argmax(20 * np.log10(np.abs(h)))]/1000)
        ax.set_ylim(-40, 5)
        ax.grid(True)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain (dB)')
        ax.set_title(title)
        fig.savefig(os.path.join(figures_path, title + '.png'))

    def filter_clean_trails(self, ripples_df_or_idx, clean_trails_start, trails_duration=8):
        if isinstance(ripples_df_or_idx, pd.DataFrame):
            t_peak = ripples_df_or_idx['peak'].to_numpy()
        else:
            t_peak = np.array(ripples_df_or_idx)
        pos = np.searchsorted(clean_trails_start, t_peak, side="right") - 1

        in_window = np.zeros_like(t_peak, dtype=bool)
        valid_pos = pos >= 0
        if valid_pos.any():
            ts = clean_trails_start[pos[valid_pos]]
            in_window[valid_pos] = (
                    (t_peak[valid_pos] >= ts) &
                    (t_peak[valid_pos] < ts + trails_duration)
            )
        if isinstance(ripples_df_or_idx, pd.DataFrame):
            output = ripples_df_or_idx.iloc[in_window]
        else:
            output = ripples_df_or_idx[in_window]
        return output

    def get_only_clean_trails(self, signal, start_of_clean_trails, fs, trails_duration=8):
        clean_signal = np.hstack(
            [np.array(signal[int(start_of_clean_trail * fs):int((start_of_clean_trail + trails_duration) * fs)]) for
             start_of_clean_trail in start_of_clean_trails])
        return clean_signal


class NormanRipplesDetectionMethod(RipplesDetection):
    def __init__(self, hp_channel_section, mean_LFP_section, fs, start_of_sections, th = [2, 4, 4], ripple_band = [70, 180], IED_band = [25, 60], verbose=True):
        super().__init__(verbose=verbose)
        self.hp_channel_section = hp_channel_section
        self.mean_LFP_section = mean_LFP_section
        self.start_of_sections = start_of_sections
        self.fs = fs
        # ripple detection parameters for Norman method
        self.th = th # Thresholds in stdev [onset/offset peak/noise peak] [minimal SD for ripple extenstion, minimal SD for clipping and detection, minimal SD for noise rejection]
        self.minDistance = 0.030  # in sec
        self.minRippleDuration = 0.020  # in sec
        self.maxRippleDuration = 0.200  # in sec
        # self.negative_threshold = 4  # in std
        # self.minDistance_negative = 0.2  # in sec
        self.ripple_band = ripple_band
        self.IED_band = IED_band

    def filter_ripple_IED_bands(self, raw_data_filtered):
        transition_bw = 5  # Transition bandwidth in Hz
        N = int(np.ceil((3.3 * self.fs / transition_bw) / 2) * 2 - 1)  # Filter order

        full_data_filtered_ripples = self.filter_signal(raw_data_filtered, self.ripple_band, N, 'hamming')
        full_data_filtered_IED = self.filter_signal(raw_data_filtered, self.IED_band, N, 'hamming')

        return full_data_filtered_ripples, full_data_filtered_IED

    def processing_data_for_ripples_detection(self, signalA, signalB, signalC):
        """

        :param signalA: hippocampus electrode in the ripple band [70 180] Hz
        :param signalB: mean LFP signal
        :param signalC: hippocampus electrode in the IED band [25 60] Hz
        :return:
        """
        # Hilbert envelope
        absSignal = np.abs(hilbert(signalA))
        # Clipping the signal:
        topLim = np.nanmean(absSignal) + self.th[1] * np.nanstd(absSignal)
        absSignal[absSignal > topLim] = topLim
        # Squaring:
        squaredSignal = absSignal ** 2

        # Smoothing using a lowpass filter:
        LPcutoff = int(np.round(np.mean(self.ripple_band) / np.pi))

        # FIR kaiserwin lowpass filter
        width = 10 / (2 * self.fs)
        ripple_db = 60  # dB, attenuation
        N, beta = kaiserord(ripple_db, width)
        lpFilt = firwin(N, LPcutoff / (self.fs / 2), window=('kaiser', beta))

        # Filter squared signal
        squaredSignal = filtfilt(lpFilt, 1, squaredSignal)

        # get only clean trails for avg and std
        cleanSquaredSignal = self.get_only_clean_trails(squaredSignal, fs=self.fs, start_of_clean_trails=self.start_of_sections)

        # Compute means and std:
        avg = np.nanmean(cleanSquaredSignal)
        stdev = np.nanstd(cleanSquaredSignal)

        # Hilbert envelope for other signals
        absSignalA = np.abs(hilbert(signalA))
        absSignalB = np.abs(hilbert(signalB))
        absSignalC = np.abs(hilbert(signalC))

        # Squaring the signals
        squaredSignalA = absSignalA ** 2
        squaredSignalB = absSignalB ** 2
        squaredSignalC = absSignalC ** 2

        # Filtering squared signals
        squaredSignalA = filtfilt(lpFilt, 1, squaredSignalA)
        squaredSignalB = filtfilt(lpFilt, 1, squaredSignalB)
        squaredSignalC = filtfilt(lpFilt, 1, squaredSignalC)

        # Z-scoring
        squaredSignalNormA = (squaredSignalA - avg) / stdev
        squaredSignalNormB = (squaredSignalB - np.nanmean(squaredSignalB)) / np.nanstd(squaredSignalB)
        squaredSignalNormC = (squaredSignalC - np.nanmean(squaredSignalC)) / np.nanstd(squaredSignalC)

        squaredSignalNormB = squaredSignalNormB[:len(squaredSignalNormA)]

        return squaredSignalNormA, squaredSignalNormB, squaredSignalNormC

    def ripples_detection_excluding_IED(self, signal, BP, noise_ch, IED_ch, start_of_section):
        t = np.arange(0, len(BP) / self.fs, 1 / self.fs)  # time vector in sec

        # Ensure row vector orientation
        signal = np.ravel(signal)
        noise_ch = np.ravel(noise_ch)
        IED_ch = np.ravel(IED_ch)

        # Finding peaks with height greater than th[1]
        all_pks, properties = find_peaks(signal, height=self.th[1])
        pks = all_pks #[properties['peak_heights'] <= th[2]]

        # Calculate envelope and normalize
        ENV = np.abs(hilbert(BP))
        ENV = 10 * np.log10(ENV / np.nanmedian(ENV))  # Ripple amplitude in dB relative to median

        # Noise rejection
        if noise_ch.size > 0:
            noise_idx, noise_prop = find_peaks(noise_ch, height=self.th[2])
            len_before_noise = len(pks)
            for i in range(len(noise_idx)):
                tmp = np.where(abs(pks - noise_idx[i]) < (0.05 * self.fs))
                if len(tmp)>0:
                    pks = np.delete(pks, tmp)

            self.log(f' *** rejected {len_before_noise - len(pks)} / {len_before_noise} events based on noise channel correlation ***')

        # IED rejection
        if IED_ch.size > 0:
            IED_idx, IED_prop = find_peaks(IED_ch, height=self.th[2])
            len_before_IED = len(pks)
            for i in range(len(IED_idx)):
                tmp = np.where(abs(pks - IED_idx[i]) < (0.05 * self.fs))
                if len(tmp) > 0:
                    pks = np.delete(pks, tmp)
            self.log(f' *** rejected {len_before_IED-len(pks)} / {len_before_IED} events based on IED channel correlation ***')

        # Find start and end points of ripples
        ripples = []
        for pk in pks:
            start = pk
            while start > 0 and signal[start] > self.th[0]:
                start -= 1
            end = pk
            while end < len(signal)-1 and signal[end] > self.th[0]:
                end += 1

            # Calculating ripple peak position based on negative peaks within start to end
            segment_BP = BP[start:end]
            min_indices, _ = find_peaks(-segment_BP)
            if len(min_indices) > 0:
                peak_position = min_indices[np.argmin(np.abs(min_indices - (pk - start)))] + start
            else:
                continue

            if start != pk and end != pk:
                ripples.append([t[start], t[peak_position], t[end], ENV[peak_position], np.max(signal[start:end+1])-self.th[1]])

        ripples_df = pd.DataFrame(ripples, columns=['start', 'peak', 'end', 'amplitude', 'confidence']) #confidence is the trigger amplitude

        self.log(f' *** Number of ripples before filtering by duration: {len(ripples_df)} ***')

        # Merging close ripples
        if len(ripples_df) == 0:
            return ripples_df

        gap = ripples_df['start'].shift(-1) - ripples_df['end']
        merged_ripples = []
        current_start = ripples_df.iloc[0]['start']
        current_end = ripples_df.iloc[0]['end']
        current_peak = ripples_df.iloc[0]['peak']
        current_confidence = ripples_df.iloc[0]['confidence']

        for i in range(1, len(ripples_df)):
            if gap.iloc[i - 1] <= self.minDistance:
                current_end = ripples_df.iloc[i]['end']
                current_peak = ripples_df.iloc[i - np.argmax([ripples_df.iloc[i]['amplitude'], ripples_df.iloc[i-1]['amplitude']])]['peak']
                current_confidence = np.mean(ripples_df.iloc[[i, i-1]]['confidence'])
            else:
                merged_ripples.append((current_start, current_peak, current_end, current_confidence))
                current_start = ripples_df.iloc[i]['start']
                current_peak = ripples_df.iloc[i]['peak']
                current_end = ripples_df.iloc[i]['end']
                current_confidence = ripples_df.iloc[i]['confidence']
        merged_ripples.append((current_start, current_peak, current_end, current_confidence))
        ripples_df = pd.DataFrame(merged_ripples, columns=['start', 'peak', 'end', 'confidence'])

        # Filter by duration
        ripples_df = ripples_df[(ripples_df['end'] - ripples_df['start'] >= self.minRippleDuration) &
                                (ripples_df['end'] - ripples_df['start'] <= self.maxRippleDuration)]

        # Filter to clean trails only:
        ripples_df = self.filter_clean_trails(ripples_df, start_of_section)

        # # find negative examples
        # if extract_negative:
        #     below_threshold = signal < negative_threshold
        #     indices_below_threshold = np.where(below_threshold)[0]
        #     continuous_segments = np.split(indices_below_threshold, np.where(np.diff(indices_below_threshold) != 1)[0] + 1)
        #     relevant_segments = [segment for segment in continuous_segments if len(segment) >= win_size*self.fs]
        #
        #     # Process each long segment to find valid sub-segments
        #     valid_sub_segments = []
        #     sub_segment_length = int(win_size*self.fs)  # Minimum sub-segment length
        #
        #     for segment in relevant_segments:
        #         start = segment[0]
        #         while start + sub_segment_length <= segment[-1]:
        #             end = start + sub_segment_length
        #
        #             # Check for overlap
        #             overlap = check_for_overlap(start / self.fs, end / self.fs, ripples_df, minDistance_negative)
        #
        #             if not overlap:
        #                 valid_sub_segments.append(range(start, end + 1))
        #                 start += sub_segment_length + 1  # Move start by sub_segment_length if no overlap
        #             else:
        #                 start += 1
        #
        #     valid_sub_segments.sort(key=lambda x: min(x))
        #
        #     negative_segments_data = [{'start': min(segment)/self.fs, 'end': max(segment)/self.fs} for segment in valid_sub_segments]
        #     negative_segments_df = pd.DataFrame(negative_segments_data)
        #     ripples_df.reset_index(inplace=True, drop=True)
        #     negative_segments_df.reset_index(inplace=True, drop=True)
        #     negative_segments_df['peak'] = (negative_segments_df['start'] + negative_segments_df['end']) / 2
        #
        #     self.log(f' *** Number of Negative Examples: {len(negative_segments_df)} ***')
        #
        # else:
        #     negative_segments_df = pd.DataFrame()

        self.log(f' *** final number of ripples: {len(ripples_df)} ***')

        return ripples_df

    def ripple_detection(self):
        full_data_filtered_ripples, full_data_filtered_IED = self.filter_ripple_IED_bands(self.hp_channel_section)
        squaredSignalNormA, squaredSignalNormB, squaredSignalNormC = self.processing_data_for_ripples_detection(full_data_filtered_ripples,
                                                                        self.mean_LFP_section, full_data_filtered_IED)
        ripples_df = self.ripples_detection_excluding_IED(squaredSignalNormA, full_data_filtered_ripples, squaredSignalNormB,
                                                                        squaredSignalNormC, self.start_of_sections)
        return ripples_df


# class LeonardHoffmanRipplesDetectionMethod(RipplesDetection):
#     def __init__(self, hp_channel_section, fs, start_of_section=0):
#         super().__init__()
#         self.hp_channel_section = hp_channel_section
#         self.start_of_section = start_of_section
#         self.fs = fs
#
#     def ripple_detection(self):
#         high_cutoff = min(250, self.fs/2 - 1)
#         ripple_band = [100, high_cutoff]
#         filtered_hippo_signal = self.filter_signal(self.hp_channel_section, ripple_band, 101, 'hamming', self.fs)
#         z_signal = self.z_score(filtered_hippo_signal)
#         rectified_signal = np.abs(z_signal)
#         event_signal = self.filter_signal(rectified_signal, [1, 20], 101, 'hamming')
#         threshold = 3
#         min_duration_samples = 5e-3 * self.fs
#
#         start_indices, end_indices = self.events_exceed_th(event_signal, threshold)
#
#         ripples = []
#         for start, end in zip(start_indices, end_indices):
#             if end - start >= min_duration_samples:
#                 if z_signal[start] >= 1 and z_signal[end] >= 1:
#                     ripples.append((start/self.fs, end/self.fs))  # in seconds
#         ripples_df = pd.DataFrame(ripples, columns=['start', 'end'])
#         ripples_df += self.start_of_section
#         self.log(f' *** Number of ripples: {len(ripples_df)} ***')
#         return ripples_df

class VazRipplesDetectionMethod(RipplesDetection):
    def __init__(self, hp_channel_section, fs, start_of_sections, verbose=True):
        super().__init__(verbose=verbose)
        self.hp_channel_section = hp_channel_section
        self.start_of_sections = start_of_sections
        self.fs = fs

    def ripple_detection(self):
        min_duration = 25e-3
        ripple_band = [80, 120]
        filtered_hippo_signal = self.filter_signal(self.hp_channel_section, ripple_band, 2, 'butter', self.fs)
        hilbert_hippo_signal = np.abs(hilbert(filtered_hippo_signal))
        # get only clean trails for mean and std calculation
        clean_hilbert_hippo_signal = self.get_only_clean_trails(hilbert_hippo_signal, fs=self.fs, start_of_clean_trails=self.start_of_sections)

        mean_amp = np.mean(clean_hilbert_hippo_signal)
        std_amp = np.std(clean_hilbert_hippo_signal)
        threshold_2 = mean_amp + 2 * std_amp
        threshold_3 = mean_amp + 3 * std_amp

        start_indices, end_indices = self.events_exceed_th(hilbert_hippo_signal, threshold_2)
        ripples = []
        for start, end in zip(start_indices, end_indices):
            if end - start >= min_duration * self.fs:
                if np.max(hilbert_hippo_signal[start:end]) > threshold_3:
                    peak = get_peak_index(hilbert_hippo_signal[start:end], self.hp_channel_section[start:end], start)
                    confidence = np.max(hilbert_hippo_signal[start:end]) - threshold_2
                    ripples.append((start, peak, end, confidence))  # in samples
        ripples_df = pd.DataFrame(ripples, columns=['start', 'peak', 'end', 'confidence'])
        len_before_merging = len(ripples_df)

        if len(ripples_df) == 0:
            return ripples_df

        # joined adjacent ripples that were separated by less than 15 ms
        gap = (ripples_df['start'].shift(-1) - ripples_df['end'])
        merged_ripples = []
        current_start = ripples_df.iloc[0]['start']
        current_end = ripples_df.iloc[0]['end']
        current_peak = ripples_df.iloc[0]['peak']
        current_confidence = ripples_df.iloc[0]['confidence']

        for i in range(1, len(ripples_df)):
            if gap.iloc[i - 1] <= 15e-3 * self.fs:
                current_end = ripples_df.iloc[i]['end']
                current_confidence = np.mean(ripples_df.iloc[[i, i-1]]['confidence'])
                current_peak = get_peak_index(hilbert_hippo_signal[int(current_start):int(current_end)], self.hp_channel_section[int(current_start):int(current_end)], int(current_start))
            else:
                merged_ripples.append((current_start, current_peak, current_end, current_confidence))
                current_start = ripples_df.iloc[i]['start']
                current_peak = ripples_df.iloc[i]['peak']
                current_end = ripples_df.iloc[i]['end']
                current_confidence = ripples_df.iloc[i]['confidence']
        merged_ripples.append((current_start, current_peak, current_end, current_confidence))
        ripples_df = pd.DataFrame(merged_ripples, columns=['start', 'peak', 'end', 'confidence'])
        ripples_df[['start', 'peak', 'end']] /= self.fs

        len_after_merging = len(ripples_df)
        self.log(f' *** rejected {len_before_merging-len_after_merging} / {len_before_merging} events after merging adjacent events ***')

        # exclude ripples that are tightly associated with IEDs
        filter_partial_function = partial(self.filter_signal, band=[250], N=4, filter_type='chebyshev', fs=self.fs, btype='high')
        start_times, end_times = self.IED_detection(self.hp_channel_section, filter_partial_function, std_threshold=5, fs=self.fs, method='vaz')
        ripples_df = ripples_df[~ripples_df.apply(lambda x: any([(x['start'] - 100e-3 >= end_time) & (x['end'] + 100e-3 <= start_time) for start_time, end_time in zip(start_times, end_times)]), axis=1)]
        self.log(f' *** rejected {len_after_merging - len(ripples_df)} / {len_after_merging} events based on IED rejection ***')
        self.log(f' *** Number of ripples: {len(ripples_df)} ***')

        ripples_df = self.filter_clean_trails(ripples_df, self.start_of_sections)

        return ripples_df


class StaresinaRipplesDetectionMethod(RipplesDetection):
    def __init__(self, hp_channel_section, fs, start_of_sections, verbose=True):
        super().__init__(verbose=verbose)
        self.hp_channel_section = hp_channel_section
        self.start_of_sections = start_of_sections
        self.fs = fs

    def extract_clean_rms_for_threshold(self, rms_signal, clean_trial_start_times,
                                        sampling_rate,
                                        rms_window_samples, trial_duration=8):
        # Convert trial times to sample indices in ORIGINAL signal space
        trial_start_samples = np.round(clean_trial_start_times * sampling_rate).astype(int)
        trial_duration_samples = int(trial_duration * sampling_rate)
        trial_end_samples = trial_start_samples + trial_duration_samples

        # Create a mask for the ORIGINAL signal indicating which samples are clean
        # We need to know the original signal length
        # Estimate from RMS signal length: original_length = rms_length + window_size - 1
        original_signal_length = len(rms_signal) + rms_window_samples - 1
        is_clean_sample = np.zeros(original_signal_length, dtype=bool)

        # Mark all samples within clean trials as clean
        for trial_start, trial_end in zip(trial_start_samples, trial_end_samples):
            # Ensure indices are within bounds
            trial_start = max(0, trial_start)
            trial_end = min(original_signal_length, trial_end)
            is_clean_sample[trial_start:trial_end] = True

        # Initialize mask for RMS signal
        clean_window_mask = np.zeros(len(rms_signal), dtype=bool)

        # For each RMS position, check if:
        # 1. Window STARTS in a clean trial (position i must be clean)
        # 2. Window does NOT extend into a noisy trial (all samples i to i+window must be clean)
        for i in range(len(rms_signal)):
            window_start = i
            window_end = i + rms_window_samples

            # Check if window start is in a clean trial
            if is_clean_sample[window_start]:
                # Check if ALL samples in the window are clean
                # (this prevents windows that start clean but extend into noisy trials)
                if np.all(is_clean_sample[window_start:window_end]):
                    clean_window_mask[i] = True

        # Extract clean RMS values
        clean_rms_values = rms_signal[clean_window_mask]

        return clean_rms_values

    def ripple_detection(self):
        ripple_band = [80, 100]
        filter_order = int(3 * self.fs / ripple_band[0])
        filtered_hippo = self.filter_signal(self.hp_channel_section, ripple_band, filter_order, 'hamming', self.fs)
        RMS_window = int(0.02 * self.fs)
        RMS_signal = np.sqrt(np.convolve(np.square(filtered_hippo), np.ones(RMS_window) / RMS_window, mode='valid'))
        # get new start of clean trails
        clean_filtered_hippo = self.extract_clean_rms_for_threshold(RMS_signal, sampling_rate=self.fs, rms_window_samples=RMS_window,
                                                                    clean_trial_start_times=self.start_of_sections)
        threshold = np.percentile(clean_filtered_hippo, 99)

        min_duration_samples = int(38e-3 * self.fs)
        start_indices, end_indices = self.events_exceed_th(RMS_signal, threshold)

        window_size = 3  # One adjacent points on each side + current point
        smoothed_signal = np.convolve(self.hp_channel_section, np.ones(window_size) / window_size, mode='valid')
        baseline_mad = median_abs_deviation(clean_filtered_hippo, scale='normal')

        ripples = []
        for start_rms, end_rms in zip(start_indices, end_indices):
            if end_rms - start_rms >= min_duration_samples:
                start_orig = start_rms
                end_orig = end_rms + RMS_window - 1
                start_smoothed = start_orig
                end_smoothed = end_orig - window_size + 1
                segment = smoothed_signal[start_smoothed:end_smoothed]
                peaks, _ = find_peaks(segment)
                troughs, _ = find_peaks(-segment)
                if len(peaks) >= 3 or len(troughs) >= 3:
                    peak = np.argmax(filtered_hippo[start_orig: end_orig+1])
                    peak = peak+start_orig
                    confidence_score = (np.max(RMS_signal[start_rms:end_rms]) - threshold) / baseline_mad
                    ripples.append((start_orig/self.fs, peak/self.fs, end_orig/self.fs, confidence_score)) # in seconds

        ripples_df = pd.DataFrame(ripples, columns=['start', 'peak', 'end', 'confidence'])
        len_before = len(ripples_df)

        filter_partial_function = partial(self.filter_signal, band=[250], N=4, filter_type='chebyshev', fs=self.fs, btype='high')
        start_times, end_times = self.artifacts_detection_staresina(self.hp_channel_section, filter_partial_function, self.fs)
        ripples_df = ripples_df[~ripples_df.apply(lambda x: any(
            [(x['start'] - 1000e-3 >= end_time) & (x['end'] + 1000e-3 <= start_time) for start_time, end_time in
             zip(start_times, end_times)]), axis=1)]
        self.log(f' *** Excluded: {len_before - len(ripples_df)}/{len_before} after artifacts rejection ***')
        self.log(f' *** Number of ripples: {len(ripples_df)} ***')
        if len(ripples_df)>0:
            ripples_df = self.filter_clean_trails(ripples_df, self.start_of_sections)
        else:
            return pd.DataFrame()
        return ripples_df

    def artifacts_detection_staresina(self, hp_channel, filter_partial_function, fs):
        # Step 1: Compute necessary signals and z-scores
        filtered_signal = filter_partial_function(hp_channel)
        absolute_differences = np.abs(np.diff(hp_channel))
        z_absolute_amplitude = self.z_score(np.abs(hp_channel))
        z_filtered_absolute_amplitude = self.z_score(np.abs(filtered_signal))
        z_absolute_differences = self.z_score(absolute_differences)

        # Step 2: Detect artifacts based on condition (i) - z-score > 5
        start_indices_1_amp, end_indices_1_amp = self.events_exceed_th(z_absolute_amplitude, th=5)
        start_indices_1_filt, end_indices_1_filt = self.events_exceed_th(z_filtered_absolute_amplitude, th=5)
        start_indices_1_diff, end_indices_1_diff = self.events_exceed_th(z_absolute_differences, th=5)

        # Combine indices for condition (i)
        start_indices_1 = np.concatenate([start_indices_1_amp, start_indices_1_filt, start_indices_1_diff])
        end_indices_1 = np.concatenate([end_indices_1_amp, end_indices_1_filt, end_indices_1_diff])

        # Step 3: Detect artifacts based on condition (ii) - conjunction of z-scores > 3
        start_amp, end_amp = self.events_exceed_th(z_absolute_amplitude, th=3)
        start_grad, end_grad = self.events_exceed_th(z_filtered_absolute_amplitude, th=3)
        start_diff, end_diff = self.events_exceed_th(z_absolute_differences, th=3)

        # Combine gradient and high-frequency conditions
        start_combined = np.concatenate([start_grad, start_diff])
        end_combined = np.concatenate([end_grad, end_diff])

        # Find overlapping events (amplitude AND combined conditions)
        start_indices_2 = []
        end_indices_2 = []
        for start_amp_event, end_amp_event in zip(start_amp, end_amp):
            # Check for overlap with any combined event
            overlap = (start_combined <= end_amp_event) & (end_combined >= start_amp_event)
            if np.any(overlap):
                start_indices_2.append(start_amp_event)
                end_indices_2.append(end_amp_event)

        start_indices_2 = np.array(start_indices_2)
        end_indices_2 = np.array(end_indices_2)

        # Step 4: Combine artifacts from conditions (i) and (ii)
        start_indices = np.concatenate([start_indices_1, start_indices_2])
        end_indices = np.concatenate([end_indices_1, end_indices_2])

        # Sort and remove duplicates
        sorted_indices = np.argsort(start_indices)
        start_indices = start_indices[sorted_indices]
        end_indices = end_indices[sorted_indices]

        start_times = start_indices / fs
        end_times = end_indices / fs

        return start_times, end_times



class SkelinRipplesDetectionMethod(RipplesDetection):
    def __init__(self, hp_channel_section, fs, start_of_sections, verbose=True):
        super().__init__(verbose=verbose)
        self.hp_channel_section = hp_channel_section
        self.start_of_sections = start_of_sections
        self.fs = fs

    def ripple_detection(self):
        filtered_signal = self.filter_signal(self.hp_channel_section, [0.5], 4, 'chebyshev', self.fs, btype='high')
        filtered_signal = self.filter_signal(filtered_signal, [80,150], 4, 'chebyshev', self.fs, btype='band')
        rectified_signal = np.abs(filtered_signal)
        upper_envelope = np.abs(hilbert(rectified_signal))
        clean_upper_envelope = self.get_only_clean_trails(upper_envelope, start_of_clean_trails=self.start_of_sections, fs = self.fs)

        z_score = (upper_envelope - np.mean(clean_upper_envelope)) / np.std(clean_upper_envelope)

        min_duration_samples = int(0.02 * self.fs)  # Minimum event duration (20 ms)
        max_duration_samples = int(0.1 * self.fs)  # Maximum event duration (100 ms)

        lower_threshold = 2
        upper_threshold = 5

        start_indices, end_indices = self.events_exceed_th(z_score, lower_threshold)

        ripples = []
        for start, end in zip(start_indices, end_indices):
            duration = end - start
            # Check duration constraint
            if min_duration_samples <= duration <= max_duration_samples:
                # Find peak in the segment
                segment = z_score[start:end]
                peak_amplitude = np.max(segment)

                # Check if peak exceeds upper threshold
                if peak_amplitude >= upper_threshold:
                    peak = np.argmax(segment) + start
                    confidence_score = np.max(z_score[start:end]) - lower_threshold
                    ripples.append((start / self.fs, peak/ self.fs, end / self.fs, confidence_score))

        ripples_df = pd.DataFrame(ripples, columns=['start', 'peak', 'end', 'confidence'])
        len_before = len(ripples_df)

        filter_partial_function = partial(self.filter_signal, band=[300], N=4, filter_type='chebyshev', fs=self.fs, btype='low')
        start_times, end_times = self.IED_detection(self.hp_channel_section, filter_partial_function, std_threshold=5, fs=self.fs, method='skelin')
        ripples_df = ripples_df[~ripples_df.apply(lambda x: any([(x['start'] - 1 >= end_time) & (x['end'] + 1 <= start_time) for start_time, end_time in zip(start_times, end_times)]), axis=1)]
        self.log(f' *** Rejected {len_before - len(ripples_df)} / {len_before} events based on IED rejection ***')
        self.log(f' *** final number of ripples: {len(ripples_df)} ***')
        if len(ripples_df)>0:
            ripples_df = self.filter_clean_trails(ripples_df, clean_trails_start=self.start_of_sections)
            return ripples_df
        else:
            return pd.DataFrame()


class HeninRipplesDetectionMethod(RipplesDetection):
    def __init__(self, hp_channel_section, fs, start_of_sections, verbose=True):
        super().__init__(verbose=verbose)
        self.hp_channel_section = hp_channel_section
        self.start_of_sections = start_of_sections
        self.fs = fs

    def ripple_detection(self):
        ripple_band = [80, 120]
        filtered_hippo = self.filter_signal(self.hp_channel_section, ripple_band, 101, 'hamming', self.fs)
        hilbert_hippo = np.abs(hilbert(filtered_hippo))
        clean_hilbert_hippo = self.get_only_clean_trails(hilbert_hippo, start_of_clean_trails=self.start_of_sections, fs=self.fs)
        std = np.std(clean_hilbert_hippo)
        mean = np.mean(clean_hilbert_hippo)
        z_score_hippo = (hilbert_hippo - mean) / std

        # threshold = mean + 2 * std
        threshold = 2
        start_indices, end_indices = self.events_exceed_th(z_score_hippo, threshold)
        min_duration_samples = 20e-3 * self.fs
        max_duration_samples = 200e-3 * self.fs

        ripples = []
        for start, end in zip(start_indices, end_indices):
            duration = end - start
            if min_duration_samples <= duration <= max_duration_samples:
                peak = get_peak_index(hilbert_hippo[start:end], self.hp_channel_section[start:end], start)
                confidence = np.max(z_score_hippo[start:end]) - threshold
                ripples.append((start / self.fs, peak / self.fs, end / self.fs, confidence))

        ripples_df = pd.DataFrame(ripples, columns=['start', 'peak', 'end', 'confidence'])
        len_before = len(ripples_df)

        # reject IEDs
        spike_times = self.get_IED_events_henin()
        ripples_df = ripples_df[~ripples_df.apply(lambda x: any([x['start'] < spike_event < x['end'] for spike_event in spike_times]), axis=1)]
        self.log(f' *** rejected {len_before - len(ripples_df)} / {len_before} events based on IED rejection ***')
        self.log(f' *** Number of ripples after IED rejection: {len(ripples_df)} ***')

        if len(ripples_df)>0:
            ripples_df = self.filter_clean_trails(ripples_df, clean_trails_start=self.start_of_sections)
            return ripples_df
        else:
            return pd.DataFrame()

    def log_normal_model(self, data):
        shape, loc, scale = lognorm.fit(data, floc=0)
        mode = scale * np.exp(-shape ** 2)
        median = scale * np.exp(0)
        return mode, median

    def get_IED_events_henin(self):
        IED_band = [10, 60]
        filtered_hippo = self.filter_signal(self.hp_channel_section, IED_band, 101, 'hamming', self.fs)
        hilbert_hippo = np.abs(hilbert(filtered_hippo))

        window_size = int(5 * self.fs)  # 5-second window
        overlap = int(4 * self.fs)      # 4-second overlap
        step_size = window_size - overlap

        spike_indices_list = []

        for start in range(0, len(hilbert_hippo) - window_size, step_size):
            window = hilbert_hippo[start:start + window_size]

            # Compute log-normal distribution parameters
            mode_log_env, median_log_env = self.log_normal_model(window)

            threshold = 3.3 * mode_log_env + median_log_env

            spikes = (window > threshold)
            spike_indices = np.where(spikes)

            # Convert spike indices to time
            spike_indices = spike_indices[0] + start
            spike_indices_list.append(spike_indices)

        spike_times = np.concatenate(spike_indices_list) / self.fs
        return spike_times

class SakonKahanaRipplesDetectionMethod(RipplesDetection):
    def __init__(self, hp_channel_section, mean_LFP, fs, start_of_sections, th = [2, 3, 4], ripple_band = [70, 178], IED_band = [25, 58],  verbose=True):
        super().__init__(verbose=verbose)
        self.hp_channel_section = hp_channel_section
        self.mean_LFP = mean_LFP
        self.start_of_sections = start_of_sections
        self.fs = fs
        self.ripple_band = ripple_band
        self.IED_band = IED_band
        self.th = th
        self.minDistance = 0.030  # in sec
        self.minRippleDuration = 0.020  # in sec
        self.maxRippleDuration = 0.200  # in sec


    def ripple_detection(self):
        norman_ripples_method = NormanRipplesDetectionMethod(self.hp_channel_section, self.mean_LFP, start_of_sections=self.start_of_sections, th=self.th, ripple_band=self.ripple_band, IED_band=self.IED_band, fs=self.fs, verbose=self.verbose)
        ripples_df = norman_ripples_method.ripple_detection()
        return ripples_df

class CharupanitRipplesDetectionMethod(RipplesDetection):
    def __init__(self, hp_channel_section, fs, start_of_sections, verbose=True):
        super().__init__(verbose=verbose)
        self.hp_channel_section = hp_channel_section
        self.start_of_sections = start_of_sections
        self.fs = fs

    def ripple_detection(self):
        # Filter design - CORRECTED: n1 = 700 (not 701)
        n1 = 701
        f1 = np.array([0, 70, 80, 250, 260, self.fs / 2]) / (self.fs / 2)
        a1 = [0, 0, 1, 1, 0, 0]
        b1 = firls(n1, f1, a1)
        filtered = filtfilt(b1, [1], self.hp_channel_section)
        filtered_abs = np.abs(filtered)

        # Find peaks - CORRECTED: Using MATLAB-style peak detection
        peakInd = self._find_peaks_matlab_style(filtered_abs)

        peakTimes = np.where(peakInd)[0] / self.fs
        clean_peakTimes = self.filter_clean_trails(peakTimes, clean_trails_start=self.start_of_sections)
        peakVal = filtered_abs[(clean_peakTimes * self.fs).astype(int)]


        # Iterative threshold - CORRECTED: returns final threshold and keeps original peakVal
        detections, final_threshold, scale = self.iterative_threshold(peakVal.copy(), alpha=0.001)
        conf = (peakVal - final_threshold) / scale

        # Parameters
        nCycles = 6
        npeakth = 5
        t_step = 0.010

        # Event detection
        temp_sum = detections.sum(axis=0)
        inarow = np.convolve(temp_sum, np.ones(nCycles), mode='same')
        inarow[inarow < npeakth] = 0
        in_a_row2 = np.convolve(inarow, np.ones(nCycles), mode='same')
        in_a_row2[in_a_row2 != 0] = 1
        flag_temp, num_events = label(in_a_row2 > 0)

        # Collect candidate events
        candidates = []
        for zz in range(1, num_events + 1):
            temp_find = np.where(flag_temp == zz)[0]
            if len(temp_find) > 0:
                # CORRECTED: Use original peakVal for confidence, not zeroed version
                candidates.append([
                    clean_peakTimes[temp_find[0]],
                    clean_peakTimes[temp_find[-1]],
                    np.median(conf[temp_find])  # CORRECTED: np.median() not .median()
                ])

        if not candidates:
            return pd.DataFrame()

        candidates = np.array(candidates)

        # Merge events that are too close
        if len(candidates) > 1:
            gap_times = candidates[1:, 0] - candidates[:-1, 1]
            merge_idx = np.where(gap_times < t_step)[0]

            for idx in reversed(merge_idx):  # iterate in reverse
                candidates[idx, 1] = candidates[idx + 1, 1]
                candidates = np.delete(candidates, idx + 1, axis=0)

        # Add peak times
        peak_times = []
        for start, end, _ in candidates:
            start_idx = int(start * self.fs)
            end_idx = int(end * self.fs)
            peak_idx = get_peak_index(
                np.abs(hilbert(filtered))[start_idx:end_idx],
                self.hp_channel_section[start_idx:end_idx],
                start_idx
            )
            peak_times.append(peak_idx / self.fs)

        candidates = np.column_stack((candidates, peak_times))

        ripples_df = pd.DataFrame(candidates, columns=['start', 'end', 'confidence', 'peak'])
        self.log(f' *** Number of ripples: {len(ripples_df)} ***')
        return ripples_df

    def _find_peaks_matlab_style(self, signal):
        """
        MATLAB-style peak detection: peak must be strictly higher than both neighbors
        Equivalent to: peakInd = ((R2 > R3) & (R2 > R1))
        """
        R1 = signal[:-2]
        R2 = signal[1:-1]
        R3 = signal[2:]
        peakInd = (R2 > R3) & (R2 > R1)
        # Add False at start and end to match original length
        return np.concatenate([[False], peakInd, [False]])

    def iterative_threshold(self, peakVal, alpha=0.01, n_iterations=16):
        """
        CORRECTED: Implements MATLAB iterative threshold calculation
        Returns detections matrix and final threshold value
        """
        detections = np.zeros((n_iterations, len(peakVal)))
        kai = np.zeros(n_iterations)

        # Create discrete distribution for threshold calculation
        nMax = np.max(peakVal) * 3
        nn = np.linspace(0, nMax, 3000)

        for jj in range(n_iterations):
            # Fit gamma distribution to remaining peaks
            valid_peaks = peakVal[peakVal > 0]
            if len(valid_peaks) == 0:
                break

            # MATLAB gamfit returns [shape, scale] with 2 params
            shape, loc, scale = gamma.fit(valid_peaks, floc=0)

            # Calculate CDF at discrete points
            P = gamma.cdf(nn, shape, loc=loc, scale=scale)
            cumulative = 1 - P

            # Find threshold where cumulative < alpha
            alpha_ind = np.where(cumulative < alpha)[0]
            if len(alpha_ind) > 0:
                kai[jj] = nn[np.min(alpha_ind)]
            else:
                kai[jj] = nMax

            # Identify detections
            detections[jj, :] = peakVal > kai[jj]

            # Remove detected peaks for next iteration
            peakVal[peakVal > kai[jj]] = 0

        return detections, kai[-1], scale  # Return final threshold

class FrauscherRipplesDetectionMethod(RipplesDetection):
    def __init__(self, hp_channel_section, fs, start_of_sections, verbose=True):
        super().__init__(verbose=verbose)
        self.hp_channel_section = hp_channel_section
        self.start_of_sections = start_of_sections
        self.fs = fs

        self.background_window_length_sec = 5
        self.threshold = 3  # 2
        self.min_separation_ripple = 0.02  # seconds

        self.bandpass_filter_order = 200
        self.narrowband_filter_order = 508
        self.weights= [10,1,1]

    def ripple_detection(self):
        # Section 1: parameters
        num_samples = len(self.hp_channel_section)
        window_length = int(self.background_window_length_sec * self.fs)
        if num_samples < window_length + 1:
            window_length = num_samples - 3

        # Section 2: design filters
        # Bandpass filter
        band_edges = np.array([0, 65, 80, min(0.45 * self.fs, 500),
                      min(0.45 * self.fs + 15, 515), 0.5 * self.fs]) / self.fs
        desired_response = [0, 1, 0]
        BandpassFilter = remez(self.bandpass_filter_order + 1, band_edges, desired_response, weight=self.weights)

        # w, h = freqz(BandpassFilter, [1], worN=2000, fs=self.fs)
        # self.plot_response(w, h, "Frauscher Band-pass Filter")

        # Frequency bands (only for ripples)
        FrequencyBands = np.array([80, 90, 105, 120, 140, 160, 185, 215, 250
                                   ]) / self.fs # 285, 330, 380, 435, 500
        FrequencyBands = FrequencyBands[FrequencyBands <= 0.45]
        NumberOfBands = len(FrequencyBands) - 1
        narrowband_filters = np.zeros((NumberOfBands, self.narrowband_filter_order + 1))

        desired_response = [0, 1, 0]
        for i in range(NumberOfBands):
            band_edges = [0,
                          FrequencyBands[i] - 7 / self.fs,
                          FrequencyBands[i] + 3 / self.fs,
                          FrequencyBands[i + 1] - 3 / self.fs,
                          FrequencyBands[i + 1] + 7 / self.fs,
                          0.5]
            narrowband_filters[i, :] = remez(self.narrowband_filter_order + 1, band_edges, desired_response, weight=self.weights)

        FrequencyBands = (FrequencyBands[:-1] + FrequencyBands[1:]) / 2

        # Compute the effective duration of the filter response
        MinimumLength = self.calculate_filter_minimum_length(self.narrowband_filter_order, FrequencyBands, narrowband_filters, NumberOfBands) # TODO

        Offset = int((self.narrowband_filter_order + self.bandpass_filter_order) / 2)
        # Section 3
        # Bandpass filter
        BanspassSignal = filtfilt(BandpassFilter, [1], self.hp_channel_section)
        events = []

        for ii in range(NumberOfBands):
            NarrowbandSignal = filtfilt(narrowband_filters[ii, :], [1], BanspassSignal)
            # Compute RMS (squared) value in 4 cycles window
            FourCyclesHalfDuration = int(2 / FrequencyBands[ii])
            FourCyclesWindow = np.ones(int(2 * FourCyclesHalfDuration + 1)) / (2 * FourCyclesHalfDuration + 1)
            filtered_squared_signal = np.maximum(filtfilt(FourCyclesWindow, [1], NarrowbandSignal ** 2), 0)
            RMSSignal = np.sqrt(filtered_squared_signal)

            RMSSignal_clean = self.get_only_clean_trails(RMSSignal, start_of_clean_trails=self.start_of_sections,fs=self.fs)
            clean_window_length = min(window_length, len(RMSSignal_clean) - 3)
            MovingBackground_clean = np.convolve(
                RMSSignal_clean,
                np.ones(clean_window_length) / clean_window_length,
                mode='same'
            )

            # 2. Use statistics from clean trials to set a FIXED threshold
            mean_background = np.mean(MovingBackground_clean)
            detection_threshold = self.threshold * mean_background

            # 3. Apply FIXED threshold to entire session
            CandidateDetections = RMSSignal > detection_threshold

            # Find beginning and end of detections
            CandidateDetections[:Offset] = False
            CandidateDetections[-Offset:] = False

            # Keep only detections of enough length
            CandidateDetections = binary_opening(CandidateDetections, structure=np.ones(MinimumLength[ii]))
            transitions = np.logical_xor(CandidateDetections[:-1], CandidateDetections[1:])
            event_indices = np.where(transitions)[0]

            if len(event_indices) >= 2:  # Must have at least start and end
                for idx in range(0, len(event_indices), 2):
                    if idx + 1 >= len(event_indices):
                        break

                    start = event_indices[idx] + 1
                    end = event_indices[idx + 1]

                    # Get peak index
                    peak = get_peak_index(np.abs(hilbert(NarrowbandSignal))[start:end],
                                          self.hp_channel_section[start:end], start)


                    peak_value = max(RMSSignal[start:end])

                    # Calculate confidence: how many times above threshold
                    confidence = peak_value / detection_threshold

                    events.append([
                        self.fs * FrequencyBands[ii],  # Frequency
                        start / self.fs,  # Start time
                        peak / self.fs,  # Peak time
                        end / self.fs,  # End time
                        confidence  # Confidence (times above threshold)
                    ])

        events = np.array(events)
        if len(events) == 0:
            self.log(f' *** Number of ripples: 0 ***')
            return pd.DataFrame()
        ripple_events = events[events[:, 0] < 250]
        if len(ripple_events) == 0:
            self.log(f' *** Number of ripples: 0 ***')
            return pd.DataFrame()
        ripple_events = pd.DataFrame(ripple_events,
                                columns=['frequency', 'start', 'peak', 'end', 'confidence'])
        ripples_df = self.merge_events(ripple_events, self.min_separation_ripple)
        if len(ripples_df)>0:
            ripples_df = self.filter_clean_trails(ripples_df, clean_trails_start=self.start_of_sections)
            return ripples_df
        else:
            return pd.DataFrame()


    def calculate_filter_minimum_length(self, NarrowbandFilterOrder: int, FrequencyBands, NarrowBandFilters, NumberOfBands):
        order_range = np.arange(-NarrowbandFilterOrder / 2, NarrowbandFilterOrder / 2 + 1)
        order_matrix = np.tile(order_range, (NumberOfBands, 1))
        weighted_squares = np.sum((order_matrix ** 2) * (NarrowBandFilters ** 2), axis=1)
        filter_sums = np.sum(NarrowBandFilters ** 2, axis=1)
        ratio = weighted_squares / filter_sums
        MinimumLength = np.round(np.sqrt(ratio) + 4 / FrequencyBands).astype(int)
        return MinimumLength

    def merge_events(self, events_df, min_separation):
        if len(events_df) == 0:
            return np.array([])

        # Sort events by start time
        events_df = events_df.sort_values(by='start')

        merged = []
        current_event = events_df.iloc[0]

        for _, next_event in events_df.iloc[1:].iterrows():
            if (next_event['start'] - current_event['end']) <= min_separation:
                # Merge events
                end_time = max(current_event['end'], next_event['end'])
                current_event = pd.Series({'frequency': np.mean(np.array([current_event['frequency'], next_event['frequency']])),
                                           'start': current_event['start'],
                                           'peak': max(current_event['peak'], next_event['peak']),
                                           'end': end_time,
                                           'confidence': np.mean(np.array([current_event['confidence'], next_event['confidence']]))})
            else:
                merged.append(current_event)
                current_event = next_event

        merged.append(current_event)
        merged_df = pd.DataFrame(merged)
        return merged_df


def get_peak_index(filtered_data, data, start_idx):
    """

    :param filtered_data: hilbert transform of the filtered ripple band + hilbert transform
    :param data: raw signal
    :param start_idx:
    :return:
    """
    all_pks, _ = find_peaks(filtered_data)
    if len(all_pks) == 0:
        return start_idx + np.argmax(filtered_data)
    peak = all_pks[np.argmax(filtered_data[all_pks])]
    min_indices, _ = find_peaks(-data)
    if len(min_indices) == 0:
        return peak + start_idx
    peak_idx = min_indices[np.argmin(np.abs(min_indices - peak))]
    return peak_idx + start_idx




