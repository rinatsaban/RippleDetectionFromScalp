import pandas as pd

from ripple_detection.ripples_detection_methods_hybrid import *
from time import time
import concurrent.futures
import mne
mne.set_log_level('CRITICAL')


def run_norman_method(patient, session, hippo_data_wm, hippo_data_bp, hippo_data_mastoids_bp, start_of_trails_WM, start_of_trails_BP, start_of_trails_BP_mas, hippo_elec_names_wm, hippo_elec_names_bp, hippo_elec_names_bp_mas):
    ######## Uncomment if want to run norman_bp! ########
    # hippo_data_wm = hippo_data_bp
    # start_of_trails_WM = start_of_trails_BP
    # hippo_elec_names_wm = hippo_elec_names_bp

    trails_results = pd.DataFrame(columns=['patient', 'session', 'electrode', 'start', 'end', 'peak', 'confidence'])
    # start = time()
    ieeg_fs_500 = 500
    hippo_data_wm = hippo_data_wm.copy().resample(ieeg_fs_500)
    hippo_data = hippo_data_wm.get_data(picks=hippo_elec_names_wm)
    mean_LFP = hippo_data_wm.get_data().copy().mean(axis=0)
    print('Extracting ripples with Norman method')
    # all_sessions_ripples_df = []
    for i, hippo_elec in enumerate(hippo_elec_names_wm):
        norman_ripples_method = NormanRipplesDetectionMethod(hippo_data[i, :], mean_LFP, fs=ieeg_fs_500,
                                                             start_of_sections=start_of_trails_WM,
                                                             verbose=False)
        ripples_df = norman_ripples_method.ripple_detection()
        if len(ripples_df) > 0:
            ripples_df['electrode'] = hippo_elec
            ripples_df['patient'] = patient
            ripples_df['session'] = session
            if len(trails_results)>0:
                trails_results = pd.concat([trails_results, ripples_df], axis=0, ignore_index=True)
            else:
                trails_results = ripples_df
                # all_sessions_ripples_df.append(ripples_df)
    # all_sessions_ripples_df = [ripples_df for ripples_df in all_sessions_ripples_df if len(ripples_df) > 0]
    # all_ripples_df = pd.concat(all_sessions_ripples_df, axis=0, ignore_index=True)
    # end = time()
    # print(f'norman method took {end-start}')
    # print(f'norman total ripples {len(trails_results)}')
    return trails_results


def run_vaz_method(patient, session, hippo_data_wm, hippo_data_bp, hippo_data_mastoids_bp, start_of_trails_WM, start_of_trails_BP, start_of_trails_BP_mas, hippo_elec_names_wm, hippo_elec_names_bp, hippo_elec_names_bp_mas):
    trails_results = pd.DataFrame(columns=['patient', 'session', 'electrode', 'start', 'end', 'peak', 'confidence'])
    # start = time()
    ieeg_fs_1000 = 1000
    bp_referenced_epochs_ieeg = hippo_data_bp.copy().resample(ieeg_fs_1000)
    hippo_data = bp_referenced_epochs_ieeg.get_data(picks=hippo_elec_names_bp)
    print('Extracting ripples with VAZ method')
    for i, hippo_elec in enumerate(hippo_elec_names_bp):
        vaz_ripples_method = VazRipplesDetectionMethod(hippo_data[i, :], ieeg_fs_1000,
                                                       start_of_sections=start_of_trails_BP, verbose=False)
        ripples_df = vaz_ripples_method.ripple_detection()
        if len(ripples_df) > 0:
            ripples_df['electrode'] = hippo_elec
            ripples_df['patient'] = patient
            ripples_df['session'] = session
            if len(trails_results)>0:
                trails_results = pd.concat([trails_results, ripples_df], axis=0, ignore_index=True)
            else:
                trails_results = ripples_df
    # end = time()
    # print(f'vaz method took {end-start}')
    # print(f'vaz total ripples {len(trails_results)}')

    return trails_results


def run_staresina_method(patient, session, hippo_data_wm, hippo_data_bp, hippo_data_mastoids_bp, start_of_trails_WM, start_of_trails_BP, start_of_trails_BP_mas, hippo_elec_names_wm, hippo_elec_names_bp, hippo_elec_names_bp_mas):
    trails_results = pd.DataFrame(columns=['patient', 'session', 'electrode', 'start', 'end', 'peak', 'confidence'])
    ieeg_fs_1000 = 1000
    # start = time()

    hippo_data_mastoids_bp_refrenced = hippo_data_mastoids_bp.copy().resample(ieeg_fs_1000)
    hippo_data = hippo_data_mastoids_bp_refrenced.get_data(picks=hippo_elec_names_bp_mas)
    print('Extracting ripples with STARESINA method')
    for i, hippo_elec in enumerate(hippo_elec_names_bp_mas):
        staresina_ripples_method = StaresinaRipplesDetectionMethod(hippo_data[i, :], ieeg_fs_1000,
                                                                   start_of_sections=start_of_trails_BP_mas, verbose=False)
        ripples_df = staresina_ripples_method.ripple_detection()
        if len(ripples_df) > 0:
            ripples_df['electrode'] = hippo_elec
            ripples_df['patient'] = patient
            ripples_df['session'] = session
            if len(trails_results) > 0:
                trails_results = pd.concat([trails_results, ripples_df], axis=0, ignore_index=True)
            else:
                trails_results = ripples_df
    # end = time()
    # print(f'staresina method took {end-start}')
    # print(f'staresina total ripples {len(trails_results)}')
    return trails_results


def run_skelin_method(patient, session, hippo_data_wm, hippo_data_bp, hippo_data_mastoids_bp, start_of_trails_WM, start_of_trails_BP, start_of_trails_BP_mas, hippo_elec_names_wm, hippo_elec_names_bp, hippo_elec_names_bp_mas):
    trails_results = pd.DataFrame(columns=['patient', 'session', 'electrode', 'start', 'end', 'peak', 'confidence'])

    # start = time()
    ieeg_fs_2000 = 2000
    bp_referenced_ieeg = hippo_data_bp.copy().resample(ieeg_fs_2000)
    hippo_data = bp_referenced_ieeg.get_data(picks=hippo_elec_names_bp)

    print('Extracting ripples with SKELIN method')
    for i, hippo_elec in enumerate(hippo_elec_names_bp):
        skelin_ripples_method = SkelinRipplesDetectionMethod(hippo_data[i, :], ieeg_fs_2000,
                                                             start_of_sections=start_of_trails_BP,
                                                             verbose=False)
        ripples_df = skelin_ripples_method.ripple_detection()
        if len(ripples_df) > 0:
            ripples_df['electrode'] = hippo_elec
            ripples_df['patient'] = patient
            ripples_df['session'] = session
            if len(trails_results) > 0:
                trails_results = pd.concat([trails_results, ripples_df], axis=0, ignore_index=True)
            else:
                trails_results = ripples_df
    # end = time()
    # print(f'skelin method took {end-start}')
    # print(f'skelin total ripples {len(trails_results)}')

    return trails_results

def run_henin_method(patient, session, hippo_data_wm, hippo_data_bp, hippo_data_mastoids_bp, start_of_trails_WM, start_of_trails_BP, start_of_trails_BP_mas, hippo_elec_names_wm, hippo_elec_names_bp, hippo_elec_names_bp_mas):
    trails_results = pd.DataFrame(columns=['patient', 'session', 'electrode', 'start', 'end', 'peak', 'confidence'])

    start = time()
    ieeg_fs_512 = 512
    bp_referenced_ieeg = hippo_data_bp.copy().resample(ieeg_fs_512)
    hippo_data = bp_referenced_ieeg.get_data(picks=hippo_elec_names_bp)
    print('Extracting ripples with HENIN method')
    for i, hippo_elec in enumerate(hippo_elec_names_bp):
        henin_ripples_method = HeninRipplesDetectionMethod(hippo_data[i, :], ieeg_fs_512,
                                                           start_of_sections=start_of_trails_BP,
                                                           verbose=False)
        ripples_df = henin_ripples_method.ripple_detection()
        if len(ripples_df) > 0:
            ripples_df['electrode'] = hippo_elec
            ripples_df['patient'] = patient
            ripples_df['session'] = session
            if len(trails_results) > 0:
                trails_results = pd.concat([trails_results, ripples_df], axis=0, ignore_index=True)
            else:
                trails_results = ripples_df
    # end = time()
    # print(f'henin method took {end - start}')
    # print(f'henin total ripples {len(trails_results)}')

    return trails_results


def run_sakon_kahana_method(patient, session, hippo_data_wm, hippo_data_bp, hippo_data_mastoids_bp, start_of_trails_WM, start_of_trails_BP, start_of_trails_BP_mas, hippo_elec_names_wm, hippo_elec_names_bp, hippo_elec_names_bp_mas):
    trails_results = pd.DataFrame(columns=['patient', 'session', 'electrode', 'start', 'end', 'peak', 'confidence'])

    start = time()
    ieeg_fs_1000 = 1000
    bp_referenced_ieeg = hippo_data_bp.copy().resample(ieeg_fs_1000)
    hippo_data = bp_referenced_ieeg.get_data(picks=hippo_elec_names_bp)
    mean_LFP = bp_referenced_ieeg.copy().get_data().mean(axis=0)
    print('Extracting ripples with SAKON&KAHANA method')
    for i, hippo_elec in enumerate(hippo_elec_names_bp):

        sakon_kahana_ripples_method = SakonKahanaRipplesDetectionMethod(hippo_data[i, :], mean_LFP,
                                                                        ieeg_fs_1000, start_of_sections=
                                                                        start_of_trails_BP,
                                                                        verbose=False)
        ripples_df = sakon_kahana_ripples_method.ripple_detection()
        if len(ripples_df) > 0:
            ripples_df['electrode'] = hippo_elec
            ripples_df['patient'] = patient
            ripples_df['session'] = session
            if len(trails_results) > 0:
                trails_results = pd.concat([trails_results, ripples_df], axis=0, ignore_index=True)
            else:
                trails_results = ripples_df
    # end = time()
    # print(f'sakon_kahana method took {end - start}')
    # print(f'sakon_kahana total ripples {len(trails_results)}')

    return trails_results


def run_charupanit_method(patient, session, hippo_data_wm, hippo_data_bp, hippo_data_mastoids_bp, start_of_trails_WM, start_of_trails_BP, start_of_trails_BP_mas, hippo_elec_names_wm, hippo_elec_names_bp, hippo_elec_names_bp_mas):
    trails_results = pd.DataFrame(columns=['patient', 'session', 'electrode', 'start', 'end', 'peak', 'confidence'])

    start = time()
    ieeg_fs_2000 = 2000
    bp_referenced_ieeg = hippo_data_mastoids_bp.copy().resample(ieeg_fs_2000) #not specify in the paper
    hippo_data = bp_referenced_ieeg.get_data(picks=hippo_elec_names_bp)
    print('Extracting ripples with CHARUPANIT method')
    for i, hippo_elec in enumerate(hippo_elec_names_bp):
        charupanit_ripples_method = CharupanitRipplesDetectionMethod(hippo_data[i, :], ieeg_fs_2000,
                                                                     start_of_sections=start_of_trails_BP,
                                                                     verbose=False)
        ripples_df = charupanit_ripples_method.ripple_detection()
        if len(ripples_df) > 0:
            ripples_df['electrode'] = hippo_elec
            ripples_df['patient'] = patient
            ripples_df['session'] = session
            if len(trails_results) > 0:
                trails_results = pd.concat([trails_results, ripples_df], axis=0, ignore_index=True)
            else:
                trails_results = ripples_df
    # end = time()
    # print(f'charupanit method took {end - start}')
    # print(f'charupanit total ripples {len(trails_results)}')

    return trails_results



def run_frauscher_method(patient, session, hippo_data_wm, hippo_data_bp, hippo_data_mastoids_bp, start_of_trails_WM, start_of_trails_BP, start_of_trails_BP_mas, hippo_elec_names_wm, hippo_elec_names_bp, hippo_elec_names_bp_mas):
    trails_results = pd.DataFrame(columns=['patient', 'session', 'electrode', 'start', 'end', 'peak', 'confidence'])

    start = time()
    ieeg_fs_2000 = 2000
    bp_referenced_ieeg = hippo_data_bp.copy().resample(ieeg_fs_2000)
    hippo_data = bp_referenced_ieeg.get_data(picks=hippo_elec_names_bp)
    print('Extracting ripples with FRAUSCHER method')
    for i, hippo_elec in enumerate(hippo_elec_names_bp):

        frauscher_ripples_method = FrauscherRipplesDetectionMethod(hippo_data[i, :], ieeg_fs_2000,
                                                                   start_of_sections=start_of_trails_BP, verbose=False)
        ripples_df = frauscher_ripples_method.ripple_detection()
        if len(ripples_df) > 0:
            ripples_df['electrode'] = hippo_elec
            ripples_df['patient'] = patient
            ripples_df['session'] = session
            if len(trails_results) > 0:
                trails_results = pd.concat([trails_results, ripples_df], axis=0, ignore_index=True)
            else:
                trails_results = ripples_df
    # end = time()
    # print(f'frauscher method took {end - start}')
    # print(f'frauscher total ripples {len(trails_results)}')
    return trails_results


def run_all_methods(patient, session, hippo_data_wm, hippo_data_bp, hippo_data_mastoids_bp, start_of_trails_WM, start_of_trails_BP, start_of_trails_BP_mas, hippo_elec_names, trails_results, methods_to_run):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        if 'norman' in methods_to_run:
            futures.append(
                executor.submit(run_norman_method, patient, session, hippo_data_wm, start_of_trails_WM, hippo_elec_names,
                                trails_results))
        if 'vaz' in methods_to_run:
            futures.append(
                executor.submit(run_vaz_method, patient, session, hippo_data_bp, start_of_trails_BP, hippo_elec_names,
                                trails_results))
        if 'staresina' in methods_to_run:
            futures.append(
                executor.submit(run_staresina_method, patient, session, hippo_data_mastoids_bp, start_of_trails_BP_mas,
                                hippo_elec_names, trails_results))
        if 'skelin' in methods_to_run:
            futures.append(
                executor.submit(run_skelin_method, patient, session, hippo_data_bp, start_of_trails_BP, hippo_elec_names,
                                trails_results))
        if 'henin' in methods_to_run:
            futures.append(
                executor.submit(run_henin_method, patient, session, hippo_data_bp, start_of_trails_BP, hippo_elec_names,
                                trails_results))
        if 'sakon_kahana' in methods_to_run:
            futures.append(executor.submit(run_sakon_kahana_method, patient, session, hippo_data_bp, start_of_trails_BP,
                                           hippo_elec_names, trails_results))
        if 'charupanit' in methods_to_run:
            futures.append(executor.submit(run_charupanit_method, patient, session, hippo_data_bp, start_of_trails_BP,
                                           hippo_elec_names, trails_results))
        if 'frauscher' in methods_to_run:
            futures.append(executor.submit(run_frauscher_method, patient, session, hippo_data_bp, start_of_trails_BP,
                                           hippo_elec_names, trails_results))

        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None and not result.empty:
                results.append(result)

    if results:
        trails_results = pd.concat(results, axis=0, ignore_index=True)
    return trails_results