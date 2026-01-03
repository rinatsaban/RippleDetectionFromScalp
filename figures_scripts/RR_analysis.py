import joblib
import matplotlib.pyplot as plt

from RR_functions import *

## load events
events_files_path = '/Users/rinatsaban/Library/CloudStorage/OneDrive-mail.tau.ac.il/TAU/master/docs for writing/events'
start_of_trails_path = '/Users/rinatsaban/Library/CloudStorage/OneDrive-mail.tau.ac.il/TAU/master/docs for writing/start_of_trails'
figures_path = '/Users/rinatsaban/Library/CloudStorage/OneDrive-mail.tau.ac.il/TAU/master/docs for writing/figures'
ripples_path = f'/Users/rinatsaban/Library/CloudStorage/OneDrive-mail.tau.ac.il/TAU/master/docs for writing/ripples'

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

events_files = [f for f in os.listdir(events_files_path) if f.endswith('.tsv')]
all_events = pd.DataFrame()
for file in events_files:
    event_df = pd.read_csv(os.path.join(events_files_path, file), sep='\t')
    patient = file.split('_')[0]
    event_df['patient'] = patient
    all_events = pd.concat([all_events, event_df], ignore_index=True)
all_events = all_events[all_events['Artifact'] == 0]

final = compute_scores(all_events)
plot_patient_tests(final.set_index('patient'),
                   save_fname='patient_tests_by_patient.png',
                   figsize=(12, 6),
                   columns=['accuracy_all','acc_set_4', 'acc_set_6','acc_set_8']
                  )


#load norman bp ripples
n = len(methods_to_run)
fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(4*4, 3*2),
    sharex=True,
    sharey=True
)

axes = axes.flatten()  # important

all_patients_level_rates = []
for ax, method in zip(axes, methods_to_run):
    ripples_method_path =os.path.join(ripples_path, f'all_trails_w_peak_all_patients_for_writing_{method}.pkl')
    ripples_df = joblib.load(ripples_method_path)
    # calculate Ripple rate per patient per side
    plot_hist_per_window_size(ripples_df, method, start_of_trails_path, figures_path)
    ripple_rates = compute_ripple_rate_per_electrode_session(ripples_df, start_of_trails_path, method)
    # ripple_rates['electrode_side'] = ripple_rates['electrode'].apply(
    #     lambda elec: 'R' if elec[-2] == 'R' else ('L' if elec[-2] == 'L' else 'L')
    # )
    patient_level_rates = compute_patient_level_rate_with_sides(ripple_rates)
    # patient_level_rates['method'] = method
    # all_patients_level_rates.append(patient_level_rates)


# # Change the figure creation
# fig, axes = plt.subplots(
#     nrows=2,
#     ncols=4,
#     figsize=(4*4, 3*2),
#     sharex=True,
#     sharey=True  # Changed from True to False
# )
#
# axes = axes.flatten()
#
# for ax, method in zip(axes, methods_to_run):
#     ripples_method_path =os.path.join(ripples_path, f'all_trails_w_peak_all_patients_for_writing_{method}.pkl')
#     ripples_df = joblib.load(ripples_method_path)
#     x = ripples_df['confidence']
#     x_normalized = (x - x.min()) / (x.max() - x.min())
#     sns.kdeplot(
#         x_normalized,
#         ax=ax,
#         cut=0,
#         bw_adjust=0.7
#     )
#
#     ax.set_title(method)
#     ax.set_xlabel("Confidence")
#     ax.set_ylabel("Density")  # Add ylabel to each subplot
#
# plt.tight_layout()
# plt.savefig(os.path.join(figures_path, 'confidence_kde_all_methods.png'))
# plt.close()
# #
# all_methods_all_patients_level_rates = pd.concat(all_patients_level_rates, ignore_index=True)
# plot_ripple_rate_by_method_boxplot(all_methods_all_patients_level_rates, methods_to_run, figures_path)
#
# plot_ripple_rate_vs_task_score(all_methods_all_patients_level_rates, final, figures_path)
# plot_ripple_rate_vs_task_score(all_methods_all_patients_level_rates, final, figures_path, size_comparison=True)
# plot_ripple_rate_L_R_boxplot(all_methods_all_patients_level_rates, methods_to_run, figures_path)



