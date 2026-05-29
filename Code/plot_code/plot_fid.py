import pandas as pd
import matplotlib.pyplot as plt
import os

configs = [
    {'path': '/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/eval_2d/fid_results/summary_fid.csv', 'label': '2d'},
    {'path': '/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/eval_256/fid_results/summary_fid.csv', 'label': '256'},
    {'path': '/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/eval_spline/fid_results/summary_fid.csv', 'label': 'spline'},
]

plot_dir = '/home/cxv166/PhantomTesting/Code/plots/fid/'
os.makedirs(plot_dir, exist_ok=True)

# --- Plot 1: per volume bar plot for each model ---
for config in configs:
    df = pd.read_csv(config['path'])
    df = df[df['volume'] != 'OVERALL'].copy()
    df['volume'] = df['volume'].str.split('_filter_').str[0]

    smooth = df[df['type'] == 'smooth'].set_index('volume')['fid']
    sharp = df[df['type'] == 'sharp'].set_index('volume')['fid']
    volumes = df['volume'].unique()

    x = range(len(volumes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar([i - width/2 for i in x], smooth.reindex(volumes), width=width, label='smooth', color='steelblue')
    ax.bar([i + width/2 for i in x], sharp.reindex(volumes), width=width, label='sharp', color='salmon')

    ax.set_xticks(x)
    ax.set_xticklabels(volumes, rotation=90)
    ax.set_ylabel('FID')
    ax.set_title(f'FID per volume — {config["label"]}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'fid_per_volume_{config["label"]}.png'), bbox_inches='tight', dpi=150)
    plt.close()

records = []
for config in configs:
    df = pd.read_csv(config['path'])
    row = df[df['volume'] == 'OVERALL'].copy()
    row['dataset'] = config['label']
    records.append(row)

overall = pd.concat(records)
smooth = overall[overall['type'] == 'smooth'].set_index('dataset')['fid']
sharp = overall[overall['type'] == 'sharp'].set_index('dataset')['fid']

datasets = [c['label'] for c in configs]
x = range(len(datasets))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

ax1.bar(x, smooth[datasets], width=width, color='steelblue')
ax1.set_title('Smooth')
ax1.set_xticks(x)
ax1.set_xticklabels(datasets)
ax1.set_ylabel('FID')

ax2.bar(x, sharp[datasets], width=width, color='salmon')
ax2.set_title('Sharp')
ax2.set_xticks(x)
ax2.set_xticklabels(datasets)

plt.suptitle('Overall FID per model')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'overall_fid.png'), bbox_inches='tight', dpi=150)
plt.close()

print('done')
