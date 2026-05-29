import pandas as pd
import matplotlib.pyplot as plt
import os

configs = [
    {'path': '/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/eval_2d/ssim_results/summary_ssim.csv', 'label': '2d'},
    {'path': '/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/eval_256/ssim_results/summary_ssim.csv', 'label': '256'},
    {'path': '/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/eval_spline/ssim_results/summary_ssim.csv', 'label': 'spline'},
]

plot_dir = r'/home/cxv166/PhantomTesting/Code/plots/ssim_per_volume/'
os.makedirs(plot_dir, exist_ok=True)

for config in configs:
    df = pd.read_csv(config['path'])
    df = df[df['volume'] != 'OVERALL'].copy()
    df['volume'] = df['volume'].str.split('_filter_').str[0]

    smooth = df[df['type'] == 'smooth'].set_index('volume')['avg_ssim']
    sharp = df[df['type'] == 'sharp'].set_index('volume')['avg_ssim']
    volumes = df['volume'].unique()

    x = range(len(volumes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar([i - width/2 for i in x], smooth.reindex(volumes), width=width, label='smooth', color='steelblue')
    ax.bar([i + width/2 for i in x], sharp.reindex(volumes), width=width, label='sharp', color='salmon')

    ax.set_xticks(x)
    ax.set_xticklabels(volumes, rotation=90)
    ax.set_ylabel('avg_ssim')
    ax.set_title(f'SSIM per volume — {config["label"]}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'ssim_per_volume_{config["label"]}.png'), bbox_inches='tight', dpi=150)
    plt.close()

print('done')
