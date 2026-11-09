"""
generate_figures.py
Tao cac hinh anh bieu do tu ket qua thuc nghiem de dua vao bao cao.
Luu tai: ABSA_Project/figures/
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

os.chdir('d:/Ct255-NVTM(BI)/nhom2')

# -- Config --
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})
COLORS = ['#2563EB', '#EA580C', '#16A34A', '#7C3AED', '#DC2626', '#6B7280']
os.makedirs('ABSA_Project/figures', exist_ok=True)

# -- Load data --
with open('ABSA_Project/results/evaluation/detailed_metrics_test.csv', encoding='utf-8') as f:
    metrics_df = pd.read_csv(io.StringIO(f.read()), sep=';')

err_df = pd.read_csv('ABSA_Project/results/evaluation/error_analysis_test.csv', encoding='utf-8')
bench_df = pd.read_csv('ABSA_Project/results/evaluation/model_complexity_benchmark.csv')
conf_df = pd.read_csv('ABSA_Project/results/evaluation/confusion_matrices_test.csv', encoding='utf-8')

aspects = metrics_df['Aspect'].tolist()
acc     = metrics_df['Accuracy'].values * 100
f1mac   = metrics_df['F1_Macro_4Class'].values * 100
acd_f1  = metrics_df['ACD_F1'].values * 100
asc_f1  = metrics_df['ASC_Sentiment_F1'].values * 100

print("[OK] Data loaded")

# =========================================================
# Fig 1: Accuracy vs Macro F1 (10 aspects)
# =========================================================
fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(aspects))
w = 0.38
b1 = ax.bar(x - w/2, acc,   w, label='Accuracy (%)',    color=COLORS[0], edgecolor='white')
b2 = ax.bar(x + w/2, f1mac, w, label='Macro F1-4 (%)', color=COLORS[1], edgecolor='white')

for b in b1:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{b.get_height():.1f}',
            ha='center', va='bottom', fontsize=8)
for b in b2:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{b.get_height():.1f}',
            ha='center', va='bottom', fontsize=8)

ax.axhline(np.mean(acc),   color=COLORS[0], ls=':', lw=1.5, alpha=0.7, label=f'Mean Acc={np.mean(acc):.1f}%')
ax.axhline(np.mean(f1mac), color=COLORS[1], ls=':', lw=1.5, alpha=0.7, label=f'Mean F1={np.mean(f1mac):.1f}%')
ax.set_xticks(x); ax.set_xticklabels(aspects, rotation=15, ha='right')
ax.set_ylim(0, 118); ax.set_ylabel('Score (%)'); ax.legend(loc='upper right', fontsize=9)
ax.set_title('Hinh 4.1: Accuracy va Macro F1 (4 class) theo tung khia canh tren tap Test')
ax.yaxis.grid(True, alpha=0.3, ls='--'); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('ABSA_Project/figures/fig1_accuracy_f1macro.png')
plt.close()
print("[OK] Figure 1 saved")

# =========================================================
# Fig 2: ACD F1 vs ASC Sentiment F1
# =========================================================
fig, ax = plt.subplots(figsize=(12, 5.5))
b1 = ax.bar(x - w/2, acd_f1, w, label='ACD F1 (%)',           color=COLORS[2], edgecolor='white')
b2 = ax.bar(x + w/2, asc_f1, w, label='ASC Sentiment F1 (%)', color=COLORS[3], edgecolor='white')

for b in b1:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{b.get_height():.1f}',
            ha='center', va='bottom', fontsize=8)
for b in b2:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{b.get_height():.1f}',
            ha='center', va='bottom', fontsize=8)

ax.axhline(np.mean(acd_f1), color=COLORS[2], ls=':', lw=1.5, alpha=0.7, label=f'Mean ACD={np.mean(acd_f1):.1f}%')
ax.axhline(np.mean(asc_f1), color=COLORS[3], ls=':', lw=1.5, alpha=0.7, label=f'Mean ASC={np.mean(asc_f1):.1f}%')
ax.set_xticks(x); ax.set_xticklabels(aspects, rotation=15, ha='right')
ax.set_ylim(0, 118); ax.set_ylabel('F1-score (%)'); ax.legend(loc='upper right', fontsize=9)
ax.set_title('Hinh 4.2: ACD F1 va ASC Sentiment F1 theo tung khia canh tren tap Test')
ax.yaxis.grid(True, alpha=0.3, ls='--'); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('ABSA_Project/figures/fig2_acd_asc_f1.png')
plt.close()
print("[OK] Figure 2 saved")

# =========================================================
# Fig 3: Pie - Error distribution
# =========================================================
error_counts = err_df['error_type'].value_counts()
total_errors = error_counts.sum()
label_map = {
    'false_positive':              'False Positive\n(Nhan dien thua)',
    'sentiment_misclassification': 'Sentiment Error\n(Sai cuc tinh)',
    'false_negative':              'False Negative\n(Bo sot khia canh)',
}
raw_labels = error_counts.index.tolist()
pie_labels = [f"{label_map.get(l, l)}\n{error_counts[l]} ({error_counts[l]/total_errors*100:.1f}%)"
              for l in raw_labels]
pie_colors = [COLORS[1], COLORS[0], COLORS[2]][:len(raw_labels)]

fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts = ax.pie(error_counts.values, labels=pie_labels, colors=pie_colors,
                       startangle=140, wedgeprops=dict(edgecolor='white', linewidth=2))
for t in texts:
    t.set_fontsize(10)
ax.set_title(f'Hinh 4.3: Phan bo loai loi ({total_errors} truong hop) tren tap Test (2,224 mau)', pad=12)
plt.tight_layout()
plt.savefig('ABSA_Project/figures/fig3_error_distribution.png')
plt.close()
print("[OK] Figure 3 saved")

# =========================================================
# Fig 4: Confusion matrices for BATTERY, CAMERA, GENERAL
# =========================================================
CLASS_NAMES = ['None', 'Negative', 'Neutral', 'Positive']

def build_cm(aspect_name):
    subset = conf_df[conf_df['Aspect'] == aspect_name]
    if subset.empty:
        return None
    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for _, row in subset.iterrows():
        t_lbl = str(row.get('True_Label', row.get('True_Class', '')))
        p_lbl = str(row.get('Pred_Label', row.get('Pred_Class', '')))
        cnt   = int(row.get('Count', row.get('count', 0)))
        try:
            ti = CLASS_NAMES.index(t_lbl)
            pi = CLASS_NAMES.index(p_lbl)
            cm[ti][pi] += cnt
        except ValueError:
            pass
    return cm

print("Confusion matrix columns:", conf_df.columns.tolist())
print(conf_df.head(6).to_string())

target_aspects = ['BATTERY', 'CAMERA', 'GENERAL']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for asp, ax_ in zip(target_aspects, axes):
    cm = build_cm(asp)
    if cm is None or cm.sum() == 0:
        ax_.text(0.5, 0.5, f'No data for {asp}', ha='center', va='center', transform=ax_.transAxes)
        ax_.set_title(asp); continue
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm / row_sums * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax_, linewidths=0.5, linecolor='gray', cbar=True, vmin=0, vmax=100)
    ax_.set_title(asp, fontweight='bold')
    ax_.set_xlabel('Predicted Label', fontsize=9)
    ax_.set_ylabel('True Label', fontsize=9)
    ax_.tick_params(axis='x', rotation=15)
    ax_.tick_params(axis='y', rotation=0)

fig.suptitle('Hinh 4.4: Ma tran nham lan (%) - BATTERY, CAMERA, GENERAL  |  Gia tri % theo hang True Label', fontsize=11)
plt.tight_layout()
plt.savefig('ABSA_Project/figures/fig4_confusion_matrices.png')
plt.close()
print("[OK] Figure 4 saved")

# =========================================================
# Fig 5: max_len comparison
# =========================================================
ml_labels  = ['max_len=128\n(Duoc chon)', 'max_len=256']
latency_v  = [339.1, 1579.5]
vram_v     = [2622.5, 3937.2]
trunc_v    = [0.88, 0.02]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
bar_clrs = [COLORS[0], COLORS[1]]

for vals, title, ylabel, ax_ in zip(
        [latency_v, vram_v, trunc_v],
        ['Latency (ms/batch)', 'VRAM su dung (MB)', 'Ty le cat ngan (%)'],
        ['ms/batch', 'MB', '%'],
        axes):
    bars = ax_.bar(ml_labels, vals, color=bar_clrs, edgecolor='white', width=0.5)
    for b, v in zip(bars, vals):
        ax_.text(b.get_x()+b.get_width()/2, b.get_height()*1.03, f'{v:.1f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_.set_title(title, fontsize=10)
    ax_.set_ylabel(ylabel)
    ax_.yaxis.grid(True, alpha=0.3, ls='--')
    ax_.set_axisbelow(True)

fig.suptitle('Hinh 4.5: So sanh max_len=128 va max_len=256  |  Latency, VRAM, Ty le mau bi cat ngan', fontsize=11)
plt.tight_layout()
plt.savefig('ABSA_Project/figures/fig5_maxlen_comparison.png')
plt.close()
print("[OK] Figure 5 saved")

# =========================================================
# Fig 6: PhoBERT vs BiLSTM
# =========================================================
model_names = ['PhoBERT\n(135M params)', 'BiLSTM\n(19M params)']
params_m    = [135.03, 19.03]
lat_ms      = [19.62, 1.18]
vram_mb     = [606.3, 171.3]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
clr_pairs = [(COLORS[0], COLORS[2]), (COLORS[1], COLORS[3]), (COLORS[4], COLORS[5])]

for vals, title, ylabel, (c1, c2), ax_ in zip(
        [params_m, lat_ms, vram_mb],
        ['So tham so (trieu)', 'Latency (ms/mau)', 'Peak VRAM (MB)'],
        ['Trieu tham so', 'ms/mau', 'MB'],
        clr_pairs,
        axes):
    bars = ax_.bar(model_names, vals, color=[c1, c2], edgecolor='white', width=0.5)
    for b, v in zip(bars, vals):
        ax_.text(b.get_x()+b.get_width()/2, b.get_height()*1.03, f'{v:.1f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_.set_title(title, fontsize=10)
    ax_.set_ylabel(ylabel)
    ax_.yaxis.grid(True, alpha=0.3, ls='--')
    ax_.set_axisbelow(True)

fig.suptitle('Hinh 4.6: So sanh PhoBERT vs BiLSTM  |  Tham so, Latency, VRAM  |  GPU RTX 2050 4GB', fontsize=11)
plt.tight_layout()
plt.savefig('ABSA_Project/figures/fig6_phobert_vs_bilstm.png')
plt.close()
print("[OK] Figure 6 saved")

# =========================================================
# Fig 7: Radar - Macro F1 per aspect
# =========================================================
vals_radar = list(f1mac / 100) + [f1mac[0] / 100]
num_vars = len(aspects)
angles = [n / num_vars * 2 * np.pi for n in range(num_vars)] + [0]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles, vals_radar, 'o-', linewidth=2.5, color=COLORS[0], markersize=6)
ax.fill(angles, vals_radar, alpha=0.2, color=COLORS[0])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(aspects, size=11)
ax.set_ylim(0, 1.0)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
ax.yaxis.grid(True, alpha=0.4)
ax.xaxis.grid(True, alpha=0.4)
ax.set_title('Hinh 4.7: Bieu do radar - Macro F1 (4 class) cua MultiHead-PhoBERT\ntren 10 khia canh san pham', pad=25)
plt.tight_layout()
plt.savefig('ABSA_Project/figures/fig7_radar_f1macro.png')
plt.close()
print("[OK] Figure 7 saved")

# =========================================================
# Summary
# =========================================================
print("\n=== DONE: All figures saved to ABSA_Project/figures/ ===")
for f in sorted(os.listdir('ABSA_Project/figures')):
    if f.endswith('.png'):
        p = f'ABSA_Project/figures/{f}'
        sz = os.path.getsize(p) / 1024
        print(f"  {f:<50s}  {sz:.1f} KB")
