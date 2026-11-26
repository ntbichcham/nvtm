import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Create target directories
os.makedirs('ABSA_Project/figures', exist_ok=True)
os.makedirs('ABSA_Project/results/report_assets', exist_ok=True)

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11.5
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['legend.fontsize'] = 9.0

# ==============================================================================
# DATASETS
# ==============================================================================

# 1. Five-seed data for project models (Unit: %)
# Seeds: 42, 123, 2024, 7, 777
phobert_10ep_acd = [90.54, 90.54, 90.44, 90.61, 90.53]  # Mean: 90.53, Std: 0.06
phobert_10ep_asc = [76.43, 75.98, 76.04, 76.43, 75.63]  # Mean: 76.10, Std: 0.34

bilstm_ft_acd    = [86.53, 86.69, 86.37, 86.52, 86.67]  # Mean: 86.55, Std: 0.13
bilstm_ft_asc    = [68.45, 69.95, 69.04, 69.16, 69.01]  # Mean: 69.12, Std: 0.54

linear_svc_acd   = [84.53] * 5                           # Deterministic: 84.53
linear_svc_asc   = [66.41] * 5                           # Deterministic: 66.41

# Literature Baselines from Table 4.23 (Phan et al., Thin et al.)
lit_baselines = [
    {"name": "Thin et al. - PhoBERT",     "acd": None,  "asc": 74.44, "color": "#1D4ED8", "ls": "--", "marker": "s"},
    {"name": "SA2SL (Phan et al.) Bi-LSTM", "acd": 84.48, "asc": 63.06, "color": "#D97706", "ls": "-.", "marker": "D"},
    {"name": "Thin et al. - CNN",         "acd": None,  "asc": 65.65, "color": "#059669", "ls": ":",  "marker": "^"},
    {"name": "Thin et al. - SVM",         "acd": None,  "asc": 64.84, "color": "#6B7280", "ls": ":",  "marker": "v"},
    {"name": "SA2SL (Phan et al.) LSTM",  "acd": 80.27, "asc": 52.13, "color": "#DC2626", "ls": "--", "marker": "x"},
    {"name": "SA2SL (Phan et al.) CNN",   "acd": 69.70, "asc": 27.16, "color": "#7C3AED", "ls": "-.", "marker": "+"},
]

# 2. Per-Aspect F1 (%) for the 10 aspects
aspects = ['Screen', 'Camera', 'Features', 'Battery', 'Performance', 'Storage', 'Design', 'Price', 'General', 'Ser&Acc']

# PhoBERT 10ep (Seed 42 / Mean) per aspect F1 (Macro-F1 4-class per aspect):
# BATTERY: 79.99, CAMERA: 85.05, DESIGN: 70.65, FEATURES: 75.67, GENERAL: 75.31,
# PERFORMANCE: 77.77, PRICE: 81.34, SCREEN: 72.75, SER&ACC: 66.39, STORAGE: 57.23
phobert_per_aspect = [72.75, 85.05, 75.67, 79.99, 77.77, 57.23, 70.65, 81.34, 75.31, 66.39]

# SA2SL (Phan et al. 2021) Table 5 per aspect ASC F1:
# Screen: 62.30, Camera: 74.69, Features: 64.77, Battery: 72.73, Performance: 66.75,
# Storage: 30.10, Design: 71.41, Price: 71.41, General: 65.72, Ser&Acc: 59.22
sa2sl_per_aspect = [62.30, 74.69, 64.77, 72.73, 66.75, 30.10, 71.41, 71.41, 65.72, 59.22]

# Linear SVC per aspect Macro-F1:
# BATTERY: 71.40, CAMERA: 72.48, DESIGN: 54.12, FEATURES: 61.35, GENERAL: 64.31,
# PERFORMANCE: 65.49, PRICE: 70.91, SCREEN: 57.07, SER&ACC: 55.20, STORAGE: 43.57
linear_svc_per_aspect = [57.07, 72.48, 61.35, 71.40, 65.49, 43.57, 54.12, 70.91, 64.31, 55.20]

# ==============================================================================
# PLOTTING FIGURE: 2 SUBPLOTS
# ==============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={'width_ratios': [1.1, 1.0]})

# ------------------------------------------------------------------------------
# PANEL A: 5-Seed Boxplot for Project Models vs. Literature Baselines
# ------------------------------------------------------------------------------
# Group positions:
# ACD: positions 1 (PhoBERT), 2 (BiLSTM), 3 (Linear SVC)
# ASC: positions 5 (PhoBERT), 6 (BiLSTM), 7 (Linear SVC)

positions_acd = [1.0, 1.8, 2.6]
positions_asc = [4.2, 5.0, 5.8]

box_data_acd = [phobert_10ep_acd, bilstm_ft_acd, linear_svc_acd]
box_data_asc = [phobert_10ep_asc, bilstm_ft_asc, linear_svc_asc]

colors = ['#1E40AF', '#0D9488', '#64748B']  # Navy Blue, Teal, Slate
model_labels = ['PhoBERT (10-ep)', 'BiLSTM+fastText', 'linear_svc']

# Plot ACD boxplots
bp_acd = ax1.boxplot(box_data_acd, positions=positions_acd, widths=0.45, patch_artist=True,
                     boxprops=dict(linewidth=1.3),
                     medianprops=dict(color='darkred', linewidth=1.8),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2),
                     flierprops=dict(marker='o', markersize=4, markerfacecolor='gray'))

# Plot ASC boxplots
bp_asc = ax1.boxplot(box_data_asc, positions=positions_asc, widths=0.45, patch_artist=True,
                     boxprops=dict(linewidth=1.3),
                     medianprops=dict(color='darkred', linewidth=1.8),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2),
                     flierprops=dict(marker='o', markersize=4, markerfacecolor='gray'))

for bp in [bp_acd, bp_asc]:
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

# Overlay individual seed points (jittered strip plot)
np.random.seed(42)
for pos_group, data_group in [(positions_acd, box_data_acd), (positions_asc, box_data_asc)]:
    for pos, vals in zip(pos_group, data_group):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax1.scatter(np.array([pos]*len(vals)) + jitter, vals, color='black', alpha=0.7, s=24, zorder=4)

# Add literature benchmark markers / horizontal reference levels
# For ACD (x range: 0.5 to 3.1)
ax1.hlines(84.48, 0.6, 3.0, colors='#D97706', linestyles='--', linewidth=1.4, label='SA2SL Bi-LSTM (84.48%)')
ax1.hlines(80.27, 0.6, 3.0, colors='#DC2626', linestyles=':', linewidth=1.4, label='SA2SL LSTM (80.27%)')
ax1.hlines(69.70, 0.6, 3.0, colors='#7C3AED', linestyles='-.', linewidth=1.4, label='SA2SL CNN (69.70%)')

# For ASC (x range: 3.8 to 6.2)
ax1.hlines(74.44, 3.8, 6.2, colors='#1D4ED8', linestyles='--', linewidth=1.5, label='Thin et al. PhoBERT (74.44%)')
ax1.hlines(65.65, 3.8, 6.2, colors='#059669', linestyles=':', linewidth=1.4, label='Thin et al. CNN (65.65%)')
ax1.hlines(64.84, 3.8, 6.2, colors='#6B7280', linestyles=':', linewidth=1.4, label='Thin et al. SVM (64.84%)')
ax1.hlines(63.06, 3.8, 6.2, colors='#D97706', linestyles='-.', linewidth=1.4, label='SA2SL Bi-LSTM (63.06%)')
ax1.hlines(52.13, 3.8, 6.2, colors='#DC2626', linestyles='--', linewidth=1.2, label='SA2SL LSTM (52.13%)')
ax1.hlines(27.16, 3.8, 6.2, colors='#7C3AED', linestyles='-.', linewidth=1.2, label='SA2SL CNN (27.16%)')

# Labeling Panel A
ax1.set_xticks([1.8, 5.0])
ax1.set_xticklabels(['Phát hiện khía cạnh (ACD)', 'Phân loại cảm xúc (ASC)'], fontweight='bold', fontsize=10.5)
ax1.set_ylabel('Macro F1-score (%)', fontweight='bold')
ax1.set_ylim(20, 96)
ax1.set_title('(a) Phân bố 5 Seed của Nhóm 2 vs. Các công bố cùng bộ dữ liệu', fontweight='bold', pad=12)

# Custom legend for Panel A models
patches_models = [mpatches.Patch(color=c, alpha=0.7, label=l) for c, l in zip(colors, model_labels)]
legend1 = ax1.legend(handles=patches_models, loc='lower left', frameon=True, facecolor='white', framealpha=0.9, title='Mô hình Nhóm 2')
ax1.add_artist(legend1)

# Text annotation highlighting PhoBERT dominance
ax1.annotate('PhoBERT vượt trội tuyệt đối\n(ACD: 90.53%, ASC: 76.10%)',
             xy=(1.0, 90.53), xytext=(1.4, 92.5),
             arrowprops=dict(arrowstyle="->", color='#1E40AF', lw=1.2),
             fontsize=8.8, fontweight='semibold', color='#1E40AF',
             bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec="#93C5FD", lw=1))

ax1.annotate('Vượt kỷ lục Thin et al.\n(+1.66% ASC)',
             xy=(4.2, 76.10), xytext=(4.45, 82.0),
             arrowprops=dict(arrowstyle="->", color='#1E40AF', lw=1.2),
             fontsize=8.8, fontweight='semibold', color='#1E40AF',
             bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec="#93C5FD", lw=1))


# ------------------------------------------------------------------------------
# PANEL B: Per-Aspect F1 Spread across 10 Aspects (PhoBERT vs SA2SL vs Baseline)
# ------------------------------------------------------------------------------
aspect_box_data = [phobert_per_aspect, sa2sl_per_aspect, linear_svc_per_aspect]
aspect_positions = [1.0, 1.8, 2.6]
aspect_colors = ['#1E40AF', '#D97706', '#64748B']
aspect_labels = ['PhoBERT\n(Nhóm 2)', 'SA2SL Bi-LSTM\n(Phan et al.)', 'linear_svc\n(Tham chiếu)']

bp_aspect = ax2.boxplot(aspect_box_data, positions=aspect_positions, widths=0.45, patch_artist=True,
                        boxprops=dict(linewidth=1.3),
                        medianprops=dict(color='darkred', linewidth=2.0),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker='o', markersize=5, markerfacecolor='red'))

for patch, c in zip(bp_aspect['boxes'], aspect_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.65)

# Overlay aspect points and annotate outliers / key aspects
np.random.seed(123)
for pos, vals, c in zip(aspect_positions, aspect_box_data, aspect_colors):
    jitter = np.random.normal(0, 0.035, size=len(vals))
    ax2.scatter(np.array([pos]*len(vals)) + jitter, vals, color=c, edgecolors='black', linewidth=0.5, alpha=0.85, s=36, zorder=4)

# Annotate Storage aspect disparity (the hardest rare class)
# In SA2SL: Storage is 30.10%
# In PhoBERT: Storage is 57.23%
ax2.annotate('Storage (30.10%)\nĐộ lệch nhãn hiếm rất cao',
             xy=(1.8, 30.10), xytext=(1.95, 23.0),
             arrowprops=dict(arrowstyle="->", color='#D97706', lw=1.2),
             fontsize=8.5, color='#B45309',
             bbox=dict(boxstyle="round,pad=0.3", fc="#FEF3C7", ec="#FCD34D", lw=1))

ax2.annotate('Storage (57.23%)\nPhoBERT kéo nhãn hiếm lên +27.13%',
             xy=(1.0, 57.23), xytext=(0.6, 47.0),
             arrowprops=dict(arrowstyle="->", color='#1E40AF', lw=1.2),
             fontsize=8.5, color='#1E40AF',
             bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec="#93C5FD", lw=1))

# Camera aspect (top performing in both)
ax2.annotate('Camera (85.05%)', xy=(1.0, 85.05), xytext=(1.1, 88.0),
             fontsize=8.0, color='#1E40AF', fontweight='bold')

ax2.set_xticks(aspect_positions)
ax2.set_xticklabels(aspect_labels, fontweight='bold', fontsize=9.5)
ax2.set_ylabel('Per-Aspect F1-score (%)', fontweight='bold')
ax2.set_ylim(20, 96)
ax2.set_title('(b) Phân bố độ biến thiên F1 qua 10 khía cạnh (Per-Aspect Spread)', fontweight='bold', pad=12)

# Grid & layout adjustment
plt.tight_layout()

# Save high-resolution figures
output_path_1 = 'ABSA_Project/figures/hinh_4_10_boxplot_bang_423.png'
output_path_2 = 'ABSA_Project/results/report_assets/boxplot_table_423.png'
plt.savefig(output_path_1, dpi=300, bbox_inches='tight')
plt.savefig(output_path_2, dpi=300, bbox_inches='tight')
plt.close()

print(f"Figure saved successfully to:\n1. {output_path_1}\n2. {output_path_2}")
