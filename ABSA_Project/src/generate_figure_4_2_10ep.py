import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Create destination folder if not exists
os.makedirs('ABSA_Project/figures/report_member3', exist_ok=True)
os.makedirs('ABSA_Project/results/report_assets', exist_ok=True)

# Load 10-epoch official log
csv_path = 'ABSA_Project/results/evaluation/train_log_official_ep10_seed42.csv'
df = pd.read_csv(csv_path)

epochs = df['epoch'].values
train_loss = df['train_loss'].values
dev_loss = df['dev_loss'].values
macro_f1 = df['dev_macro_f1_4class'].values * 100
sent_f1 = df['dev_sentiment_macro_f1'].values * 100
acd_f1 = df['dev_acd_f1'].values * 100
em = df['dev_exact_match'].values * 100

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
fig.patch.set_facecolor('#FAFAFA')

# -------------------------------------------------------------
# Subplot (a): Loss curves
# -------------------------------------------------------------
ax1.set_facecolor('#FFFFFF')
ax1.plot(epochs, train_loss, 'o-', color='#DC2626', linewidth=2.2, markersize=6, label='Train Loss (Multi-Task Weighted CE)')
ax1.plot(epochs, dev_loss, 's--', color='#2563EB', linewidth=2.2, markersize=6, label='Dev Loss')

# Vertical line at epoch 5 marking development baseline vs official 10-epoch
ax1.axvline(x=5, color='#94A3B8', linestyle=':', linewidth=1.8, label='Mốc 5 epoch (Baseline thăm dò)')
ax1.annotate('Baseline 5-epoch\nTrain Loss: 3.05 | Dev Loss: 3.90',
             xy=(5, 3.895), xytext=(2.6, 7.5),
             arrowprops=dict(arrowstyle="->", color='#475569', lw=1.2, connectionstyle="arc3,rad=-0.15"),
             fontsize=8.5, fontweight='semibold', color='#334155',
             bbox=dict(boxstyle="round,pad=0.3", fc="#F1F5F9", ec="#CBD5E1", lw=1))

# Annotation at epoch 10
ax1.annotate(f'Epoch 10 (Chính thức)\nTrain Loss: {train_loss[-1]:.2f} | Dev Loss: {dev_loss[-1]:.2f}',
             xy=(10, train_loss[-1]), xytext=(6.8, 1.8),
             arrowprops=dict(arrowstyle="->", color='#DC2626', lw=1.2, connectionstyle="arc3,rad=0.1"),
             fontsize=8.5, fontweight='semibold', color='#991B1B',
             bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#FECACA", lw=1))

ax1.set_title('(a) Diễn biến Hàm Mất mát (Loss) qua 10 Epochs', fontsize=12, fontweight='bold', pad=12, color='#0F172A')
ax1.set_xlabel('Epoch', fontsize=11, fontweight='semibold', labelpad=8)
ax1.set_ylabel('Loss', fontsize=11, fontweight='semibold', labelpad=8)
ax1.set_xticks(epochs)
ax1.set_ylim(0.5, 11.0)
ax1.grid(True, linestyle='--', alpha=0.5, color='#E2E8F0')
ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.95)

# Value labels on key points
for ep, tl, dl in zip([1, 5, 10], [train_loss[0], train_loss[4], train_loss[9]], [dev_loss[0], dev_loss[4], dev_loss[9]]):
    ax1.text(ep, tl + 0.35, f'{tl:.2f}', ha='center', va='bottom', fontsize=8, color='#B91C1C', fontweight='bold')
    ax1.text(ep, dl + 0.35, f'{dl:.2f}', ha='center', va='bottom', fontsize=8, color='#1D4ED8', fontweight='bold')

# -------------------------------------------------------------
# Subplot (b): Validation Metrics curves
# -------------------------------------------------------------
ax2.set_facecolor('#FFFFFF')
ax2.plot(epochs, acd_f1, '^-', color='#0D9488', linewidth=2.0, markersize=5.5, label='Dev ACD F1 (Phát hiện khía cạnh)')
ax2.plot(epochs, macro_f1, 'o-', color='#7C3AED', linewidth=2.4, markersize=6.5, label='Dev Macro-F1 (4 lớp, tiêu chí chính)')
ax2.plot(epochs, sent_f1, 'd--', color='#EA580C', linewidth=2.0, markersize=5.5, label='Dev Sentiment Macro-F1 (3 lớp có đề cập)')
ax2.plot(epochs, em, 'x:', color='#64748B', linewidth=1.6, markersize=5.5, label='Dev Exact Match Ratio (Khớp chính xác 100%)')

# Vertical line at epoch 5
ax2.axvline(x=5, color='#94A3B8', linestyle=':', linewidth=1.8, label='Mốc 5 epoch (Dev Macro-F1 = 78.95%)')

# Peak annotation at epoch 10
best_f1 = macro_f1[-1]
ax2.annotate(f'Epoch 10 Tốt nhất\nMacro-F1: {best_f1:.2f}%\nACD F1: {acd_f1[-1]:.2f}%',
             xy=(10, best_f1), xytext=(7.0, 68),
             arrowprops=dict(arrowstyle="->", color='#6D28D9', lw=1.4, connectionstyle="arc3,rad=-0.15"),
             fontsize=8.5, fontweight='bold', color='#5B21B6',
             bbox=dict(boxstyle="round,pad=0.3", fc="#F5F3FF", ec="#DDD6FE", lw=1))

ax2.set_title('(b) Diễn biến Chỉ số Đánh giá trên Tập Phát triển (%)', fontsize=12, fontweight='bold', pad=12, color='#0F172A')
ax2.set_xlabel('Epoch', fontsize=11, fontweight='semibold', labelpad=8)
ax2.set_ylabel('Điểm số (%)', fontsize=11, fontweight='semibold', labelpad=8)
ax2.set_xticks(epochs)
ax2.set_ylim(10, 100)
ax2.grid(True, linestyle='--', alpha=0.5, color='#E2E8F0')
ax2.legend(loc='lower right', fontsize=8.2, framealpha=0.95)

# Value labels on key points for Macro-F1
for ep, mf in zip([1, 3, 5, 7, 10], [macro_f1[0], macro_f1[2], macro_f1[4], macro_f1[6], macro_f1[9]]):
    ax2.text(ep, mf + 2.0, f'{mf:.1f}%', ha='center', va='bottom', fontsize=8, color='#6D28D9', fontweight='bold')

plt.suptitle('Hình 4.2. Diễn biến huấn luyện Multi-Head PhoBERT qua 10 Epochs chính thức\n(Minh chứng sự hội tụ liên tục từ mốc 5 epoch đến điểm tối ưu ở epoch 10)',
             fontsize=13, fontweight='bold', y=1.03, color='#0F172A')

plt.tight_layout()

out1 = 'ABSA_Project/figures/report_member3/hinh_4_2_duong_hoc.png'
out2 = 'ABSA_Project/results/report_assets/hinh_4_2_duong_hoc.png'
plt.savefig(out1, dpi=300, bbox_inches='tight')
plt.savefig(out2, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] 10-Epoch Figure 4.2 successfully generated:\n  - {out1}\n  - {out2}")
