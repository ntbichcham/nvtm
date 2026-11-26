import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, BoxStyle
import numpy as np
import os

# Create directories
os.makedirs('ABSA_Project/figures/report_chapter3', exist_ok=True)
os.makedirs('ABSA_Project/results/report_assets', exist_ok=True)

# Set base fonts and style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ==============================================================================
# HINH 3.8: QUY TRINH 8+1 BUOC TU DU LIEU THO DEN BRAND HEALTH INDEX
# ==============================================================================
def draw_figure_3_8():
    fig, ax = plt.subplots(figsize=(19, 10.5))
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    # Background canvas
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    # Title header
    ax.text(9.5, 9.9, "QUY TRÌNH PIPELINE TOÀN DIỆN TỪ DỮ LIỆU THÔ ĐẾN QUẢN TRỊ THƯƠNG HIỆU (BHI)",
            ha='center', va='center', fontsize=15, fontweight='bold', color='#0F172A')
    ax.text(9.5, 9.5, "Ánh xạ chi tiết 9 bước xử lý, kỹ thuật/giải thuật áp dụng, mã nguồn tương ứng và đề xuất cải tiến",
            ha='center', va='center', fontsize=11, fontstyle='italic', color='#475569')

    # Define the 9 steps arranged in a clean 3x3 grid or 3 rows flow
    # Row 1 (Steps 1, 2, 3): Preprocessing & Ingestion
    # Row 2 (Steps 4, 5, 6): Tokenization, Weights & Modeling
    # Row 3 (Steps 7, 8, 9): Optimization, Evaluation & BI Analytics
    
    steps = [
        # Step 1
        {
            "step": "BƯỚC 1", "title": "Tiếp nhận Dữ liệu & Kiểm định",
            "file": "check_dataset.py",
            "tech": "• Kiểm tra schema, missing values\n• Cross-split duplicate audit (MD5)\n• Kiểm tra nhãn & định dạng ngày",
            "io": "IN: Train/Dev/Test CSV (UIT-ViSFD)\nOUT: Cleaned Data + Quality Logs",
            "color": "#DBEAFE", "border": "#2563EB", "badge_col": "#1D4ED8",
            "improv": "Cải tiến: Tự động hóa data contract (Pydantic/Great Expectations)"
        },
        # Step 2
        {
            "step": "BƯỚC 2", "title": "Phân tích cú pháp nhãn ABSA",
            "file": "dataset_loader.py (parse_label_string)",
            "tech": "• Biểu thức chính quy (Regex re.findall)\n• Tách cặp {Aspect#Polarity}\n• Ánh xạ nhãn: None(0), Neg(1), Neu(2), Pos(3)",
            "io": "IN: Chuỗi ký tự nhãn thô\nOUT: Ma trận nhãn số Y ∈ {0..3}^(N×10)",
            "color": "#DBEAFE", "border": "#2563EB", "badge_col": "#1D4ED8",
            "improv": "Cải tiến: Hỗ trợ span extraction (vị trí từ ý kiến opinion term)"
        },
        # Step 3
        {
            "step": "BƯỚC 3", "title": "Tách từ & Tiền xử lý tiếng Việt",
            "file": "preprocess.py (VnCoreNLP/pyvi)",
            "tech": "• Tách từ ghép (Word Segmentation CRF)\n• Chuẩn hóa Unicode dựng sẵn (NFC)\n• Bảo tồn âm tiết tiếng Việt & teencode",
            "io": "IN: Văn bản thô (comment)\nOUT: Chuỗi tách từ (comment_segmented)",
            "color": "#E0E7FF", "border": "#4F46E5", "badge_col": "#4338CA",
            "improv": "Cải tiến: Bổ sung từ điển từ lóng công nghệ (Tech Teencode Dictionary)"
        },
        # Step 4
        {
            "step": "BƯỚC 4", "title": "Mã hóa Token & Tạo Tensor",
            "file": "ViSFDABSADataset.__getitem__",
            "tech": "• Byte-Pair Encoding (BPE PhoBERT)\n• Thêm token <s> ([CLS]) và </s> ([SEP])\n• Cắt cụt/đệm: max_len=160 (train) / 128 (deploy)",
            "io": "IN: comment_segmented\nOUT: input_ids (B,L), attention_mask (B,L)",
            "color": "#FEF3C7", "border": "#D97706", "badge_col": "#B45309",
            "improv": "Cải tiến: Dynamic padding theo batch để tiết kiệm thêm 20% FLOPs"
        },
        # Step 5
        {
            "step": "BƯỚC 5", "title": "Trọng số lớp làm mịn (Loss Weights)",
            "file": "compute_smoothed_class_weights",
            "tech": "• Nghịch đảo tần suất có lũy thừa γ=0.5:\n  w_(k,c) = (N / max(N_(k,c), 1))^0.5\n• Chuẩn hóa Σ w = C = 4 chống bùng nổ loss",
            "io": "IN: Phân bố nhãn tập Train\nOUT: 10 vector trọng số w_k ∈ R^4",
            "color": "#FEF3C7", "border": "#D97706", "badge_col": "#B45309",
            "improv": "Cải tiến: Khảo sát Focal Loss hoặc Class-Balanced Loss (Cui et al.)"
        },
        # Step 6
        {
            "step": "BƯỚC 6", "title": "Mô hình Multi-Head PhoBERT",
            "file": "model_phobert.py (MultiHeadPhoBERT)",
            "tech": "• Backbone: PhoBERT-base (12 layers, d=768)\n• Pooling [CLS] + Dropout (p=0.1)\n• 10 Linear Heads song song: R^768 → R^4",
            "io": "IN: input_ids, attention_mask, w_k\nOUT: Logits (B,10,4), Weighted Multi-Task CE Loss",
            "color": "#DCFCE7", "border": "#16A34A", "badge_col": "#15803D",
            "improv": "Cải tiến: Thêm Cross-Head Attention mô hình hóa tương quan khía cạnh"
        },
        # Step 7
        {
            "step": "BƯỚC 7", "title": "Tối ưu hóa & Huấn luyện",
            "file": "train_phobert.py (train)",
            "tech": "• AdamW (lr=2e-5, weight_decay=0.01)\n• Linear Warmup 10% + Cosine Decay\n• Mixed Precision FP16 (PyTorch AMP)\n• Early Stopping (patience=3 theo Dev Macro-F1)",
            "io": "IN: DataLoader, Model, Weights\nOUT: Checkpoint tốt nhất (best_model.pt), Log",
            "color": "#DCFCE7", "border": "#16A34A", "badge_col": "#15803D",
            "improv": "Cải tiến: Áp dụng Gradient Accumulation cho phép batch size ảo 32-64"
        },
        # Step 8
        {
            "step": "BƯỚC 8", "title": "Đánh giá Đa chiều & Phân tích lỗi",
            "file": "evaluate_absa.py / evaluate_and_analyze.py",
            "tech": "• Đánh giá ACD F1, ASC F1, Macro-F1, Micro-F1\n• Exact Match Ratio (EMR), Confusion Matrix\n• Phân loại lỗi: bỏ sót, cảnh báo nhầm, đảo cực",
            "io": "IN: best_model.pt + Test Set\nOUT: Metrics CSV, Confusion Matrix, Error Report",
            "color": "#F3E8FF", "border": "#9333EA", "badge_col": "#7E22CE",
            "improv": "Cải tiến: Kiểm định ý nghĩa thống kê tự động (Bootstrap/Permutation)"
        },
        # Step 9
        {
            "step": "BƯỚC 9", "title": "Tổng hợp Nghiệp vụ & Sức khỏe BHI",
            "file": "generate_figures.py / eda_visualization.py",
            "tech": "• Share of Voice: SOV(a) = N(a) / Σ N\n• Net Sentiment Score: NSS(a) ∈ [-100, +100]\n• Brand Health Index: BHI = Σ w(a) * (NSS+100)/2\n• Trực quan hóa Radar Chart, Dual-Axis Trend",
            "io": "IN: Bảng dự đoán đa khía cạnh\nOUT: Dashboard BHI, Radar NSS, Báo cáo BI",
            "color": "#F3E8FF", "border": "#9333EA", "badge_col": "#7E22CE",
            "improv": "Cải tiến: Tích hợp trọng số AHP đa tiêu chí và cảnh báo khủng hoảng realtime"
        }
    ]

    # Grid coordinates: 3 rows, 3 columns
    # Row 0: Steps 1, 2, 3 (left to right)
    # Row 1: Steps 6, 5, 4 (right to left for snake flow) or 4, 5, 6
    # Let's do standard left to right for each row with clear connecting routing arrows!
    xs = [0.8, 6.8, 12.8]
    ys = [6.4, 3.5, 0.6]  # bottom y of boxes
    w_box = 5.4
    h_box = 2.5

    box_coords = [
        (xs[0], ys[0]), (xs[1], ys[0]), (xs[2], ys[0]),
        (xs[0], ys[1]), (xs[1], ys[1]), (xs[2], ys[1]),
        (xs[0], ys[2]), (xs[1], ys[2]), (xs[2], ys[2])
    ]

    for idx, (b_x, b_y) in enumerate(box_coords):
        s = steps[idx]
        
        # Outer card box
        card = FancyBboxPatch((b_x, b_y), w_box, h_box,
                              boxstyle="round,pad=0.08,rounding_size=0.15",
                              facecolor=s["color"], edgecolor=s["border"],
                              linewidth=1.6, alpha=0.92, zorder=2)
        ax.add_patch(card)

        # Header Badge
        badge = FancyBboxPatch((b_x + 0.15, b_y + h_box - 0.42), 1.1, 0.32,
                               boxstyle="round,pad=0.03,rounding_size=0.08",
                               facecolor=s["badge_col"], edgecolor='none', zorder=3)
        ax.add_patch(badge)
        ax.text(b_x + 0.70, b_y + h_box - 0.26, s["step"],
                ha='center', va='center', fontsize=8.5, fontweight='bold', color='white', zorder=4)

        # Step Title
        ax.text(b_x + 1.35, b_y + h_box - 0.26, s["title"],
                ha='left', va='center', fontsize=9.8, fontweight='bold', color='#0F172A', zorder=4)

        # Script filename bar
        ax.text(b_x + 0.2, b_y + h_box - 0.58, f"File: {s['file']}",
                ha='left', va='center', fontsize=7.8, fontweight='semibold', color='#334155', family='monospace', zorder=4)

        # Horizontal separator line
        ax.plot([b_x + 0.2, b_x + w_box - 0.2], [b_y + h_box - 0.70, b_y + h_box - 0.70],
                color=s["border"], linewidth=0.7, alpha=0.5, zorder=3)

        # Technical algorithms & details
        ax.text(b_x + 0.2, b_y + h_box - 1.15, s["tech"],
                ha='left', va='top', fontsize=7.6, color='#1E293B', linespacing=1.25, zorder=4)

        # Input / Output
        ax.text(b_x + 0.2, b_y + 0.50, s["io"],
                ha='left', va='bottom', fontsize=7.2, fontweight='medium', color='#0369A1' if 'IN' in s["io"] else '#047857',
                linespacing=1.2, zorder=4)

        # Improvement proposal badge
        imp_box = FancyBboxPatch((b_x + 0.15, b_y + 0.08), w_box - 0.3, 0.30,
                                 boxstyle="round,pad=0.02,rounding_size=0.05",
                                 facecolor='#FFFFFF', edgecolor=s["border"],
                                 linewidth=0.8, linestyle='--', zorder=3)
        ax.add_patch(imp_box)
        ax.text(b_x + 0.25, b_y + 0.23, s["improv"],
                ha='left', va='center', fontsize=6.8, fontstyle='italic', color='#475569', zorder=4)

    # Draw Connecting Arrows
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=14, linewidth=1.8, color='#0284C7')
    
    # Row 1: 1 -> 2 -> 3
    ax.annotate('', xy=(box_coords[1][0], box_coords[0][1] + h_box/2),
                xytext=(box_coords[0][0] + w_box, box_coords[0][1] + h_box/2),
                arrowprops=arrow_kw, zorder=5)
    ax.annotate('', xy=(box_coords[2][0], box_coords[1][1] + h_box/2),
                xytext=(box_coords[1][0] + w_box, box_coords[1][1] + h_box/2),
                arrowprops=arrow_kw, zorder=5)

    # Route Row 1 -> Row 2: 3 down to 4
    # From bottom-center of Step 3 to top-center of Step 4
    ax.annotate('', xy=(box_coords[3][0] + w_box*0.8, box_coords[3][1] + h_box),
                xytext=(box_coords[2][0] + w_box*0.2, box_coords[2][1]),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, linewidth=1.8, color='#0284C7',
                                connectionstyle="arc3,rad=-0.15"), zorder=5)

    # Row 2: 4 -> 5 -> 6
    ax.annotate('', xy=(box_coords[4][0], box_coords[3][1] + h_box/2),
                xytext=(box_coords[3][0] + w_box, box_coords[3][1] + h_box/2),
                arrowprops=arrow_kw, zorder=5)
    ax.annotate('', xy=(box_coords[5][0], box_coords[4][1] + h_box/2),
                xytext=(box_coords[4][0] + w_box, box_coords[4][1] + h_box/2),
                arrowprops=arrow_kw, zorder=5)

    # Route Row 2 -> Row 3: 6 down to 7
    # From bottom-center of Step 6 to top-center of Step 7
    ax.annotate('', xy=(box_coords[6][0] + w_box*0.8, box_coords[6][1] + h_box),
                xytext=(box_coords[5][0] + w_box*0.2, box_coords[5][1]),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, linewidth=1.8, color='#0284C7',
                                connectionstyle="arc3,rad=-0.15"), zorder=5)

    # Row 3: 7 -> 8 -> 9
    ax.annotate('', xy=(box_coords[7][0], box_coords[6][1] + h_box/2),
                xytext=(box_coords[6][0] + w_box, box_coords[6][1] + h_box/2),
                arrowprops=arrow_kw, zorder=5)
    ax.annotate('', xy=(box_coords[8][0], box_coords[7][1] + h_box/2),
                xytext=(box_coords[7][0] + w_box, box_coords[7][1] + h_box/2),
                arrowprops=arrow_kw, zorder=5)

    plt.tight_layout()
    out1 = 'ABSA_Project/figures/report_chapter3/hinh_3_8_quy_trinh.png'
    out2 = 'ABSA_Project/results/report_assets/hinh_3_8_quy_trinh.png'
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Hình 3.8 saved to {out1} & {out2}")


# ==============================================================================
# HINH 3.9: KIEN TRUC MULTI-HEAD PHOBERT CHO ABSA 10 KHIA CANH
# ==============================================================================
def draw_figure_3_9():
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # Background canvas
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    # Title header
    ax.text(9.0, 10.5, "KIẾN TRÚC MULTI-HEAD PHOBERT CHO BÀI TOÁN ABSA 10 KHÍA CẠNH",
            ha='center', va='center', fontsize=15, fontweight='bold', color='#0F172A')
    ax.text(9.0, 10.1, "Biểu diễn toàn diện cấu trúc mạng nơ-ron, luồng tensor và cơ chế phân loại song song độc lập",
            ha='center', va='center', fontsize=11, fontstyle='italic', color='#475569')

    # 1. INPUT BLOCK (Left side or Bottom-to-Top flow)
    # We will arrange a clear vertical pipeline:
    # Level 1 (y: 0.8 - 1.8): Input & Tokenizer (Raw text -> VnCoreNLP -> PhoBERT Tokenizer)
    # Level 2 (y: 2.3 - 3.3): Input Embeddings (Word, Position, Token_type + LayerNorm)
    # Level 3 (y: 3.8 - 6.2): PhoBERT Backbone (12x Transformer Layers with Self-Attention & FFN)
    # Level 4 (y: 6.7 - 7.5): CLS Pooling & Regularization (Dropout p=0.1)
    # Level 5 (y: 8.0 - 9.6): 10 Parallel Linear Classification Heads & Loss

    # Level 1: Input text & Tokenizer
    box_input = FancyBboxPatch((1.5, 0.5), 15.0, 1.2,
                               boxstyle="round,pad=0.08,rounding_size=0.12",
                               facecolor='#EFF6FF', edgecolor='#3B82F6', linewidth=1.5, zorder=2)
    ax.add_patch(box_input)
    ax.text(1.8, 1.35, "1. KHỐI TIẾP NHẬN & XỬ LÝ ĐẦU VÀO (INPUT & TOKENIZATION)",
            fontsize=10.5, fontweight='bold', color='#1D4ED8')
    
    # Input stages inside Level 1
    sample_raw = "Văn bản thô: \"Pin dùng rất trâu nhưng màn hình hơi ám vàng...\""
    sample_seg = "VnCoreNLP Tách từ: \"Pin dùng rất trâu nhưng màn_hình hơi ám_vàng...\""
    sample_tok = "PhoBERT Tokenizer (max_len = 128, phủ 99.1%): <s> Pin dùng rất trâu ... </s> [PAD] ... [PAD]"
    ax.text(1.8, 1.05, sample_raw, fontsize=8.8, color='#334155')
    ax.text(1.8, 0.82, sample_seg, fontsize=8.8, color='#0F766E', fontweight='medium')
    ax.text(1.8, 0.60, sample_tok, fontsize=8.2, color='#4338CA', family='monospace')

    # Tensor badges
    badge_t1 = FancyBboxPatch((12.8, 0.95), 3.4, 0.32, boxstyle="round,pad=0.02,rounding_size=0.06",
                              facecolor='#1E40AF', edgecolor='none', zorder=3)
    ax.add_patch(badge_t1)
    ax.text(14.5, 1.11, "input_ids ∈ {Z}^(B × 128)", color='white', fontsize=8.0, fontweight='bold', ha='center', va='center')

    badge_t2 = FancyBboxPatch((12.8, 0.58), 3.4, 0.32, boxstyle="round,pad=0.02,rounding_size=0.06",
                              facecolor='#0284C7', edgecolor='none', zorder=3)
    ax.add_patch(badge_t2)
    ax.text(14.5, 0.74, "attention_mask ∈ {0,1}^(B × 128)", color='white', fontsize=8.0, fontweight='bold', ha='center', va='center')

    # Arrow L1 -> L2
    ax.annotate('', xy=(9.0, 2.0), xytext=(9.0, 1.7),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=12, linewidth=1.8, color='#2563EB'))

    # Level 2: Embeddings Layer
    box_emb = FancyBboxPatch((2.5, 2.0), 13.0, 1.05,
                             boxstyle="round,pad=0.08,rounding_size=0.12",
                             facecolor='#F1F5F9', edgecolor='#64748B', linewidth=1.5, zorder=2)
    ax.add_patch(box_emb)
    ax.text(2.8, 2.78, "2. TẦNG NHÚNG (PHOBERT EMBEDDINGS LAYER - 49.55M THAM SỐ)",
            fontsize=10.0, fontweight='bold', color='#334155')
    ax.text(2.8, 2.45, "• Word Embeddings: Vocab ~64,000 × 768   • Position Embeddings: 514 × 768   • Token Type Embeddings: 1 × 768",
            fontsize=8.5, color='#475569')
    ax.text(2.8, 2.18, "• Tổng hợp: E = LayerNorm(E_word + E_pos + E_tok) + Dropout(p=0.1)  →  Kích thước Tensor: (B × 128 × 768)",
            fontsize=8.5, color='#0369A1', fontweight='medium')

    # Arrow L2 -> L3
    ax.annotate('', xy=(9.0, 3.35), xytext=(9.0, 3.05),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=12, linewidth=1.8, color='#2563EB'))

    # Level 3: PhoBERT Backbone (12 Transformer Blocks)
    box_trans = FancyBboxPatch((1.5, 3.35), 15.0, 2.5,
                               boxstyle="round,pad=0.08,rounding_size=0.15",
                               facecolor='#ECFDF5', edgecolor='#059669', linewidth=1.8, zorder=2)
    ax.add_patch(box_trans)
    ax.text(1.8, 5.60, "3. BACKBONE TRANSFOMER (PHOBERT-BASE: 12 LAYERS, d = 768, 12 HEADS - 85.45M THAM SỐ)",
            fontsize=10.5, fontweight='bold', color='#065F46')

    # Draw 3 sample visual transformer layer sub-cards inside
    layer_w = 4.4
    layer_h = 1.8
    layer_xs = [2.0, 6.8, 11.6]
    layer_names = ["Tầng 1 (Low-level syntax)", "Tầng 2..11 (Contextual semantic)", "Tầng 12 (High-level ABSA rep)"]
    
    for lx, lname in zip(layer_xs, layer_names):
        sub_card = FancyBboxPatch((lx, 3.55), layer_w, layer_h,
                                  boxstyle="round,pad=0.04,rounding_size=0.08",
                                  facecolor='#FFFFFF', edgecolor='#10B981', linewidth=1.2, zorder=3)
        ax.add_patch(sub_card)
        ax.text(lx + layer_w/2, 5.15, lname, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#047857')
        
        # Sub-blocks inside layer
        ax.text(lx + 0.2, 4.80, "• Multi-Head Self-Attention:\n  12 heads, d_k = 64 (d = 768)", fontsize=7.6, color='#1E293B')
        ax.text(lx + 0.2, 4.35, "• Residual Connection & LayerNorm\n  Norm(x + Sublayer(x))", fontsize=7.4, color='#475569')
        ax.text(lx + 0.2, 3.90, "• Feed-Forward Network (GELU):\n  768 → 3072 → 768 + LayerNorm", fontsize=7.6, color='#1E293B')

    # Arrow inside Level 3
    ax.annotate('', xy=(6.8, 4.45), xytext=(6.4, 4.45),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=10, linewidth=1.5, color='#059669'))
    ax.annotate('', xy=(11.6, 4.45), xytext=(11.2, 4.45),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=10, linewidth=1.5, color='#059669'))

    # Arrow L3 -> L4
    ax.annotate('', xy=(9.0, 6.15), xytext=(9.0, 5.85),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=12, linewidth=1.8, color='#059669'))

    # Level 4: Pooling & Dropout
    box_pool = FancyBboxPatch((3.5, 6.15), 11.0, 0.85,
                              boxstyle="round,pad=0.06,rounding_size=0.10",
                              facecolor='#FEF3C7', edgecolor='#D97706', linewidth=1.5, zorder=2)
    ax.add_patch(box_pool)
    ax.text(3.8, 6.75, "4. POOLING [CLS] & REGULARIZATION", fontsize=9.8, fontweight='bold', color='#92400E')
    ax.text(3.8, 6.45, "• Trích xuất vector toàn câu tại token <s> / [CLS] ở vị trí 0: h_[CLS] = last_hidden_state[:, 0, :] ∈ R^(B × 768)",
            fontsize=8.4, color='#78350F')
    ax.text(3.8, 6.25, "• Áp dụng Dropout (p = 0.1) chống quá khớp trong quá trình huấn luyện: h_rep = Dropout(h_[CLS])",
            fontsize=8.4, color='#B45309', fontweight='medium')

    # Arrow L4 -> L5 (fan out to 10 heads)
    ax.annotate('', xy=(9.0, 7.30), xytext=(9.0, 7.00),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=12, linewidth=1.8, color='#D97706'))

    # Level 5: 10 Parallel Linear Classification Heads
    box_heads = FancyBboxPatch((0.5, 7.30), 17.0, 2.5,
                               boxstyle="round,pad=0.08,rounding_size=0.15",
                               facecolor='#FAF5FF', edgecolor='#9333EA', linewidth=1.8, zorder=2)
    ax.add_patch(box_heads)
    ax.text(0.8, 9.60, "5. MULTI-HEAD CLASSIFIER: 10 ĐẦU PHÂN LOẠI TUYẾN TÍNH SONG SONG ĐỘC LẬP (30,760 THAM SỐ)",
            fontsize=10.5, fontweight='bold', color='#6B21A8')

    aspects_list = [
        ("BATTERY", "#3B82F6"), ("CAMERA", "#0D9488"), ("DESIGN", "#10B981"),
        ("FEATURES", "#F59E0B"), ("GENERAL", "#6366F1"), ("PERFORMANCE", "#EC4899"),
        ("PRICE", "#8B5CF6"), ("SCREEN", "#14B8A6"), ("SER&ACC", "#F97316"),
        ("STORAGE", "#EF4444")
    ]

    head_w = 1.52
    head_h = 1.70
    start_x = 0.8
    gap_x = 0.15

    for i, (asp_name, col) in enumerate(aspects_list):
        hx = start_x + i * (head_w + gap_x)
        hy = 7.50
        
        # Individual Head Card
        hcard = FancyBboxPatch((hx, hy), head_w, head_h,
                               boxstyle="round,pad=0.03,rounding_size=0.06",
                               facecolor='#FFFFFF', edgecolor=col, linewidth=1.4, zorder=3)
        ax.add_patch(hcard)

        # Head title badge
        hbadge = FancyBboxPatch((hx + 0.05, hy + head_h - 0.28), head_w - 0.1, 0.24,
                                boxstyle="round,pad=0.01,rounding_size=0.04",
                                facecolor=col, edgecolor='none', zorder=4)
        ax.add_patch(hbadge)
        ax.text(hx + head_w/2, hy + head_h - 0.16, asp_name,
                ha='center', va='center', fontsize=7.0, fontweight='bold', color='white', zorder=5)

        # Projection details
        ax.text(hx + head_w/2, hy + head_h - 0.45, "Linear(768 → 4)",
                ha='center', va='center', fontsize=6.8, family='monospace', color='#334155', zorder=4)
        ax.text(hx + head_w/2, hy + head_h - 0.65, "z_k ∈ R^(B × 4)",
                ha='center', va='center', fontsize=6.8, family='monospace', color='#0284C7', zorder=4)

        # Softmax & 4 Classes
        ax.plot([hx + 0.1, hx + head_w - 0.1], [hy + 0.85, hy + 0.85], color='#E2E8F0', lw=0.8, zorder=3)
        ax.text(hx + head_w/2, hy + 0.70, "Softmax 4 lớp:", ha='center', va='center', fontsize=6.5, color='#475569', zorder=4)
        ax.text(hx + head_w/2, hy + 0.52, "0: None", ha='center', va='center', fontsize=6.2, color='#64748B', zorder=4)
        ax.text(hx + head_w/2, hy + 0.38, "1: Negative", ha='center', va='center', fontsize=6.2, color='#DC2626', fontweight='bold', zorder=4)
        ax.text(hx + head_w/2, hy + 0.24, "2: Neutral", ha='center', va='center', fontsize=6.2, color='#D97706', zorder=4)
        ax.text(hx + head_w/2, hy + 0.10, "3: Positive", ha='center', va='center', fontsize=6.2, color='#16A34A', fontweight='bold', zorder=4)

    # Multi-task Loss footer bar
    loss_box = FancyBboxPatch((0.8, 7.35), 16.4, 0.42,
                              boxstyle="round,pad=0.03,rounding_size=0.06",
                              facecolor='#F3E8FF', edgecolor='#9333EA', linewidth=1.0, linestyle='--', zorder=3)
    ax.add_patch(loss_box)
    loss_formula = r"Tổng hàm mất mát đa nhiệm: $\mathcal{L} = \sum_{k=1}^{10} \mathcal{L}_{CE}(z_k, y_k; w_k)$ với trọng số làm mịn $w_k = (N / N_{k,c})^{0.5}$"
    ax.text(9.0, 7.56, loss_formula, ha='center', va='center', fontsize=8.4, fontweight='semibold', color='#581C87', zorder=4)

    plt.tight_layout()
    out1 = 'ABSA_Project/figures/report_chapter3/hinh_3_9_kien_truc.png'
    out2 = 'ABSA_Project/results/report_assets/hinh_3_9_kien_truc.png'
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Hình 3.9 saved to {out1} & {out2}")

if __name__ == '__main__':
    draw_figure_3_8()
    draw_figure_3_9()
