"""
UAV Position Optimization for 6G Network Sum-Rate Maximization
================================================================
Streamlit application simulating three UAV placement algorithms:
  1. Stationary  – UAV hovers at the origin (0, 0).
  2. Greedy      – UAV flies toward the farthest ground user.
  3. Optimized   – Exhaustive grid search for the (x, y) that maximizes
                   the total Shannon capacity across all ground users.

Author : Brady – DTU CS-723 Advanced Wireless Networks
Date   : 2026-05-16
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch

# ────────────────────────────────────────────────────────────────────
# Page configuration
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAV 6G Sum-Rate Optimization",
    page_icon="🛩️",
    layout="wide",
)

# ────────────────────────────────────────────────────────────────────
# Custom CSS for a polished look
# ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 20px 24px;
        color: #f1f5f9;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,.25);
        margin-bottom: 8px;
    }
    .metric-card .label { font-size: .82rem; color: #94a3b8; margin-bottom: 4px; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; }

    /* Section header */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 16px 0 8px 0;
        padding: 8px 14px;
        border-left: 4px solid #3b82f6;
        background: rgba(59,130,246,.08);
        border-radius: 4px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────
# Helper: Shannon Capacity
# ────────────────────────────────────────────────────────────────────
# Constants
BANDWIDTH_HZ = 20e6          # 20 MHz channel bandwidth
NOISE_POWER_DBM = -100       # thermal noise (dBm)
FREQ_GHZ = 28                # mmWave carrier for 6G
PATH_LOSS_EXP = 2.2          # path-loss exponent (LoS UAV)
LIGHT_SPEED = 3e8
NOISE_POWER_W = 10 ** ((NOISE_POWER_DBM - 30) / 10)


def _fspl_ref(freq_ghz: float) -> float:
    """Free-space path loss at 1 m reference distance (linear)."""
    lam = LIGHT_SPEED / (freq_ghz * 1e9)
    return (4 * np.pi / lam) ** 2


FSPL_REF = _fspl_ref(FREQ_GHZ)


def compute_sum_rate(
    uav_x: float,
    uav_y: float,
    uav_h: float,
    users: np.ndarray,
    power_dbm: float,
) -> float:
    """Return total downlink Shannon capacity (bps) for all users."""
    power_w = 10 ** ((power_dbm - 30) / 10)
    dx = users[:, 0] - uav_x
    dy = users[:, 1] - uav_y
    dist_3d = np.sqrt(dx**2 + dy**2 + uav_h**2)
    # Path-loss (linear)
    pl = FSPL_REF * dist_3d ** PATH_LOSS_EXP
    snr = (power_w / pl) / NOISE_POWER_W
    capacity = BANDWIDTH_HZ * np.log2(1 + snr)
    return float(np.sum(capacity))


def compute_per_user_rate(
    uav_x: float,
    uav_y: float,
    uav_h: float,
    users: np.ndarray,
    power_dbm: float,
) -> np.ndarray:
    """Return per-user downlink capacity array (bps)."""
    power_w = 10 ** ((power_dbm - 30) / 10)
    dx = users[:, 0] - uav_x
    dy = users[:, 1] - uav_y
    dist_3d = np.sqrt(dx**2 + dy**2 + uav_h**2)
    pl = FSPL_REF * dist_3d ** PATH_LOSS_EXP
    snr = (power_w / pl) / NOISE_POWER_W
    return BANDWIDTH_HZ * np.log2(1 + snr)


# ────────────────────────────────────────────────────────────────────
# Algorithms
# ────────────────────────────────────────────────────────────────────
def algo_stationary(users: np.ndarray):
    """UAV stays at (0, 0)."""
    return 0.0, 0.0


def algo_greedy(users: np.ndarray):
    """UAV flies toward the farthest user (from origin)."""
    dists = np.linalg.norm(users, axis=1)
    farthest = users[np.argmax(dists)]
    # Move 70 % toward the farthest user (heuristic)
    return float(farthest[0] * 0.7), float(farthest[1] * 0.7)


def algo_optimized(users: np.ndarray, uav_h: float, power_dbm: float, grid_n: int = 120):
    """Exhaustive 2-D grid search for max sum-rate."""
    xs = np.linspace(-60, 60, grid_n)
    ys = np.linspace(-60, 60, grid_n)
    best_rate = -1.0
    best_x, best_y = 0.0, 0.0
    for x in xs:
        for y in ys:
            r = compute_sum_rate(x, y, uav_h, users, power_dbm)
            if r > best_rate:
                best_rate = r
                best_x, best_y = x, y
    return float(best_x), float(best_y)


# ────────────────────────────────────────────────────────────────────
# Sidebar controls
# ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛩️ Cấu hình mô phỏng")
    st.markdown("---")

    num_users = st.slider(
        "👥 Số người dùng (Ground Users)",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
    )

    uav_height = st.slider(
        "📏 Độ cao UAV (m)",
        min_value=10,
        max_value=200,
        value=50,
        step=5,
    )

    tx_power = st.slider(
        "⚡ Công suất phát (dBm)",
        min_value=10,
        max_value=46,
        value=30,
        step=1,
    )

    seed = st.number_input(
        "🎲 Random seed",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
    )

    grid_resolution = st.slider(
        "🔍 Độ phân giải lưới tìm kiếm",
        min_value=40,
        max_value=200,
        value=120,
        step=10,
        help="Số điểm mỗi chiều cho thuật toán Tối ưu (lưới NxN).",
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#64748b;font-size:.78rem;'>"
        "DTU – CS 723<br>Advanced Wireless Networks<br>© 2026</div>",
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────────
# Generate user positions
# ────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(int(seed))
users = rng.uniform(-50, 50, size=(num_users, 2))  # 100 × 100 m area centred at origin

# ────────────────────────────────────────────────────────────────────
# Run algorithms
# ────────────────────────────────────────────────────────────────────
pos_stat = algo_stationary(users)
pos_greed = algo_greedy(users)
pos_opt = algo_optimized(users, uav_height, tx_power, grid_n=grid_resolution)

rate_stat = compute_sum_rate(*pos_stat, uav_height, users, tx_power)
rate_greed = compute_sum_rate(*pos_greed, uav_height, users, tx_power)
rate_opt = compute_sum_rate(*pos_opt, uav_height, users, tx_power)

rates_per_user_stat = compute_per_user_rate(*pos_stat, uav_height, users, tx_power)
rates_per_user_greed = compute_per_user_rate(*pos_greed, uav_height, users, tx_power)
rates_per_user_opt = compute_per_user_rate(*pos_opt, uav_height, users, tx_power)

# Convert to Mbps for display
to_mbps = 1e-6

# ────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>🛩️ Tối ưu hóa vị trí UAV cho mạng 6G</h1>"
    "<p style='text-align:center;color:#94a3b8;margin-top:-8px;'>"
    "So sánh 3 thuật toán: <b>Đứng yên</b> · <b>Tham lam</b> · <b>Tối ưu hóa</b></p>",
    unsafe_allow_html=True,
)
st.markdown("")

# ────────────────────────────────────────────────────────────────────
# KPI metric cards
# ────────────────────────────────────────────────────────────────────
cols_kpi = st.columns(3)

algo_names = ["📍 Đứng yên (0,0)", "🚀 Tham lam", "🎯 Tối ưu hóa"]
algo_rates = [rate_stat, rate_greed, rate_opt]
algo_colors = ["#ef4444", "#f59e0b", "#22c55e"]

for col, name, rate, color in zip(cols_kpi, algo_names, algo_rates, algo_colors):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{name}</div>
            <div class="value" style="color:{color};">{rate * to_mbps:.2f} Mbps</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# ────────────────────────────────────────────────────────────────────
# Charts
# ────────────────────────────────────────────────────────────────────
col_scatter, col_bar = st.columns([1.2, 1])

# —— 1. Scatter plot ——
with col_scatter:
    st.markdown('<div class="section-header">🗺️ Bản đồ vị trí UAV & Người dùng</div>', unsafe_allow_html=True)

    fig1, ax1 = plt.subplots(figsize=(7, 7), facecolor="#0f172a")
    ax1.set_facecolor("#1e293b")

    # Grid
    ax1.grid(True, color="#334155", linewidth=0.5, linestyle="--", alpha=0.6)

    # Ground users
    ax1.scatter(
        users[:, 0],
        users[:, 1],
        s=130,
        c="#38bdf8",
        edgecolors="#0ea5e9",
        linewidths=1.5,
        zorder=5,
        label="Người dùng (GU)",
        marker="o",
    )
    for i, (ux, uy) in enumerate(users):
        ax1.annotate(
            f"U{i+1}",
            (ux, uy),
            textcoords="offset points",
            xytext=(7, 7),
            fontsize=8,
            color="#7dd3fc",
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="#0f172a")],
        )

    # UAV positions
    uav_data = [
        (pos_stat, "#ef4444", "D", "Đứng yên"),
        (pos_greed, "#f59e0b", "s", "Tham lam"),
        (pos_opt, "#22c55e", "^", "Tối ưu"),
    ]
    for (px, py), color, marker, label in uav_data:
        ax1.scatter(
            px, py, s=260, c=color, edgecolors="white", linewidths=2,
            marker=marker, zorder=10, label=f"UAV – {label}",
        )
        # Draw faint lines from UAV to each user
        for ux, uy in users:
            ax1.plot(
                [px, ux], [py, uy],
                color=color, alpha=0.12, linewidth=0.8, zorder=2,
            )

    ax1.set_xlim(-60, 60)
    ax1.set_ylim(-60, 60)
    ax1.set_xlabel("X (m)", color="#94a3b8", fontsize=11)
    ax1.set_ylabel("Y (m)", color="#94a3b8", fontsize=11)
    ax1.set_title(
        f"Khu vực phủ sóng – {num_users} người dùng · Cao {uav_height} m",
        color="#e2e8f0", fontsize=12, fontweight="bold", pad=14,
    )
    ax1.tick_params(colors="#64748b")
    for spine in ax1.spines.values():
        spine.set_color("#334155")

    legend = ax1.legend(
        loc="upper left", fontsize=8, frameon=True,
        facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0",
    )
    legend.get_frame().set_alpha(0.85)

    st.pyplot(fig1)
    plt.close(fig1)

# —— 2. Bar chart ——
with col_bar:
    st.markdown('<div class="section-header">📊 So sánh Sum-Rate (Mbps)</div>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(6, 5.5), facecolor="#0f172a")
    ax2.set_facecolor("#1e293b")

    bar_labels = ["Đứng yên\n(0,0)", "Tham lam\n(Greedy)", "Tối ưu\n(Optimized)"]
    bar_values = [r * to_mbps for r in algo_rates]
    bar_colors_gradient = ["#ef4444", "#f59e0b", "#22c55e"]

    bars = ax2.bar(
        bar_labels,
        bar_values,
        color=bar_colors_gradient,
        edgecolor="white",
        linewidth=1.2,
        width=0.55,
        zorder=3,
    )

    # Value labels on top of bars
    for bar, val in zip(bars, bar_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(bar_values) * 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#f1f5f9",
            path_effects=[pe.withStroke(linewidth=2, foreground="#0f172a")],
        )

    ax2.set_ylabel("Tổng Sum-Rate (Mbps)", color="#94a3b8", fontsize=11)
    ax2.set_title(
        "Hiệu năng 3 thuật toán",
        color="#e2e8f0", fontsize=13, fontweight="bold", pad=14,
    )
    ax2.tick_params(colors="#94a3b8")
    ax2.grid(axis="y", color="#334155", linewidth=0.5, linestyle="--", alpha=0.5)
    for spine in ax2.spines.values():
        spine.set_color("#334155")
    ax2.set_axisbelow(True)

    # % improvement annotation
    if rate_stat > 0:
        pct_greed = (rate_greed - rate_stat) / rate_stat * 100
        pct_opt = (rate_opt - rate_stat) / rate_stat * 100
        ax2.annotate(
            f"+{pct_opt:.1f}%",
            xy=(2, bar_values[2]),
            xytext=(2, bar_values[2] + max(bar_values) * 0.12),
            ha="center",
            fontsize=10,
            color="#22c55e",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#22c55e", lw=1.5),
        )

    st.pyplot(fig2)
    plt.close(fig2)

    # —— Per-user breakdown table ——
    st.markdown('<div class="section-header">📋 Tốc độ mỗi người dùng (Mbps)</div>', unsafe_allow_html=True)

    import pandas as pd

    df_per_user = pd.DataFrame(
        {
            "Người dùng": [f"U{i+1}" for i in range(num_users)],
            "Đứng yên": np.round(rates_per_user_stat * to_mbps, 2),
            "Tham lam": np.round(rates_per_user_greed * to_mbps, 2),
            "Tối ưu": np.round(rates_per_user_opt * to_mbps, 2),
        }
    )
    st.dataframe(df_per_user, use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────────────────────────
# Per-user bar chart (detailed)
# ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Tốc độ từng người dùng theo thuật toán</div>', unsafe_allow_html=True)

fig3, ax3 = plt.subplots(figsize=(10, 4.5), facecolor="#0f172a")
ax3.set_facecolor("#1e293b")

x_indices = np.arange(num_users)
width = 0.25

ax3.bar(x_indices - width, rates_per_user_stat * to_mbps, width,
        color="#ef4444", edgecolor="white", linewidth=0.8, label="Đứng yên", zorder=3)
ax3.bar(x_indices, rates_per_user_greed * to_mbps, width,
        color="#f59e0b", edgecolor="white", linewidth=0.8, label="Tham lam", zorder=3)
ax3.bar(x_indices + width, rates_per_user_opt * to_mbps, width,
        color="#22c55e", edgecolor="white", linewidth=0.8, label="Tối ưu", zorder=3)

ax3.set_xticks(x_indices)
ax3.set_xticklabels([f"U{i+1}" for i in range(num_users)])
ax3.set_ylabel("Capacity (Mbps)", color="#94a3b8", fontsize=11)
ax3.set_title("Shannon Capacity – từng người dùng", color="#e2e8f0", fontsize=12, fontweight="bold", pad=12)
ax3.tick_params(colors="#94a3b8")
ax3.grid(axis="y", color="#334155", linewidth=0.5, linestyle="--", alpha=0.5)
for spine in ax3.spines.values():
    spine.set_color("#334155")
ax3.set_axisbelow(True)

legend3 = ax3.legend(fontsize=9, frameon=True, facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0")
legend3.get_frame().set_alpha(0.85)

st.pyplot(fig3)
plt.close(fig3)

# ────────────────────────────────────────────────────────────────────
# Technical details expander
# ────────────────────────────────────────────────────────────────────
with st.expander("📐 Chi tiết kỹ thuật & Tọa độ UAV"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📍 Đứng yên**")
        st.code(f"x = {pos_stat[0]:.2f} m\ny = {pos_stat[1]:.2f} m\nh = {uav_height} m")
    with c2:
        st.markdown("**🚀 Tham lam**")
        st.code(f"x = {pos_greed[0]:.2f} m\ny = {pos_greed[1]:.2f} m\nh = {uav_height} m")
    with c3:
        st.markdown("**🎯 Tối ưu**")
        st.code(f"x = {pos_opt[0]:.2f} m\ny = {pos_opt[1]:.2f} m\nh = {uav_height} m")

    st.markdown("---")
    st.markdown(
        f"""
        | Thông số | Giá trị |
        |----------|---------|
        | Băng thông kênh | {BANDWIDTH_HZ/1e6:.0f} MHz |
        | Tần số sóng mang | {FREQ_GHZ} GHz (mmWave) |
        | Công suất nhiễu | {NOISE_POWER_DBM} dBm |
        | Hệ số suy hao đường truyền | {PATH_LOSS_EXP} |
        | Kích thước vùng phủ | 100 × 100 m |
        | Công suất phát | {tx_power} dBm |
        """
    )
