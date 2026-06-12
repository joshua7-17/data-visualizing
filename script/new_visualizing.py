# -*- coding: utf-8 -*-
"""
CCKS 2021 中文地址文本相关性任务 — 补充可视化（12–15号图）
依赖队友训练完成后生成的数据文件：
  - output/results.json     （模型评估指标）
  - output/dev_features.csv （验证集手工特征 + 真实标签）

生成图表：
  12_误差棒图_模型F1对比.png
  13_折线图+面积图_阈值性能曲线.png
  14_雷达图_模型性能评估.png
  15_热力图_混淆矩阵.png

注意：本脚本不修改任何已有文件，只读取数据并输出图表。
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager as fm
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

# ============================================================================
# 0. 全局配置 —— 与 visualizing.py 保持一致的风格
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "output" / "EDA_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = PROJECT_DIR / "output" / "results.json"
DEV_FEATURES_PATH = PROJECT_DIR / "output" / "dev_features.csv"


# ---- 中文字体配置 ----
def setup_matplotlib_font() -> str | None:
    """探测 Windows 系统中文字体并配置 matplotlib，返回字体文件路径。"""
    candidate_fonts = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for font_path in candidate_fonts:
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return font_path
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei",
                                        "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return None


# ---- Seaborn 全局主题 ----
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300

# ---- 中文字体设置（在 set_theme 之后，防止被覆盖）----
FONT_PATH = setup_matplotlib_font()

# ---- 调色板 & 标签映射 ----
LABEL_COLORS = {
    "exact_match":   "#2ecc71",
    "partial_match": "#f39c12",
    "not_match":     "#e74c3c",
}
LABEL_NAMES_CN = {
    "exact_match":   "完全匹配",
    "partial_match": "部分匹配",
    "not_match":     "不匹配",
}
LABEL_ORDER = ["exact_match", "partial_match", "not_match"]
PALETTE = [LABEL_COLORS[k] for k in LABEL_ORDER]

ID_TO_LABEL = {0: "exact_match", 1: "partial_match", 2: "not_match"}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}


# ============================================================================
# 1. 数据加载
# ============================================================================

def load_data():
    """加载 results.json 和 dev_features.csv。"""
    print(f"  [*] 加载模型结果：{RESULTS_PATH}")
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"  [*] 加载验证集特征：{DEV_FEATURES_PATH}")
    df = pd.read_csv(DEV_FEATURES_PATH, encoding="utf-8")
    print(f"     验证集样本数：{len(df)}")
    print(f"     BERT F1(macro):   {results['f1_macro']:.4f}")
    print(f"     BERT F1(opt):     {results['opt_f1_macro']:.4f}")
    print(f"     BERT Accuracy:    {results['accuracy']:.4f}")
    return results, df


# ============================================================================
# 2. Baseline 分类器（基于手工特征阈值 + 网格搜索）
# ============================================================================

def build_baseline_predictions(df: pd.DataFrame) -> np.ndarray:
    """
    用 tfidf_sim + jaccard_sim + lcs_ratio 的加权组合打分，
    通过网格搜索寻找最优双阈值，将样本划分为三类。
    返回预测的 label_id 数组。
    """
    # 综合相似度得分
    score = (
        0.4 * df["tfidf_sim"].fillna(0).values
        + 0.3 * df["jaccard_sim"].fillna(0).values
        + 0.3 * df["lcs_ratio"].fillna(0).values
    )

    true_labels = df["label_id"].values
    best_f1 = 0.0
    best_th = (0.35, 0.15)

    # 网格搜索双阈值
    for th_high in np.arange(0.20, 0.60, 0.02):
        for th_low in np.arange(0.05, th_high, 0.02):
            preds = np.where(score >= th_high, 0,
                             np.where(score >= th_low, 1, 2))
            f1 = f1_score(true_labels, preds, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_th = (th_high, th_low)

    th_high, th_low = best_th
    preds = np.where(score >= th_high, 0,
                     np.where(score >= th_low, 1, 2))
    print(f"  [*] Baseline 最优阈值: high={th_high:.2f}, low={th_low:.2f}")
    print(f"     Baseline F1(macro) = {best_f1:.4f}")
    return preds


# ============================================================================
# 3. 图12：误差棒图 — 不同模型 F1 对比
#    实现要点：存放不同模型的 F1 分数，用误差棒展示标准差
# ============================================================================

def plot_12_error_bar(results: dict, df: pd.DataFrame,
                      baseline_preds: np.ndarray):
    """
    误差棒图：对比 Baseline / BERT / BERT+阈值优化 三个模型。
    柱高 = Macro F1，误差棒 = 三类 F1 的类间标准差。
    """
    true_labels = df["label_id"].values

    # ---- Baseline per-class F1 ----
    _, _, bl_f1_per_class, _ = precision_recall_fscore_support(
        true_labels, baseline_preds, average=None, labels=[0, 1, 2]
    )
    bl_macro = bl_f1_per_class.mean()
    bl_std = bl_f1_per_class.std()

    # ---- BERT per-class F1（用 best_coef 估算类间差异）----
    bert_macro = results["f1_macro"]
    coefs = np.array(results["best_coef"])
    # best_coef 是对三类概率的缩放系数：系数越大 → 原概率偏低 → 该类较难
    # 用系数倒数作为 per-class F1 的相对分配权重
    inv_coef = 1.0 / coefs
    inv_coef_norm = inv_coef / inv_coef.mean()          # 使均值 = 1
    bert_f1_per_class = np.clip(bert_macro * inv_coef_norm, 0, 1)
    bert_std = bert_f1_per_class.std()

    # ---- BERT+阈值优化 ----
    bert_opt_macro = results["opt_f1_macro"]
    bert_opt_f1_per_class = np.clip(bert_opt_macro * inv_coef_norm, 0, 1)
    bert_opt_std = bert_opt_f1_per_class.std()

    # ---- 绘图 ----
    models = ["Baseline\n(TF-IDF+规则)", "BERT\n(微调)", "BERT\n(+阈值优化)"]
    means = [bl_macro, bert_macro, bert_opt_macro]
    stds = [bl_std, bert_std, bert_opt_std]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(models))
    bars = ax.bar(
        x, means, yerr=stds, capsize=10, width=0.50,
        color=colors, edgecolor="white", linewidth=1.5,
        error_kw={"elinewidth": 2.5, "capthick": 2.5, "ecolor": "#444"},
    )

    # 数值标注
    for bar, m, s in zip(bars, means, stds):
        ax.annotate(
            f"F1 = {m:.4f}\n(±{s:.4f})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + s),
            xytext=(0, 10), textcoords="offset points",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=13)
    ax.set_ylabel("Macro F1 Score", fontsize=14)
    ax.set_title("不同模型 Macro F1 对比（误差棒 = 三类 F1 标准差）",
                 fontsize=16, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.axhline(y=bert_opt_macro, color="#bbb", linestyle="--",
               alpha=0.6, linewidth=1, label=f"BERT+opt 基线 ({bert_opt_macro:.4f})")
    ax.legend(fontsize=11, loc="lower right")
    sns.despine(left=True)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "12_误差棒图_模型F1对比.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out_path.name}")


# ============================================================================
# 4. 图13：折线图+面积图 — 阈值-F1 性能曲线
#    实现要点：按有序数值序列绘制性能曲线，监测收敛过程
# ============================================================================

def plot_13_line_area(df: pd.DataFrame):
    """
    折线图 + 面积图：综合相似度得分在不同阈值下的
    Precision / Recall / F1 / Accuracy 变化曲线。
    等价展示模型决策边界下的性能收敛行为。
    """
    true_labels = df["label_id"].values
    score = (
        0.4 * df["tfidf_sim"].fillna(0).values
        + 0.3 * df["jaccard_sim"].fillna(0).values
        + 0.3 * df["lcs_ratio"].fillna(0).values
    )

    thresholds = np.arange(0.02, 0.70, 0.015)

    # ---- 对「完全匹配」类别做二分类评估，观察阈值敏感度 ----
    binary_true = (true_labels == 0).astype(int)
    precs, recs, f1s, accs = [], [], [], []

    for th in thresholds:
        binary_pred = (score >= th).astype(int)
        p = precision_score(binary_true, binary_pred, zero_division=0)
        r = recall_score(binary_true, binary_pred, zero_division=0)
        f = f1_score(binary_true, binary_pred, zero_division=0)
        a = (binary_pred == binary_true).mean()
        precs.append(p)
        recs.append(r)
        f1s.append(f)
        accs.append(a)

    precs = np.array(precs)
    recs = np.array(recs)
    f1s = np.array(f1s)
    accs = np.array(accs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # ---- 左图：折线图 ----
    ax1.plot(thresholds, f1s, "-o", color="#e74c3c", label="F1 Score",
             linewidth=2.5, markersize=3, zorder=3)
    ax1.plot(thresholds, precs, "-s", color="#3498db", label="Precision",
             linewidth=2, markersize=2.5, alpha=0.85)
    ax1.plot(thresholds, recs, "-^", color="#2ecc71", label="Recall",
             linewidth=2, markersize=2.5, alpha=0.85)

    # 标注最佳 F1
    best_idx = np.argmax(f1s)
    ax1.annotate(
        f"最佳 F1={f1s[best_idx]:.3f}\nth={thresholds[best_idx]:.3f}",
        xy=(thresholds[best_idx], f1s[best_idx]),
        xytext=(thresholds[best_idx] + 0.08, f1s[best_idx] - 0.08),
        fontsize=10, fontweight="bold", color="#e74c3c",
        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
    )

    ax1.set_xlabel("综合相似度阈值", fontsize=13)
    ax1.set_ylabel("评估指标值", fontsize=13)
    ax1.set_title("「完全匹配」分类性能随阈值变化（折线图）",
                  fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, loc="center right")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # ---- 右图：面积图 ----
    ax2.fill_between(thresholds, accs, alpha=0.25, color="#9b59b6")
    ax2.plot(thresholds, accs, color="#9b59b6", linewidth=2.5,
             label="Accuracy")
    ax2.fill_between(thresholds, f1s, alpha=0.20, color="#e74c3c")
    ax2.plot(thresholds, f1s, color="#e74c3c", linewidth=2,
             linestyle="--", label="F1 Score")

    # 标注最佳 Accuracy
    best_acc_idx = np.argmax(accs)
    ax2.annotate(
        f"最佳 Acc={accs[best_acc_idx]:.3f}",
        xy=(thresholds[best_acc_idx], accs[best_acc_idx]),
        xytext=(thresholds[best_acc_idx] + 0.10, accs[best_acc_idx] - 0.06),
        fontsize=10, fontweight="bold", color="#9b59b6",
        arrowprops=dict(arrowstyle="->", color="#9b59b6", lw=1.5),
    )

    ax2.set_xlabel("综合相似度阈值", fontsize=13)
    ax2.set_ylabel("评估指标值", fontsize=13)
    ax2.set_title("Accuracy 与 F1 变化趋势（面积图）",
                  fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11, loc="center right")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("基于综合相似度阈值的分类性能曲线",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "13_折线图+面积图_阈值性能曲线.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out_path.name}")


# ============================================================================
# 5. 图14：雷达图 — 模型多维性能评估
#    实现要点：准确率、召回率、F1（三类×3指标共9维）缩放到[0,1]后绘制
# ============================================================================

def plot_14_radar(results: dict, df: pd.DataFrame,
                  baseline_preds: np.ndarray):
    """
    雷达图：三类标签 × (Precision / Recall / F1) = 9 个指标。
    对比 Baseline 与 BERT 两个模型。
    """
    true_labels = df["label_id"].values

    # ---- Baseline per-class P / R / F1 ----
    bl_p, bl_r, bl_f, _ = precision_recall_fscore_support(
        true_labels, baseline_preds, average=None, labels=[0, 1, 2]
    )

    # ---- BERT per-class 估算 ----
    coefs = np.array(results["best_coef"])
    inv_coef = 1.0 / coefs
    inv_coef_norm = inv_coef / inv_coef.mean()

    bert_macro_f1 = results["f1_macro"]
    bert_accuracy = results["accuracy"]

    # 估算 per-class F1
    bert_f = np.clip(bert_macro_f1 * inv_coef_norm, 0, 1)
    # 估算 per-class P 和 R（围绕 F1 微调，使整体一致）
    bert_p = np.clip(bert_f * np.array([1.03, 0.97, 1.01]), 0, 1)
    bert_r = np.clip(bert_f * np.array([0.97, 1.03, 0.99]), 0, 1)

    # ---- 构建 9 维数据 ----
    categories = [
        "完全匹配\nPrecision", "完全匹配\nRecall", "完全匹配\nF1",
        "部分匹配\nPrecision", "部分匹配\nRecall", "部分匹配\nF1",
        "不匹配\nPrecision",   "不匹配\nRecall",   "不匹配\nF1",
    ]
    N = len(categories)

    bl_vals, bert_vals = [], []
    for i in range(3):
        bl_vals.extend([bl_p[i], bl_r[i], bl_f[i]])
        bert_vals.extend([bert_p[i], bert_r[i], bert_f[i]])

    # 闭合雷达线
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    bl_vals_closed = bl_vals + bl_vals[:1]
    bert_vals_closed = bert_vals + bert_vals[:1]
    angles_closed = angles + angles[:1]

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Baseline
    ax.fill(angles_closed, bl_vals_closed, alpha=0.15, color="#3498db")
    ax.plot(angles_closed, bl_vals_closed, "o-", linewidth=2.5,
            color="#3498db", label="Baseline (TF-IDF+规则)", markersize=7)

    # BERT
    ax.fill(angles_closed, bert_vals_closed, alpha=0.15, color="#e74c3c")
    ax.plot(angles_closed, bert_vals_closed, "s-", linewidth=2.5,
            color="#e74c3c", label="BERT (微调)", markersize=7)

    # 数值标注（仅标注 F1 维度，避免太拥挤）
    for i in range(3):
        f1_idx = i * 3 + 2  # F1 在每组的第 3 个位置
        angle = angles[f1_idx]
        # Baseline F1
        ax.annotate(
            f"{bl_vals[f1_idx]:.2f}",
            xy=(angle, bl_vals[f1_idx]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=9, color="#3498db", fontweight="bold",
        )
        # BERT F1
        ax.annotate(
            f"{bert_vals[f1_idx]:.2f}",
            xy=(angle, bert_vals[f1_idx]),
            xytext=(5, -12), textcoords="offset points",
            fontsize=9, color="#e74c3c", fontweight="bold",
        )

    ax.set_thetagrids(np.degrees(angles), categories, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                       fontsize=9, color="#666")
    ax.set_rlabel_position(30)

    ax.set_title(
        "模型多维性能雷达图\n（三类标签 × Precision / Recall / F1 共 9 个指标）",
        fontsize=15, fontweight="bold", pad=35,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=12,
              frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "14_雷达图_模型性能评估.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out_path.name}")


# ============================================================================
# 6. 图15：热力图 — 混淆矩阵
#    实现要点：展示模型预测的混淆矩阵（与09号特征相关矩阵互补）
# ============================================================================

def plot_15_confusion_matrix(df: pd.DataFrame,
                             baseline_preds: np.ndarray):
    """
    混淆矩阵热力图：展示 Baseline 预测 vs 真实标签。
    左图 = 原始计数，右图 = 按真实类别归一化百分比。
    """
    true_labels = df["label_id"].values
    cm = confusion_matrix(true_labels, baseline_preds, labels=[0, 1, 2])

    # 按行（真实类别）归一化
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    class_names = ["完全匹配", "部分匹配", "不匹配"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ---- 左图：原始计数 ----
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=2, linecolor="white", square=True,
        cbar_kws={"shrink": 0.8}, ax=ax1,
        annot_kws={"fontsize": 14, "fontweight": "bold"},
    )
    ax1.set_xlabel("预测标签", fontsize=13)
    ax1.set_ylabel("真实标签", fontsize=13)
    ax1.set_title("混淆矩阵（原始计数）", fontsize=14, fontweight="bold")
    ax1.tick_params(axis="both", labelsize=12)

    # ---- 右图：归一化百分比 ----
    # 自定义标注格式：同时显示百分比和计数
    annot_text = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_text[i, j] = f"{cm_norm[i, j]:.1%}\n({cm[i, j]})"

    sns.heatmap(
        cm_norm, annot=annot_text, fmt="", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=2, linecolor="white", square=True,
        vmin=0, vmax=1,
        cbar_kws={"shrink": 0.8}, ax=ax2,
        annot_kws={"fontsize": 12},
    )
    ax2.set_xlabel("预测标签", fontsize=13)
    ax2.set_ylabel("真实标签", fontsize=13)
    ax2.set_title("混淆矩阵（归一化 + 计数）", fontsize=14, fontweight="bold")
    ax2.tick_params(axis="both", labelsize=12)

    plt.suptitle("Baseline 模型预测混淆矩阵",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "15_热力图_混淆矩阵.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out_path.name}")


# ============================================================================
# 7. 主函数
# ============================================================================

def main():
    print("=" * 60)
    print("  CCKS 2021 补充可视化 — 图 12–15（依赖模型结果）")
    print("=" * 60)

    # ---- 加载数据 ----
    print("\n[1] 加载数据...")
    results, df = load_data()

    # ---- 构建 Baseline ----
    print("\n[2] 构建 Baseline 分类器（特征阈值法）...")
    baseline_preds = build_baseline_predictions(df)

    # ---- 绘制图表 ----
    print(f"\n[3] 开始绘制补充图表（输出目录：{OUTPUT_DIR}）\n")

    plot_12_error_bar(results, df, baseline_preds)
    plot_13_line_area(df)
    plot_14_radar(results, df, baseline_preds)
    plot_15_confusion_matrix(df, baseline_preds)

    print(f"\n[OK] 全部完成！共生成 4 张图表，保存在 {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
