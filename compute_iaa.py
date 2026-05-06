#!/usr/bin/env python3
"""
compute_iaa.py — Inter-Annotator Agreement for West Wing Power Dynamics

Metrics computed
────────────────
PRIMARY  – Power Rating (ordinal −2 to +2)
           Krippendorff α (ordinal & nominal), Fleiss' κ, ICC(2,1),
           exact / adjacent percent agreement, pairwise weighted Cohen's κ

PRIMARY  – Per-Annotator Quality
           WAWA (mean pairwise weighted κ), systematic bias, outlier rate
           Dawid-Skene EM: per-annotator confusion matrices + inferred labels

SECONDARY – Power Shift (binary)
            Krippendorff α, Fleiss' κ, percent agreement

SECONDARY – Power Strategies (multi-label, 8 labels)
            Per-strategy Fleiss' κ, α, percent agreement
            Pairwise Jaccard similarity

ROBUSTNESS – No-shift subset re-run of primary rating metrics

Outputs: results/iaa/  (7 PNG figures + iaa_report.md)
"""

import json
import os
import warnings
from collections import defaultdict

import krippendorff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss_kappa

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────

ANNOTATORS = ["claire", "evan", "galileo", "nathan", "quinn", "shai"]
RATING_LABELS = ["-2", "-1", "0", "+1", "+2"]
RATING_VALUES = {"-2": -2, "-1": -1, "0": 0, "+1": 1, "+2": 2}
INT_TO_LABEL = {v: k for k, v in RATING_VALUES.items()}
STRATEGIES = [
    "Direct orders or instructions",
    "Controls information",
    "Dismisses or shuts down",
    "Interrogates or corners",
    "Appeals to authority or rank",
    "Humor or sarcasm to assert",
    "Manages or caretakes",
    "Emotional pressure or reprimand",
]
OUTPUT_DIR = "results/iaa"


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data():
    with open("annotations/all_annotations.json") as f:
        anns = json.load(f)

    by_doc = defaultdict(list)
    for a in anns:
        by_doc[a["doc_id"]].append(a)

    # Designated IAA set: all 6 annotators
    iaa_docs    = {doc: v for doc, v in by_doc.items() if len(v) == 6}
    # Opportunistic overlaps: exactly 2 annotators
    overlap_docs = {doc: v for doc, v in by_doc.items() if len(v) == 2}

    print(f"Designated IAA items  (6 annotators): {len(iaa_docs)}")
    print(f"Opportunistic overlaps (2 annotators): {len(overlap_docs)}")
    return iaa_docs, overlap_docs


# ─── Matrix Builders ──────────────────────────────────────────────────────────

def _rating_to_int(r):
    return RATING_VALUES.get(str(r)) if r is not None else None

def build_rating_matrix(docs):
    """Return DataFrame (items × annotators) of integer ratings; NaN for missing."""
    rows = {}
    for doc, anns in docs.items():
        row = {}
        for a in anns:
            v = _rating_to_int(a["power_rating"])
            if v is not None:
                row[a["annotator"]] = float(v)
        rows[doc] = row
    df = pd.DataFrame(rows).T
    return df.reindex(columns=ANNOTATORS)


def build_shift_matrix(docs):
    """Return DataFrame (items × annotators) of 0/1 shift flags."""
    rows = {}
    for doc, anns in docs.items():
        row = {a["annotator"]: 1 if a.get("power_shift") == "Yes" else 0 for a in anns}
        rows[doc] = row
    df = pd.DataFrame(rows).T
    return df.reindex(columns=ANNOTATORS)


def build_strategy_matrix(docs, strategy):
    """Return DataFrame (items × annotators) binary for one strategy."""
    rows = {}
    for doc, anns in docs.items():
        row = {
            a["annotator"]: 1 if strategy in (a.get("power_strategies") or []) else 0
            for a in anns
        }
        rows[doc] = row
    df = pd.DataFrame(rows).T
    return df.reindex(columns=ANNOTATORS)


# ─── Generic Metric Helpers ───────────────────────────────────────────────────

def kripp_alpha(matrix, level="ordinal"):
    """Krippendorff's alpha. matrix is items × annotators (NaN for missing)."""
    data = matrix.T.values.astype(float)   # annotators × items
    # Need at least 2 distinct values in the data
    unique_vals = np.unique(data[~np.isnan(data)])
    if len(unique_vals) < 2:
        return np.nan
    try:
        return krippendorff.alpha(
            reliability_data=data,
            level_of_measurement=level,
        )
    except Exception as e:
        print(f"  [krippendorff warning] {e}")
        return np.nan


def fleiss_kappa(matrix, categories):
    """Fleiss' κ. categories is ordered list of possible values."""
    cat_idx = {c: i for i, c in enumerate(categories)}
    n_cats = len(categories)
    counts = np.zeros((len(matrix), n_cats), dtype=int)
    for i, (_, row) in enumerate(matrix.iterrows()):
        for val in row.dropna():
            j = cat_idx.get(int(val) if isinstance(val, float) else val)
            if j is not None:
                counts[i, j] += 1
    try:
        return float(sm_fleiss_kappa(counts, method="fleiss"))
    except Exception:
        return np.nan


def exact_agreement(matrix):
    """Overall exact agreement rate + per-item list."""
    per_item = []
    for _, row in matrix.iterrows():
        vals = row.dropna().tolist()
        per_item.append(1.0 if len(vals) >= 2 and len(set(vals)) == 1 else 0.0)
    return float(np.mean(per_item)), per_item


def adjacent_agreement(rating_matrix, tol=1):
    """Fraction of all annotator pairs whose ratings are within tol steps."""
    diffs = []
    for _, row in rating_matrix.iterrows():
        vals = row.dropna().tolist()
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                diffs.append(abs(vals[i] - vals[j]) <= tol)
    return float(np.mean(diffs)) if diffs else np.nan


def compute_icc(rating_matrix):
    """ICC(2,1) — two-way mixed, absolute agreement (items × annotators)."""
    data = rating_matrix.dropna().values.astype(float)
    if data.shape[0] < 2:
        return np.nan
    n, k = data.shape
    grand = data.mean()
    row_m = data.mean(axis=1)
    col_m = data.mean(axis=0)
    ss_total = ((data - grand) ** 2).sum()
    ss_rows  = k * ((row_m - grand) ** 2).sum()
    ss_cols  = n * ((col_m - grand) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols
    ms_rows  = ss_rows  / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    ms_cols  = ss_cols  / (k - 1)
    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return float((ms_rows - ms_error) / denom) if denom != 0 else np.nan


def pairwise_kappa(matrix, weighted=True, labels=None):
    """6×6 DataFrame of pairwise Cohen's κ (weighted linear or unweighted)."""
    annotators = [a for a in ANNOTATORS if a in matrix.columns]
    n = len(annotators)
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 1.0)

    for i, a1 in enumerate(annotators):
        for j, a2 in enumerate(annotators):
            if i >= j:
                continue
            both = matrix[[a1, a2]].dropna()
            if len(both) < 2:
                continue
            v1 = both[a1].astype(int).tolist()
            v2 = both[a2].astype(int).tolist()
            try:
                w = "linear" if weighted else None
                k = cohen_kappa_score(v1, v2, weights=w, labels=labels)
                mat[i, j] = mat[j, i] = k
            except Exception:
                pass

    return pd.DataFrame(mat, index=annotators, columns=annotators)


# ─── Dawid-Skene EM ───────────────────────────────────────────────────────────

def dawid_skene_em(rating_matrix, n_iter=200, tol=1e-8):
    """
    Dawid-Skene EM for ordinal labels.

    Returns
    -------
    pi   : (I × K) posterior P(true_label = k | annotations)
    T    : (J × K × K) confusion matrices  T[j, k, l] = P(obs=l | true=k, ann=j)
    inferred : dict {doc_id: inferred_int_label}
    items, annotators, classes
    """
    items      = list(rating_matrix.index)
    annotators = [a for a in ANNOTATORS if a in rating_matrix.columns]
    classes    = list(range(-2, 3))            # [-2, -1, 0, 1, 2]
    K = len(classes)
    I = len(items)
    J = len(annotators)
    c_idx = {c: i for i, c in enumerate(classes)}

    # Build observation tensor  obs[i, j, k]  and mask
    obs      = np.zeros((I, J, K))
    obs_mask = np.zeros((I, J), dtype=bool)
    for i, item in enumerate(items):
        for j, ann in enumerate(annotators):
            val = rating_matrix.loc[item, ann]
            if not np.isnan(val):
                obs[i, j, c_idx[int(val)]] = 1.0
                obs_mask[i, j] = True

    # Initialise T with strong diagonal
    T = np.zeros((J, K, K))
    for j in range(J):
        for k in range(K):
            T[j, k] = 0.05 / (K - 1)
            T[j, k, k] = 0.95

    # Initialise pi from raw vote counts
    pi = np.zeros((I, K))
    for i in range(I):
        votes = obs[i][obs_mask[i]].sum(axis=0)
        pi[i] = votes / votes.sum() if votes.sum() > 0 else np.ones(K) / K

    prev_ll = -np.inf
    for it in range(n_iter):
        # ── E-step ──────────────────────────────────────────────────────────
        log_pi = np.zeros((I, K))
        p_class = pi.mean(axis=0)
        p_class /= p_class.sum()

        for i in range(I):
            for k in range(K):
                lp = np.log(p_class[k] + 1e-300)
                for j in range(J):
                    if obs_mask[i, j]:
                        lp += np.dot(obs[i, j], np.log(T[j, k] + 1e-300))
                log_pi[i, k] = lp
            # Softmax in log space
            log_pi[i] -= log_pi[i].max()
            pi[i] = np.exp(log_pi[i])
            pi[i] /= pi[i].sum()

        # ── M-step ──────────────────────────────────────────────────────────
        p_class = pi.mean(axis=0)
        p_class /= p_class.sum()

        for j in range(J):
            for k in range(K):
                # numerator[l] = sum_i pi[i,k] * obs[i,j,l]
                num = (pi[:, k, np.newaxis] * obs[:, j, :]).sum(axis=0)
                denom = num.sum()
                T[j, k] = num / denom if denom > 0 else np.ones(K) / K

        # ── Convergence ──────────────────────────────────────────────────────
        ll = 0.0
        for i in range(I):
            item_ll = 0.0
            for k in range(K):
                p = p_class[k]
                for j in range(J):
                    if obs_mask[i, j]:
                        p *= np.dot(obs[i, j], T[j, k]).clip(1e-300, None)
                item_ll += p
            ll += np.log(max(item_ll, 1e-300))
        if abs(ll - prev_ll) < tol:
            print(f"  Dawid-Skene converged at iteration {it + 1}")
            break
        prev_ll = ll

    inferred = {items[i]: classes[pi[i].argmax()] for i in range(I)}
    return pi, T, inferred, items, annotators, classes


# ─── Primary: Power Rating ────────────────────────────────────────────────────

def compute_rating_metrics(iaa_docs, overlap_docs):
    mat = build_rating_matrix(iaa_docs)
    print("\n── Power Rating Matrix (IAA) ──")
    print(mat.to_string())

    alpha_ord  = kripp_alpha(mat, "ordinal")
    alpha_nom  = kripp_alpha(mat, "nominal")
    fleiss_k   = fleiss_kappa(mat, list(range(-2, 3)))
    exact_pct, per_item_exact = exact_agreement(mat)
    adj_pct    = adjacent_agreement(mat)
    icc        = compute_icc(mat)

    pw          = pairwise_kappa(mat, weighted=True,  labels=list(range(-2, 3)))
    pw_unw      = pairwise_kappa(mat, weighted=False, labels=list(range(-2, 3)))
    upper_tri   = pw.values[np.triu_indices(len(pw), k=1)]
    pw_mean     = float(np.nanmean(upper_tri))
    pw_std      = float(np.nanstd(upper_tri))

    print(f"\nPrimary Power Rating Metrics  (n={len(iaa_docs)} items, 6 annotators)")
    print(f"  Krippendorff α (ordinal) : {alpha_ord:.4f}  ← primary metric")
    print(f"  Fleiss' κ                : {fleiss_k:.4f}")
    print(f"  Exact agreement          : {exact_pct:.2%}")
    print(f"  Adjacent agreement (±1)  : {adj_pct:.2%}")
    print(f"  ICC(2,1)                 : {icc:.4f}")
    print(f"  Mean pairwise wt. κ      : {pw_mean:.4f} ± {pw_std:.4f}")
    print(f"  Krippendorff α (nominal) : {alpha_nom:.4f}  (robustness baseline)")

    # ── No-shift subset ──────────────────────────────────────────────────────
    shift_mat = build_shift_matrix(iaa_docs)
    no_shift_mask = shift_mat.sum(axis=1) <= (shift_mat.shape[1] / 2)
    no_shift_items = no_shift_mask[no_shift_mask].index.tolist()

    ns_results = {}
    if len(no_shift_items) >= 2:
        mat_ns = mat.loc[no_shift_items]
        ns_results = {
            "n":        len(no_shift_items),
            "alpha":    kripp_alpha(mat_ns, "ordinal"),
            "fleiss":   fleiss_kappa(mat_ns, list(range(-2, 3))),
            "exact":    exact_agreement(mat_ns)[0],
            "adjacent": adjacent_agreement(mat_ns),
        }
        print(f"\n  No-shift subset ({ns_results['n']} items, majority said no shift):")
        print(f"    α (ordinal): {ns_results['alpha']:.4f}  "
              f"(Δ = {ns_results['alpha'] - alpha_ord:+.4f})")

    # ── Combined (IAA + opportunistic) α for reference ───────────────────────
    combined = {**iaa_docs, **overlap_docs}
    combined_mat = build_rating_matrix(combined)
    combined_alpha = kripp_alpha(combined_mat, "ordinal")
    print(f"\n  Combined α incl. opportunistic overlaps "
          f"(n={len(combined)} items): {combined_alpha:.4f}")

    return {
        "matrix":          mat,
        "alpha_ordinal":   alpha_ord,
        "alpha_nominal":   alpha_nom,
        "fleiss_kappa":    fleiss_k,
        "exact_pct":       exact_pct,
        "adjacent_pct":    adj_pct,
        "icc":             icc,
        "pw_kappa":        pw,
        "pw_kappa_unw":    pw_unw,
        "pw_mean":         pw_mean,
        "pw_std":          pw_std,
        "per_item_exact":  per_item_exact,
        "no_shift":        ns_results,
        "combined_alpha":  combined_alpha,
        "combined_n":      len(combined),
    }


# ─── Primary: Per-Annotator Quality ──────────────────────────────────────────

def compute_annotator_quality(rating_matrix):
    """WAWA-style quality metrics for each annotator."""
    pw = pairwise_kappa(rating_matrix, weighted=True, labels=list(range(-2, 3)))

    all_vals_by_item = rating_matrix.mean(axis=1)   # item-level mean as consensus proxy
    group_mean = float(rating_matrix.stack().mean())

    results = {}
    for ann in ANNOTATORS:
        if ann not in rating_matrix.columns:
            continue
        others = [a for a in ANNOTATORS if a != ann and a in pw.columns]
        kappa_vals = [pw.loc[ann, o] for o in others if not np.isnan(pw.loc[ann, o])]
        wawa = float(np.mean(kappa_vals)) if kappa_vals else np.nan

        ann_vals = rating_matrix[ann].dropna()
        bias = float(ann_vals.mean() - group_mean)

        item_medians = rating_matrix.median(axis=1)
        outlier_rate = float((np.abs(ann_vals - item_medians[ann_vals.index]) >= 2).mean())

        results[ann] = {"wawa": wawa, "bias": bias, "outlier_rate": outlier_rate}

    print("\n── Per-Annotator Quality (WAWA) ──")
    for ann, m in sorted(results.items(), key=lambda x: -x[1]["wawa"]):
        print(f"  {ann:10s}: WAWA={m['wawa']:.3f}  bias={m['bias']:+.3f}  "
              f"outlier_rate={m['outlier_rate']:.2%}")
    return results


def run_dawid_skene(rating_matrix):
    """Dawid-Skene EM and derived quality metrics."""
    print("\n── Dawid-Skene EM ──")
    pi, T, inferred, items, annotators, classes = dawid_skene_em(rating_matrix)

    # Majority vote for comparison
    majority = {}
    for item in items:
        vals = rating_matrix.loc[item].dropna().astype(int).tolist()
        if vals:
            majority[item] = max(set(vals), key=vals.count)

    agree_rate = float(np.mean([inferred[it] == majority[it]
                                for it in items if it in majority]))
    print(f"  DS-EM vs. majority vote agreement: {agree_rate:.2%}")

    ds_quality = {}
    for j, ann in enumerate(annotators):
        cm = T[j]                                # K × K
        accuracy = float(np.trace(cm) / len(classes))
        # DS bias: E[observed] − E[true] under inferred label distribution
        class_arr = np.array(classes, dtype=float)
        p_true = pi.mean(axis=0)
        exp_obs_per_true = (T[j] * class_arr[np.newaxis, :]).sum(axis=1)
        ds_bias = float((p_true * (exp_obs_per_true - class_arr)).sum())
        ds_quality[ann] = {
            "ds_accuracy": accuracy,
            "ds_bias":     ds_bias,
            "cm":          cm,
        }
        print(f"  {ann:10s}: DS acc={accuracy:.3f}  DS bias={ds_bias:+.3f}")

    return {
        "pi":               pi,
        "T":                T,
        "inferred":         inferred,
        "majority":         majority,
        "ds_majority_agree": agree_rate,
        "ds_quality":       ds_quality,
        "items":            items,
        "annotators":       annotators,
        "classes":          classes,
    }


# ─── Secondary: Power Shift ───────────────────────────────────────────────────

def compute_shift_metrics(iaa_docs):
    mat = build_shift_matrix(iaa_docs)

    alpha     = kripp_alpha(mat, "nominal")
    fleiss_k  = fleiss_kappa(mat, [0, 1])
    exact_pct, _ = exact_agreement(mat)
    prevalence = float(mat.stack().mean())

    print(f"\n── Power Shift (IAA n={len(iaa_docs)}) ──")
    print(f"  Krippendorff α (nominal) : {alpha:.4f}")
    print(f"  Fleiss' κ                : {fleiss_k:.4f}")
    print(f"  Exact agreement          : {exact_pct:.2%}")
    print(f"  Prevalence (% Yes)       : {prevalence:.2%}")

    return {
        "matrix":     mat,
        "alpha":      alpha,
        "fleiss":     fleiss_k,
        "exact":      exact_pct,
        "prevalence": prevalence,
    }


# ─── Secondary: Power Strategies ─────────────────────────────────────────────

def compute_strategy_metrics(iaa_docs):
    per_strat = {}
    for strat in STRATEGIES:
        mat      = build_strategy_matrix(iaa_docs, strat)
        exact, _ = exact_agreement(mat)
        prev     = float(mat.stack().mean())
        # Skip agreement metrics if all values are identical (no variation to measure)
        if mat.stack().nunique() < 2:
            alpha = fleiss_k = np.nan
        else:
            alpha    = kripp_alpha(mat, "nominal")
            fleiss_k = fleiss_kappa(mat, [0, 1])
        per_strat[strat] = {
            "alpha": alpha, "fleiss": fleiss_k,
            "exact": exact, "prevalence": prev,
        }

    # Pairwise Jaccard
    n_ann = len(ANNOTATORS)
    jac_sum   = np.zeros((n_ann, n_ann))
    jac_count = np.zeros((n_ann, n_ann))
    ann_idx   = {a: i for i, a in enumerate(ANNOTATORS)}

    for doc, anns in iaa_docs.items():
        ann_strats = {}
        for a in anns:
            if a["annotator"] in ann_idx:
                ann_strats[a["annotator"]] = set(a.get("power_strategies") or [])
        for a1 in ann_strats:
            for a2 in ann_strats:
                if a1 == a2:
                    continue
                i, j = ann_idx[a1], ann_idx[a2]
                s1, s2 = ann_strats[a1], ann_strats[a2]
                union = s1 | s2
                jac = 1.0 if len(union) == 0 else len(s1 & s2) / len(union)
                jac_sum[i, j]   += jac
                jac_count[i, j] += 1

    np.fill_diagonal(jac_count, 1)
    jac_mean = jac_sum / np.where(jac_count == 0, 1, jac_count)
    np.fill_diagonal(jac_mean, 1.0)
    jaccard_df = pd.DataFrame(jac_mean, index=ANNOTATORS, columns=ANNOTATORS)

    upper = jac_mean[np.triu_indices(n_ann, k=1)]
    mean_jac = float(np.nanmean(upper))

    print(f"\n── Power Strategies ──")
    for s, m in per_strat.items():
        print(f"  {s[:38]:38s}: Fleiss'κ={m['fleiss']:.3f}  α={m['alpha']:.3f}  "
              f"agree={m['exact']:.2%}  prev={m['prevalence']:.2%}")
    print(f"\n  Mean pairwise Jaccard (strategy sets): {mean_jac:.4f}")

    return {"per_strat": per_strat, "jaccard": jaccard_df, "mean_jaccard": mean_jac}


# ─── Figures ──────────────────────────────────────────────────────────────────

def _short_doc(doc):
    return doc.replace("pair", "p").replace("exc", "e")


def fig_rating_per_item_heatmap(rating_matrix):
    fig, ax = plt.subplots(figsize=(10, 6))
    mat = rating_matrix.copy()
    mat.index = [_short_doc(d) for d in mat.index]

    sns.heatmap(
        mat.astype(float),
        cmap=sns.diverging_palette(10, 240, n=5, as_cmap=True),
        vmin=-2, vmax=2,
        annot=True, fmt=".0f", linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Power Rating  (−2=A dominates, +2=B dominates)"},
    )
    ax.set_title("Power Rating per Item per Annotator  (IAA Set)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Annotator")
    ax.set_ylabel("Document")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rating_per_item_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def fig_pairwise_kappa_heatmap(pw_kappa):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        pw_kappa,
        annot=True, fmt=".3f",
        cmap="RdYlGn", vmin=-0.2, vmax=1.0,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Weighted Cohen's κ"},
    )
    ax.set_title("Pairwise Weighted Cohen's κ\n(Power Rating, IAA Set)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rating_pairwise_kappa_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def fig_annotator_quality(wawa_results, ds_results):
    annotators = sorted(wawa_results, key=lambda a: -wawa_results[a]["wawa"])
    ds_q = ds_results["ds_quality"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    # Panel 1: WAWA
    wawa_vals = [wawa_results[a]["wawa"] for a in annotators]
    palette = sns.color_palette("Blues_d", len(annotators))
    bars = axes[0].bar(annotators, wawa_vals, color=palette)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("WAWA\n(Mean Pairwise Weighted κ)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Mean weighted Cohen's κ")
    # Add padding to y-axis limits
    wawa_max = max(wawa_vals)
    axes[0].set_ylim(min(0, min(wawa_vals) - 0.05), wawa_max + 0.08)
    for b, v in zip(bars, wawa_vals):
        axes[0].text(b.get_x() + b.get_width() / 2,
                     b.get_height() + 0.01, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=9)

    # Panel 2: Bias
    bias_vals = [wawa_results[a]["bias"] for a in annotators]
    colors = ["#d73027" if v < 0 else "#4575b4" for v in bias_vals]
    bars2 = axes[1].bar(annotators, bias_vals, color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("Systematic Bias\n(Annotator Mean − Group Mean)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Rating offset  (−=toward A,  +=toward B)")
    # Add padding to y-axis limits
    bias_max = max(bias_vals)
    bias_min = min(bias_vals)
    axes[1].set_ylim(bias_min - 0.08, bias_max + 0.08)
    for b, v in zip(bars2, bias_vals):
        offset = 0.01 if v >= 0 else -0.04
        axes[1].text(b.get_x() + b.get_width() / 2,
                     v + offset, f"{v:+.3f}",
                     ha="center", va="bottom", fontsize=9)

    # Panel 3: DS accuracy
    ds_acc = [ds_q[a]["ds_accuracy"] for a in annotators]
    bars3 = axes[2].bar(annotators, ds_acc,
                        color=sns.color_palette("Greens_d", len(annotators)))
    axes[2].set_ylim(0, 1.08)
    axes[2].set_title("Dawid-Skene Accuracy\n(Diagonal mass of confusion matrix)",
                      fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Accuracy")
    for b, v in zip(bars3, ds_acc):
        axes[2].text(b.get_x() + b.get_width() / 2,
                     b.get_height() + 0.01, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=9)

    fig.suptitle("Per-Annotator Quality Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "annotator_quality.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def fig_rating_distribution(rating_matrix):
    fig, ax = plt.subplots(figsize=(9, 5))
    counts = {}
    for ann in ANNOTATORS:
        if ann not in rating_matrix.columns:
            continue
        vals = rating_matrix[ann].dropna().astype(int).tolist()
        counts[ann] = {v: vals.count(v) for v in range(-2, 3)}
    df = pd.DataFrame(counts).T.rename(columns=INT_TO_LABEL)

    colors = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]
    df[["-2", "-1", "0", "+1", "+2"]].plot(
        kind="bar", stacked=True, ax=ax,
        color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Power Rating Distribution per Annotator  (IAA Items)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Annotator")
    ax.set_ylabel("Count")
    ax.legend(title="Rating", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rating_distribution_by_annotator.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def fig_strategy_kappa_bar(strategy_results):
    strats = list(strategy_results["per_strat"].keys())
    fleiss_vals = [strategy_results["per_strat"][s]["fleiss"] for s in strats]
    alpha_vals  = [strategy_results["per_strat"][s]["alpha"]  for s in strats]

    idx = np.argsort(fleiss_vals)[::-1]
    strats_s  = [strats[i] for i in idx]
    fleiss_s  = [fleiss_vals[i] for i in idx]
    alpha_s   = [alpha_vals[i]  for i in idx]
    short = [
        s.replace(" or instructions", "").replace(" or sarcasm", "")
         .replace(" or reprimand", "").replace(" or rank", "")
         .replace(" or corners", "").strip()
        for s in strats_s
    ]

    x     = np.arange(len(strats_s))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - width / 2, fleiss_s, width, label="Fleiss' κ", color="#5ab4ac")
    b2 = ax.bar(x + width / 2, alpha_s,  width, label="Krippendorff α", color="#d8b365")
    ax.axhline(0,   color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.4, color="gray",  linewidth=0.5, linestyle=":",
               label="Moderate threshold (0.4)")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_ylabel("Agreement metric value")
    ax.set_title("Agreement per Power Strategy  (IAA Set)", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "strategy_fleiss_kappa_bar.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def fig_strategy_jaccard_heatmap(jaccard_df):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        jaccard_df,
        annot=True, fmt=".3f",
        cmap="YlGnBu", vmin=0, vmax=1,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Mean Jaccard Similarity"},
    )
    ax.set_title("Pairwise Jaccard Similarity\n(Power Strategies, IAA Set)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "strategy_pairwise_jaccard_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def fig_ds_confusion_matrices(ds_results):
    annotators = ds_results["annotators"]
    T          = ds_results["T"]
    classes    = ds_results["classes"]
    class_lbls = [str(c) for c in classes]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for j, ann in enumerate(annotators):
        cm  = T[j]
        ax  = axes[j]
        im  = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        acc = float(np.trace(cm) / len(classes))
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(class_lbls)
        ax.set_yticklabels(class_lbls)
        ax.set_xlabel("Observed label")
        ax.set_ylabel("True label")
        ax.set_title(f"{ann}  (acc = {acc:.3f})", fontsize=11, fontweight="bold")
        for row in range(len(classes)):
            for col in range(len(classes)):
                ax.text(col, row, f"{cm[row, col]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if cm[row, col] > 0.55 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Dawid-Skene EM — Per-Annotator Confusion Matrices\n"
        "(rows = inferred true label, cols = observed label)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ds_em_confusion_matrices.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# ─── Report ───────────────────────────────────────────────────────────────────

def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if not np.isnan(v) else "N/A"
    return str(v)


def _table(headers, rows):
    widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep  = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |"
    lines = [head, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |")
    return "\n".join(lines) + "\n"


def generate_report(rating_r, wawa_r, ds_r, shift_r, strat_r, iaa_docs, overlap_docs):
    ns = rating_r.get("no_shift", {})
    pw = rating_r["pw_kappa"]
    ds_q = ds_r["ds_quality"]

    lines = [
        "# West Wing IAA Report\n",
        "*Generated by `compute_iaa.py`*\n",

        "## 1. Dataset Summary\n",
        f"- **Designated IAA items** (all 6 annotators): {len(iaa_docs)}",
        f"- **Opportunistic overlap items** (2 annotators each): {len(overlap_docs)}",
        f"- **Annotators**: {', '.join(ANNOTATORS)}",
        f"- **Combined α** (IAA + opportunistic, ordinal): "
        f"{rating_r['combined_alpha']:.4f}  (n={rating_r['combined_n']} items)\n",

        "## 2. Power Rating Agreement  *(PRIMARY)*\n",
        "*Ordinal scale −2 to +2. Core annotation task.*\n",
        _table(
            ["Metric", "Value", "Notes"],
            [
                ["**Krippendorff α (ordinal)**",
                 _fmt(rating_r["alpha_ordinal"]),
                 "**Primary metric** — ordinal distance weighting"],
                ["Fleiss' κ",
                 _fmt(rating_r["fleiss_kappa"]),
                 "Nominal; widely reported in NLP"],
                ["Exact agreement",
                 f"{rating_r['exact_pct']:.2%}",
                 "All 6 annotators give identical rating"],
                ["Adjacent agreement (±1)",
                 f"{rating_r['adjacent_pct']:.2%}",
                 "All pairs within one rating step"],
                ["ICC(2,1)",
                 _fmt(rating_r["icc"]),
                 "Two-way mixed, absolute agreement"],
                ["Mean pairwise wt. κ",
                 f"{rating_r['pw_mean']:.4f} ± {rating_r['pw_std']:.4f}",
                 "Average over 15 annotator pairs"],
                ["Krippendorff α (nominal)",
                 _fmt(rating_r["alpha_nominal"]),
                 "Robustness baseline (ignores ordinal structure)"],
            ]
        ),
    ]

    if ns:
        lines += [
            f"**No-shift subset** ({ns['n']} items where majority did not flag a power shift)\n",
            _table(
                ["Metric", "Full set", "No-shift subset", "Δ"],
                [
                    ["α (ordinal)",
                     _fmt(rating_r["alpha_ordinal"]),
                     _fmt(ns["alpha"]),
                     f"{ns['alpha'] - rating_r['alpha_ordinal']:+.4f}"],
                    ["Fleiss' κ",
                     _fmt(rating_r["fleiss_kappa"]),
                     _fmt(ns["fleiss"]),
                     f"{ns['fleiss'] - rating_r['fleiss_kappa']:+.4f}"],
                    ["Exact agreement",
                     f"{rating_r['exact_pct']:.2%}",
                     f"{ns['exact']:.2%}",
                     f"{ns['exact'] - rating_r['exact_pct']:+.2%}"],
                    ["Adjacent agreement",
                     f"{rating_r['adjacent_pct']:.2%}",
                     f"{ns['adjacent']:.2%}",
                     f"{ns['adjacent'] - rating_r['adjacent_pct']:+.2%}"],
                ]
            ),
        ]

    lines += [
        "## 3. Per-Annotator Quality  *(PRIMARY)*\n",
        "**WAWA** = Worker Agreement With Aggregate (mean pairwise weighted κ against all others).  ",
        "**Bias** = annotator mean − group mean (positive → rates toward B dominance).  ",
        "**DS Accuracy** = diagonal mass of Dawid-Skene confusion matrix.\n",
        _table(
            ["Annotator", "WAWA κ", "Bias", "Outlier Rate", "DS Accuracy", "DS Bias"],
            [
                [ann,
                 _fmt(wawa_r[ann]["wawa"]),
                 f"{wawa_r[ann]['bias']:+.3f}",
                 f"{wawa_r[ann]['outlier_rate']:.2%}",
                 _fmt(ds_q[ann]["ds_accuracy"]),
                 f"{ds_q[ann]['ds_bias']:+.3f}"]
                for ann in sorted(wawa_r, key=lambda a: -wawa_r[a]["wawa"])
            ]
        ),

        "### Dawid-Skene Inferred Labels vs. Majority Vote\n",
        f"Agreement: **{ds_r['ds_majority_agree']:.2%}**\n",
        _table(
            ["Document", "DS Inferred", "Majority Vote", "Match?"],
            [
                [_short_doc(doc),
                 str(ds_r["inferred"].get(doc, "?")),
                 str(ds_r["majority"].get(doc, "?")),
                 "✓" if ds_r["inferred"].get(doc) == ds_r["majority"].get(doc) else "✗"]
                for doc in sorted(ds_r["inferred"])
            ]
        ),

        "## 4. Power Shift Agreement  *(SECONDARY)*\n",
        f"*Binary: Yes = power shifts within excerpt. Base rate: {shift_r['prevalence']:.2%}*\n",
        _table(
            ["Metric", "Value"],
            [
                ["Krippendorff α (nominal)", _fmt(shift_r["alpha"])],
                ["Fleiss' κ",               _fmt(shift_r["fleiss"])],
                ["Exact agreement",          f"{shift_r['exact']:.2%}"],
            ]
        ),

        "## 5. Power Strategy Agreement  *(SECONDARY)*\n",
        f"*Multi-label; 8 strategies. Mean pairwise Jaccard: **{strat_r['mean_jaccard']:.4f}***\n",
        _table(
            ["Strategy", "Fleiss' κ", "α (nominal)", "Exact %", "Prevalence"],
            [
                [s,
                 _fmt(m["fleiss"]),
                 _fmt(m["alpha"]),
                 f"{m['exact']:.2%}",
                 f"{m['prevalence']:.2%}"]
                for s, m in sorted(strat_r["per_strat"].items(),
                                   key=lambda x: -x[1]["fleiss"])
            ]
        ),

        "## 6. Pairwise Weighted Cohen's κ Matrix  *(ROBUSTNESS)*\n",
        _table(
            [""] + list(pw.columns),
            [
                [ann] + [f"{pw.loc[ann, c]:.3f}"
                         if not np.isnan(pw.loc[ann, c]) else "—"
                         for c in pw.columns]
                for ann in pw.index
            ]
        ),

        "## 7. Per-Item Breakdown\n",
        _table(
            ["Document"] + ANNOTATORS + ["All Agree?"],
            [
                [_short_doc(doc)] +
                [str(int(rating_r["matrix"].loc[doc, a]))
                 if not np.isnan(rating_r["matrix"].loc[doc, a]) else "—"
                 for a in ANNOTATORS] +
                ["✓" if rating_r["per_item_exact"][i] == 1.0 else "✗"]
                for i, doc in enumerate(rating_r["matrix"].index)
            ]
        ),

        "## 8. Interpretation Guide\n",
        "Standard κ / α benchmarks (Landis & Koch 1977):\n",
        _table(
            ["Range", "Interpretation"],
            [
                ["< 0.00",       "Poor / less than chance"],
                ["0.00 – 0.20",  "Slight"],
                ["0.21 – 0.40",  "Fair"],
                ["0.41 – 0.60",  "Moderate"],
                ["0.61 – 0.80",  "Substantial"],
                ["0.81 – 1.00",  "Almost perfect"],
            ]
        ),
    ]

    path = os.path.join(OUTPUT_DIR, "iaa_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    iaa_docs, overlap_docs = load_data()

    rating_r  = compute_rating_metrics(iaa_docs, overlap_docs)
    wawa_r    = compute_annotator_quality(rating_r["matrix"])
    ds_r      = run_dawid_skene(rating_r["matrix"])
    shift_r   = compute_shift_metrics(iaa_docs)
    strat_r   = compute_strategy_metrics(iaa_docs)

    print("\nGenerating figures...")
    fig_rating_per_item_heatmap(rating_r["matrix"])
    fig_pairwise_kappa_heatmap(rating_r["pw_kappa"])
    fig_annotator_quality(wawa_r, ds_r)
    fig_rating_distribution(rating_r["matrix"])
    fig_strategy_kappa_bar(strat_r)
    fig_strategy_jaccard_heatmap(strat_r["jaccard"])
    fig_ds_confusion_matrices(ds_r)

    generate_report(rating_r, wawa_r, ds_r, shift_r, strat_r, iaa_docs, overlap_docs)

    print(f"\nAll outputs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
