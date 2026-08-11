"""
importance_stats.py

Shared statistical-rigor helpers for generate_feature_importance.py and
generate_material_importance.py.

Both scripts rank predictors (turbine characteristics / material masses) by random-forest
permutation importance. Two gaps in that approach, addressed here:

1. Permutation importance is well documented to misbehave under correlated predictors: shuffling
   one of several correlated columns barely hurts the model, because it can still read the same
   signal off the correlated partner, so importance gets split/hidden rather than attributed
   (Strobl, Boulesteix, Kneib, Augustin, Zeileis, "Conditional variable importance for random
   forests", BMC Bioinformatics 2008; scikit-learn's own worked example,
   "Permutation Importance with Multicollinear or Correlated Features", uses exactly the fix
   below). Both feature sets here have real, substantial collinearity (turbine
   rated-power/diameter/hub-height/age correlate at Spearman |r| up to 0.93; several material
   masses correlate at |r| > 0.95, some exactly 1.0, since they're all near-deterministic
   functions of the same underlying P/d/h). `compute_clusters()` + `grouped_permutation_importance()`
   implement the documented mitigation: cluster near-collinear columns first (hierarchical
   clustering on 1-|Spearman r|), then shuffle every column in a cluster together (preserving
   their internal relationship, only breaking their relationship to the target) so the cluster
   gets credited as a whole instead of the credit landing arbitrarily on one member.

2. `permutation_importance()`'s raw per-repeat values (`.importances`, shape
   n_features x n_repeats) are discarded by default once you read `.importances_mean` -- but
   they're already computed for free, and they support a real significance test: whether a
   feature's importance is consistently above zero across repeats, or indistinguishable from
   noise. `permutation_significance()` does a one-sided one-sample t-test per feature.
"""
import textwrap

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import ttest_1samp
from sklearn.metrics import r2_score

# Merge two columns into the same cluster if their Spearman |r| >= 1 - CLUSTER_DISTANCE_THRESHOLD
# = 0.7, a standard "high correlation" cutoff in the multicollinearity literature (roughly
# VIF > 2). Checked empirically against both feature sets before fixing this: a stricter 0.95
# cutoff produces sensible material clusters (structural supercluster / cable plastics /
# electronics trace metals) but merges NOTHING among turbine characteristics, since their
# strongest pairwise correlation tops out at 0.93 -- silently missing the exact P/d/h/age
# collinearity this mitigation exists for. 0.7 produces a clean, consistent
# {rated power, hub height, rotor diameter, turbine age} cluster across every turbine category
# for characteristics, and an equally sensible (if slightly coarser) partition for materials.
CLUSTER_DISTANCE_THRESHOLD = 0.3


def compute_clusters(df: pd.DataFrame, cols: list, threshold: float = CLUSTER_DISTANCE_THRESHOLD) -> dict:
    """Group `cols` into clusters of near-collinear columns via average-linkage hierarchical
    clustering on Spearman-correlation distance (1 - |r|). Columns with zero variance in this
    category (e.g. a material whose percentage-of-mass table is flat across this category's P
    range, giving an undefined/NaN correlation with everything) are treated as uncorrelated
    with everything (distance 1, i.e. never merged) rather than crashing the linkage step.

    Returns {cluster_label: [member_col, ...]}, cluster_label = members joined with " + ".
    """
    corr = df[cols].corr(method="spearman").fillna(0.0).values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2, 0, None)  # symmetrize, clip tiny negative float noise
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=threshold, criterion="distance")
    groups: dict = {}
    for col, lab in zip(cols, labels):
        groups.setdefault(lab, []).append(col)
    return {" + ".join(members): members for members in groups.values()}


def wrap_cluster_label(members: list, width: int = 52) -> str:
    """Display label for a cluster: member names joined with " + ", word-wrapped onto multiple
    lines. Clusters can have up to ~9 members (e.g. onshore's structural-material bundle), and a
    single unwrapped line that long runs past the figure's left margin and gets clipped by the
    canvas -- this keeps every heatmap row's label within the reserved margin regardless of
    cluster size."""
    return "\n".join(textwrap.wrap(" + ".join(members), width=width,
                                    break_long_words=False, break_on_hyphens=False))


def grouped_permutation_importance(model, X_test: pd.DataFrame, y_test, groups: dict,
                                    n_repeats: int = 15, random_state: int = 42) -> pd.Series:
    """Permutation importance computed per cluster: every column in a cluster is shuffled with
    the SAME row permutation per repeat (so their mutual relationship survives, only their
    relationship to the target breaks). Reuses the already-fitted `model` -- no retraining."""
    rng = np.random.RandomState(random_state)
    baseline = r2_score(y_test, model.predict(X_test))
    out = {}
    for label, members in groups.items():
        drops = np.empty(n_repeats)
        for i in range(n_repeats):
            X_perm = X_test.copy()
            perm_idx = rng.permutation(len(X_test))
            X_perm[members] = X_test[members].to_numpy()[perm_idx]
            drops[i] = baseline - r2_score(y_test, model.predict(X_perm))
        out[label] = drops.mean()
    return pd.Series(out).sort_values(ascending=False)


def permutation_significance(perm_result, cols: list) -> pd.DataFrame:
    """One-sided one-sample t-test (H1: mean importance > 0) on each feature's n_repeats raw
    permutation-importance values (`perm_result.importances`, sklearn's own per-repeat array,
    shape n_features x n_repeats -- already computed, just previously discarded by both scripts
    after reading `.importances_mean`). Returns one row per feature: mean, std, p_value."""
    rows = []
    for i, col in enumerate(cols):
        vals = perm_result.importances[i]
        std = vals.std(ddof=1)
        if std == 0:
            p_one_sided = 0.0 if vals.mean() > 0 else 1.0
        else:
            t_stat, p_two_sided = ttest_1samp(vals, popmean=0.0)
            p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
        rows.append({"feature": col, "mean": vals.mean(), "std": std, "p_value": p_one_sided})
    return pd.DataFrame(rows).set_index("feature")
