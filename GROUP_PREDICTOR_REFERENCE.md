# CUBE Group Predictor — Complete Reference

The **Group Predictor** tab answers a single question: *given this animal's behavioural profile, which experimental group does it belong to?* Rather than testing whether groups differ on individual clusters one at a time, the tool trains a multivariate supervised classifier and evaluates how well the full behavioural fingerprint separates the groups — using Leave-One-Out cross-validation and a permutation test so the result is both predictive and statistically grounded.

---

## Contents

1. [Rationale](#1-rationale)
2. [Feature Matrix Construction](#2-feature-matrix-construction)
3. [Classification Algorithms](#3-classification-algorithms)
4. [The Full Computational Pipeline](#4-the-full-computational-pipeline)
5. [Leave-One-Out Cross-Validation](#5-leave-one-out-cross-validation)
6. [Balanced Accuracy and Cohen's κ](#6-balanced-accuracy-and-cohens-κ)
7. [Permutation Test](#7-permutation-test)
8. [Cluster Selection — Exhaustive and Greedy](#8-cluster-selection--exhaustive-and-greedy)
9. [Shapley Feature Importances](#9-shapley-feature-importances)
10. [Controls Reference](#10-controls-reference)
11. [Reading the Results](#11-reading-the-results)
12. [Interpreting the Null Distribution Figure](#12-interpreting-the-null-distribution-figure)
13. [Interpreting the Per-Animal Probability Figure](#13-interpreting-the-per-animal-probability-figure)
14. [Export CSV](#14-export-csv)
15. [Caveats and Limitations](#15-caveats-and-limitations)
16. [Tips and User Guidance](#16-tips-and-user-guidance)

---

## 1. Rationale

Standard group comparisons (Kruskal-Wallis, post-hoc FDR) test one cluster at a time. That is powerful for identifying *which* behaviours differ but misses *combinations* of modest differences that together discriminate the groups. A multivariate classifier captures exactly this joint pattern.

Three parallel models always run, each measuring a different behavioural dimension:

| Model | What it measures |
|---|---|
| **Frequency** | How often each cluster occurs per session (bouts/second) |
| **Total Duration** | Cumulative time spent in each cluster (fraction of session) |
| **Transition Probability** | Which clusters tend to follow which — the sequential grammar of behaviour |

Running all three simultaneously reveals *which dimension* the experimental manipulation acts on. A drug may leave bout frequency unchanged while dramatically shifting behavioural sequencing, or vice versa.

**When to use it:** after assigning experimental groups to animals (e.g., Drug vs. Control, Genotype A vs. B). Requires ≥2 groups and ≥2 animals per group.

---

## 2. Feature Matrix Construction

### 2.1 Frequency and Duration Features

For each animal, one scalar is computed per cluster:
- **Frequency** — total bout count divided by session length (bouts/second).
- **Total Duration** — sum of all bout lengths divided by session length (fraction of time).

The result is an `(n_animals × n_clusters)` matrix. Clusters not observed in a given animal receive zero, not NaN.

### 2.2 Transition Probability Features

Consecutive bouts define transitions. For every ordered pair (A → B), raw counts are divided by total transitions *out of* A (row normalisation). Only off-diagonal transitions are used — the self-persistence diagonal (A → A) is excluded because it conflates bout duration with sequencing behaviour.

When a cluster subset is specified, the full transition matrix is built first, then the sub-matrix is extracted and row-renormalised *within the subset*. This preserves relative switching probabilities between retained clusters.

> The transition model always uses raw cluster labels regardless of the Feature Source setting, because transition probabilities are inherently defined at the cluster level, not the group level.

### 2.3 Feature Source Options

| Option | Frequency / Duration features | Transition features |
|--------|-------------------------------|---------------------|
| **Individual Clusters** | One column per cluster (all clusters) | All clusters (always) |
| **Behavior Groups** | One column per user-selected group | All clusters (always) |
| **Mix (Both)** | All clusters + selected groups (deduplicated) | All clusters (always) |
| **Custom** | Only checked clusters + checked groups | All clusters (always) |

- **Individual Clusters** — default; tests whether raw cluster usage predicts group.
- **Behavior Groups** — useful when you have annotated groups ("Locomotion", "Grooming") and want to reduce dimensionality to meaningful categories.
- **Mix** — combines both; best when groups and clusters are not redundant.
- **Custom** — tests specific hypotheses: "do clusters 3 and 7 together predict treatment?" Especially useful when you suspect synergistic effects of particular cluster combinations.

---

## 3. Classification Algorithms

### 3.1 Elastic Net Logistic Regression (Default)

Learns a linear decision boundary with two simultaneous penalty terms:

- **Lasso (L1)** drives small coefficients exactly to zero — automatic feature selection. Clusters that do not help separate groups are eliminated.
- **Ridge (L2)** handles correlated features. B-SOiD clusters often co-occur; without Ridge, Lasso arbitrarily picks one correlated cluster and discards the others, making importances unstable.

For ≥3 groups, multinomial softmax is used — more principled than one-vs-rest when group sizes are unequal.

**Hyperparameter grid:**
- **C-grid:** `[0.001, 0.01, 0.1, 1, 10]` for n < 25 animals; wider grid for n ≥ 25.
- **L1-ratio grid:** `[0.3, 0.5, 0.7]` for n < 25; wider for n ≥ 25.
- Inner CV folds: `min(3, min_group_size)` — scales down gracefully for small cohorts.
- `class_weight="balanced"` — compensates for unequal group sizes.

### 3.2 SVM with Linear Kernel (Alternative)

Finds the maximum-margin hyperplane — the widest gap between nearest training examples from different groups. Raw SVM decision scores are calibrated to probabilities via Platt scaling, enabling ROC curves and per-animal probability strips.

**Prefer SVM when:** you expect linear separability and want the margin guarantee, or when Elastic Net converges slowly.

### Why Random Forest Was Excluded

With n = 5–6 animals per group, 100 trees fitted on 10 animals is overparameterised by ~10×. LOO accuracy becomes artificially high because the forest memorises the training animals. Random Forest becomes appropriate at n > 30 per group.

---

## 4. The Full Computational Pipeline

Every run executes this sequence on a background thread, keeping the GUI responsive.

### 4.1 Preprocessing

Every model passes through a fixed sequence before classification:

```
Raw features
    │
    ▼
[1] VarianceThreshold
    │  Remove features with near-zero variance (< 1×10⁻¹⁰)
    │  Eliminates clusters with identical counts across all animals
    ▼
[2] RobustScaler
    │  Centre and scale using median and IQR (robust to outlier animals)
    ▼
[3] PolynomialFeatures  (frequency/duration models only)
    │  Add pairwise interaction terms: e.g. freq(C4) × freq(C5)
    │  Captures "high C4 AND high C5 together" — cluster combinations
    │  Skipped for Transition model (already captures co-occurrence)
    ▼
[4] Adaptive PCA  (only when n_features > n_animals)
    │  Compress to min(n_animals − 2, 15) principal components
    │  Feature importances are back-projected to original feature space
    ▼
[5] Classifier (ElasticNet LR or Linear SVM)
```

**Why linear classifiers?** With typical datasets (10–30 animals), complex non-linear models overfit badly. Linear classifiers have fewer free parameters, are easier to regularise, and their coefficients directly indicate which features push predictions toward each group.

### 4.2 LOO Cross-Validation Paths

**Path A (greedy/exhaustive trace available):** LOO runs on only the selected clusters/transitions, using a fixed-C pipeline. Reported accuracy is directly comparable to the bar charts in the Shapley and cluster trace plots.

**Path B (no trace — Feature Source is Groups or Mix):** LOO runs on the full feature matrix with the full elastic-net CV pipeline. Inner CV searches Cs and l1_ratios inside each fold.

### 4.3 Threading Architecture

| Workload | Parallelism | Backend |
|---|---|---|
| Exhaustive combo evaluation | CPU process pool | loky (true multi-core) |
| Greedy step candidate evaluation | Thread pool | prefer="threads" |
| Permutation test folds | Thread pool | prefer="threads" |
| Shapley coalition batch | Thread pool | prefer="threads" |

The exhaustive search uses the loky process pool (not threads) because each `LogisticRegression.fit()` on tiny matrices spends most time in Python overhead — true multi-process bypasses the GIL entirely.

---

## 5. Leave-One-Out Cross-Validation

### Why LOO?

With small datasets (< 30 animals), standard k-fold wastes too much data for training. LOO uses N − 1 animals for training each time — the maximum possible training set. This matters when you have only 6–15 animals per group.

### What LOO Does

Repeats N times (N = total animals):
1. Hold one animal out completely (the "test" animal)
2. Train the full pipeline on the remaining N − 1 animals
3. Predict the group of the held-out animal
4. Record whether the prediction was correct

The final LOO accuracy is the fraction of animals correctly predicted.

### What LOO Accuracy Means

| LOO Accuracy | Interpretation |
|---|---|
| ≈ chance level | Model cannot distinguish groups from behaviour alone |
| Moderately above chance | Groups differ but effect is subtle |
| Much higher than chance | Strong, reliable behavioural differences |
| 100% | Perfect separation — treat cautiously with small N |

**Chance level** depends on the number of groups: 2 groups = 50%, 3 groups = 33%, etc.

> **Key caveat:** 100% LOO accuracy with N < 10 animals per group should be interpreted very cautiously. With 6 animals per group, even a model fitting a single spurious feature can achieve 100% by chance. The permutation test is essential for deciding whether to trust the result.

---

## 6. Balanced Accuracy and Cohen's κ

### Balanced Accuracy

Raw LOO accuracy can be misleading when groups are unequal in size. If you have 12 controls and 3 treated animals, a model that always predicts "control" achieves 80% raw accuracy while being completely useless for the treated group.

**Balanced accuracy** = average per-class recall:
```
Balanced accuracy = (recall_groupA + recall_groupB + ...) / n_groups
```
where recall for each group = (correct predictions for that group) / (total animals in that group).

Balanced accuracy is the **headline statistic** for all comparisons and permutation testing.

### Cohen's κ

Agreement beyond chance:
```
κ = (observed accuracy − chance accuracy) / (1 − chance accuracy)
```

| κ range | Strength |
|---------|----------|
| < 0.20 | Slight |
| 0.20 – 0.40 | Fair |
| 0.40 – 0.60 | Moderate |
| 0.60 – 0.80 | Substantial |
| > 0.80 | Almost perfect |

κ > 0.6 combined with a significant p-value is a strong indicator of a genuine and replicable behavioural difference.

---

## 7. Permutation Test

### How It Works

Builds a null distribution by repeatedly shuffling group labels and re-running the full LOO procedure. Both the observed score and the permuted scores are computed as balanced accuracy, ensuring they are on the same scale.

**p-value formula** (Phipson & Smyth 2010 correction to avoid p = 0):
```
p = (count of permutations ≥ observed_balanced_acc + 1) / (n_permutations + 1)
```

### Permutation Count

| Setting | p-value resolution | Runtime |
|---------|-------------------|---------|
| 99 | ±0.050 | Fastest |
| 199 | ±0.025 | Recommended for exploration |
| 499 | ±0.010 | Recommended for publication |
| 999 | ±0.005 | Most precise, slowest |

### Colour Coding

- **Green:** p ≤ 0.05 (significant)
- **Amber:** p ≤ 0.10 (marginal)
- **Red:** p > 0.10 (not significant)

### Conditional vs. Nested Permutation Tests

| Test | What it does | When to use |
|---|---|---|
| **Conditional p(c)** | Fixes the greedy cluster selection; only re-runs the classifier under shuffled labels | Fast; default; suitable for exploration |
| **Nested p(n)** | Re-runs the full pipeline (greedy selection + LOO) for each permutation | Slower; gold-standard; required for publication |

The nested null is wider (more conservative) because re-running greedy selection per permutation adds an extra source of variance. A result significant under the conditional test but not the nested test means the observed clusters may not be robustly re-selected under shuffled labels.

**To run the nested test:** click **Run Nested Permutation Test** in the left panel after an initial run completes. A toggle then lets you switch between conditional and nested null distributions.

> **Key caveat:** With very small N (< 6 per group), the minimum achievable p-value is bounded by the number of unique label permutations. With 4 animals per group, there are only 4! = 24 possible permutations, so p cannot go below ~0.04 regardless of how different the groups are.

---

## 8. Cluster Selection — Exhaustive and Greedy

Before the main LOO run, the pipeline identifies which clusters or transitions actually discriminate the groups. Feeding all clusters into a tiny LOO loop (n ≈ 12 animals, k = 30+ features) collapses accuracy even with regularisation.

### Exhaustive Search (default when combinations ≤ 15,000)

Evaluates every possible combination of `k` clusters simultaneously. For `k = 5` clusters and up to 19 total clusters, all C(19, 5) = 11,628 combinations are evaluated in parallel using a CPU process pool. The combination with the highest LOO balanced accuracy is selected.

| k (contributors) | n = 15 clusters | n = 20 clusters | n = 24 clusters |
|---|---|---|---|
| 1 | 15 | 20 | 24 |
| 2 | 105 | 190 | 276 |
| 3 | 455 | 1,140 | 2,024 |
| 4 | 1,365 | 4,845 | **10,626** ✓ |
| 5 | 3,003 | **15,504** → greedy | — |

### Greedy Forward Selection (fallback for large spaces)

When C(n_clusters, k) > 15,000, or for the Transition model (up to 380 directed pairs):
1. Start with empty selected set.
2. At each step, evaluate adding each remaining candidate in parallel.
3. Add the candidate giving the highest LOO balanced accuracy.
4. Stop when `max_steps` are reached, or when adding any candidate would *decrease* accuracy.

#### "All" Mode Upgrade

When Max Contributors is set to "All", greedy selection runs first to find the optimal `k`, then an exhaustive search is triggered at that `k` (if C(n_clusters, peak_k) ≤ 15,000). This gives the benefits of both strategies.

### Result Figures

**Exhaustive mode — left panel (Top-N Combinations):** horizontal ranked bar chart showing the top-5 cluster subsets sorted by balanced LOO accuracy. Title tagged `[Exhaustive]`.

**Exhaustive mode — middle panel (Cluster Essentiality):** vertical bar chart showing how many of the top-5 combinations each cluster appears in. Clusters in the single best combo are starred (★).

**Greedy mode:** incremental-gain bar chart (showing each cluster's added accuracy) and cumulative accuracy line. Title tagged `[Greedy]`.

The **Incremental Gain Bar Chart** bars are sorted by incremental contribution, with badge numbers (#1, #2, …) showing greedy selection order. The "tot X%" label shows cumulative LOO accuracy at the step that cluster was added.

---

## 9. Shapley Feature Importances

Shapley values are the game-theoretic fair attribution of credit: the exact expectation of each cluster's marginal contribution averaged across *all possible orders of inclusion* — not just the greedy order.

### Formula

For each selected cluster `c`:
```
φ(c) = Σ_{S ⊆ selected \ {c}} [|S|!(N-|S|-1)!/N!] · [v(S ∪ {c}) − v(S)]
```
where `v(S)` = LOO accuracy using only the clusters in coalition `S`.

### Computation Modes

- **Exact (N ≤ 8 clusters):** All 2^N coalitions are evaluated (2^8 × 8 = 2,048 LOO evaluations).
- **Monte Carlo (N > 8):** 150 random orderings are sampled. All distinct prefix coalitions are collected first and evaluated in parallel, then the phi summation uses dictionary lookups.

### Interpreting Shapley Values

- **φ > 0:** The cluster genuinely contributes to group discrimination on average across all coalition sizes.
- **φ ≈ 0:** The cluster is redundant — its information is already captured by other clusters.
- **φ < 0:** The cluster actively hurts on average — it introduces noise or misleads the classifier in most contexts.

> **Why can a cluster have φ < 0 yet still be selected by greedy?** Greedy picks the best cluster *given the already-selected set*. Shapley averages over *all* sets. A cluster that helps a lot when the dominant clusters are present but hurts when used alone will have positive marginal value in greedy context but negative Shapley. This is the fundamental tension between sequential and average-case importance.

### Verification

The footnote below the Shapley panel shows:
```
Σφ ≈ baseline − chance: ✓
```
This confirms self-consistency: Shapley values should sum to (baseline LOO accuracy) − (chance level). If `≈` appears instead of `✓`, the Monte Carlo approximation is imprecise — interpret values as rough rankings rather than precise contributions.

---

## 10. Controls Reference

| Control | Default | Effect |
|---|---|---|
| **Feature Source** | Individual Clusters | Which features enter Frequency and Duration models (Transition always uses clusters) |
| **Algorithm** | Elastic Net | Switches between Elastic Net Logistic Regression and linear SVM |
| **Prediction Target** | Combined | Multi-factor label combining all assigned experimental group columns |
| **Permutation Count** | 199 | Number of label shuffles; higher = more accurate p-values but slower |
| **Max Contributors** | 5 | Maximum clusters/transitions selected. "All" = no limit (greedy runs to completion, then upgrades to exhaustive at peak k if feasible) |
| **Run Models** | — | Launches the three-model pipeline on a background thread |
| **Cancel** | — | Signals the background worker to stop after the current greedy step. Completed models are displayed normally |
| **Run Nested Permutation Test** | — | Re-runs the full pipeline (greedy + LOO) for each permutation. Available after an initial run |
| **Export CSV** | — | Saves per-animal predicted labels and probabilities for all completed models |

---

## 11. Reading the Results

### 11.1 Comparison Table

| Column | What it shows |
|---|---|
| **LOO / Bal. Acc.** | Simple LOO accuracy / balanced accuracy (shown as two values when they differ by > 0.5%) |
| **Chance** | 1 / n_groups — baseline for a random classifier |
| **Perm. p** | Permutation test p-value. Green ≤ 0.05 / Amber ≤ 0.10 / Red > 0.10 |
| **Cohen's κ** | Chance-corrected agreement (0 = chance; 1 = perfect; negative = worse than chance) |

Star notation: `***` → p ≤ 0.001 · `**` → p ≤ 0.01 · `*` → p ≤ 0.05 · `†` → p ≤ 0.10 · `n.s.` → p > 0.10

### 11.2 Figure 0 — Model Overview (shown immediately after Run)

**Left panel — Accuracy bar chart:** paired bars per model (solid = LOO accuracy, hatched = balanced accuracy). Bar colour reflects p-value. Dashed horizontal line = chance level.

**Right panel — Cohen's κ scatter:** one point per model. Horizontal threshold lines at κ = 0.20, 0.40, 0.60.

### 11.3 Figure 1 — Confusion Matrix + ROC Curve

Click **View** on a model row to display its detail figures.

**Confusion matrix (left):** row-normalised recall (blue scale, 0–1). Diagonal = fraction of animals in each group correctly predicted. Off-diagonal = misclassification rates. Right margin: per-class recall. Bottom margin: per-class precision.

**ROC/AUC (right):** For 2 groups — single LOO curve with AUC. For ≥3 groups — one OvR curve per group plus a macro-average (dashed, shaded). Diagonal chance line included.

### 11.4 Figure 2 — Per-Animal LOO Probability Strips

One dot per animal, sorted by true group then by P(true group). Vertical position = classifier's predicted probability of the animal's actual group. Dots above 0.5 are correctly leaning; below 0.5 indicates uncertain or wrong predictions. Reveals which specific animals the model is most uncertain about.

### 11.5 Figure 3 — Permutation Null Distribution

Histogram of balanced accuracy values from all permuted null models. Dashed vertical line = observed balanced accuracy. Shaded tail region = the p-value. When the observed line sits far right of the null bulk, the model is detecting genuine group structure.

### 11.6 Figure 4 — Feature Contributions

**2 groups — Signed coefficient bar chart:** bars coloured by direction (orange = pushes toward group B; blue = pushes toward group A). Zero-coefficient features (Lasso-eliminated) are omitted.

**≥3 groups — Coefficient heatmap:** rows = groups, columns = top 20 features by max |coefficient|. Reveals group-specific directional patterns invisible in a mean-importance bar chart.

**Below Figure 4 — Shapley importance strip:** shows Shapley φ values for each selected cluster or transition. Shapley values are individually fair attributions, unlike raw coefficients which are jointly determined by correlated features.

### 11.7 Caveat Bar

Auto-generated warnings:

| Warning | Meaning |
|---|---|
| n < 10 per group | Small-n regime; high variance in LOO estimates |
| Only N animals in group X | LOO for that class is especially unreliable |
| n_features > n_animals | Overfitting risk even with regularisation |
| Animal X has < 10 bouts | Transition probabilities unreliable for that animal |
| Session ordering matters | Transition model assumes temporal bout order |

---

## 12. Interpreting the Null Distribution Figure

The null distribution figure (titled *"Group Discriminability — Conditional/Nested Permutation Null Test"*) has one panel per model (Frequency, Total Duration, Transition), allowing side-by-side comparison.

### Histogram (grey-blue bars)

Each bar represents a range of balanced-accuracy values obtained when group labels were shuffled. For a two-group problem, the peak should be near 0.50. A wide histogram means accuracy is highly variable across permutations (common with small N).

### KDE Trace (smooth solid line)

A smoothed probability density estimate of the null distribution, computed via `scipy.stats.gaussian_kde` (Scott's rule bandwidth). The peak of the KDE is the most common null accuracy. A steep, narrow KDE means your model's accuracy only needs to be slightly above the peak to be significant.

### Shaded Tail (coloured region right of observed line)

The area of this shaded tail is proportional to the p-value. Green (p ≤ 0.05) / Amber (p ≤ 0.10) / Red/Vermilion (p > 0.10). A tiny tail = small p-value.

### Chance Line (dotted vertical line)

Vertical dotted line at theoretical chance-level balanced accuracy = 1 / n_groups. The null distribution should peak approximately at this line. A null distribution peak substantially above chance is a warning sign that feature selection may be leaking label information.

### Observed Accuracy Line (dashed vertical line)

Dashed line at the balanced LOO accuracy from your real data. The further right it sits relative to the null bulk, the more unusual your result is under the null hypothesis.

### Annotation Box (top-right corner)

| Statistic | Meaning |
|---|---|
| **Bal. Acc. = X%** | Balanced LOO accuracy from real data |
| **p(c) = 0.045 \*** | Conditional permutation p-value |
| **Z = +1.99** | Z-score: standard deviations above the null mean |
| **Nᵖ = 199** | Number of permutations used |

The **Z-score** is useful for comparing results across datasets and conditions — Z > 2 roughly corresponds to p < 0.05 for a normal null distribution.

### Common Patterns and What They Mean

| Pattern | Biological interpretation |
|---|---|
| All three panels green (lines far right) | Groups differ in frequency, duration, AND transition structure. Robust, convergent evidence. |
| Frequency and Duration significant, Transition not | Groups perform the same behaviours in different amounts but with similar grammar. |
| Only Transition significant | Groups do behaviours with similar frequency/duration but in a different order or context. Examine the transition matrix plots. |
| No model significant (all red) | Groups are behaviourally similar, dataset is too small, or clusters don't capture relevant variation. |
| High accuracy but non-significant p-value | Happens with very small N (e.g. 3 animals per group) — too few permutations to reach p < 0.05 even with perfect discrimination. More animals needed. |

---

## 13. Interpreting the Per-Animal Probability Figure

Titled *"— Per-Animal LOO Predictions"*. Appears below the confusion matrix when the classifier supports probability output (Elastic Net does; SVM does not).

### Anatomy

**Each row** = one animal, sorted by true group then by descending predicted probability of the true class.

**Stacked horizontal bar:** each coloured segment = one experimental group. The width of each segment = the probability the model assigns to that group. The entire bar sums to 1.0.

**Border colour:** Green = model's predicted group matches true group (correct). Red = wrong.

**Left label:** animal's true experimental group. **Right label:** predicted group + ✓/✗ glyph.

**Subtitle line:** shows Balanced Acc and κ for quick reference.

### How to Use This Figure

- **Finding hardest-to-classify animals:** Animals at the bottom of each group block (lowest true-group probability) are the hardest cases — may be behavioural outliers or have recording quality issues.
- **Diagnosing misclassifications:** Export CSV to correlate misclassified animals with metadata (recording day, body weight, experimenter) to check for confounds.
- **Correctly classified but uncertain:** A green border with a nearly equal-width bar means the model could easily flip to incorrect with a slightly different training set. Not a robust prediction.
- **One consistently red animal across all three models:** likely a genuine outlier or labelling error — investigate before concluding groups don't differ.

---

## 14. Export CSV

Click **Export CSV** after running to save animal-level predictions. Columns:

| Column | Description |
|--------|-------------|
| `animal` | Animal name |
| `true_group` | Assigned experimental group |
| `predicted_group` | LOO-predicted group |
| `correct` | True/False |
| `P_true_group` | Classifier's probability assigned to the true group |

The export reflects the currently selected model (highlighted row in the table).

---

## 15. Caveats and Limitations

| Caveat | Trigger | Meaning |
|--------|---------|---------|
| High variance | min group size < 10 | LOO estimates are noisy; interpret cautiously |
| Unreliable LOO | any group ≤ 2 animals | A single misclassification can swing accuracy by 50%+ |
| Overfitting risk | n_features > n_animals | Adaptive PCA is applied but regularisation may still be insufficient |
| Transition unreliable | any animal < 10 bouts | Transition probability estimates have high variance |
| Bout order matters | always (transition model) | Transition model assumes bouts are in temporal recording order |

**Linear models only:** Both Elastic Net and SVM use linear decision boundaries. Non-linear group separation will be missed. Non-linear alternatives require n > 30.

**Correlated features:** B-SOiD clusters are often correlated. Ridge handles estimation but coefficients and Shapley values reflect *joint* importance. Do not interpret individual cluster importances as independent effect sizes.

**Multiple testing:** Running all three models simultaneously increases the probability of at least one spurious p < 0.05. If only one model is significant, especially Transition (which has the largest feature space), replicate in an independent cohort before drawing conclusions.

**HMM vs raw labels:** The Group Predictor uses HMM-smoothed bout labels if `*_hmm` files are present. To use raw MLP labels, remove or rename the `_hmm` files.

---

## 16. Tips and User Guidance

### Where to Start

1. Load your folder and assign experimental groups in the Combined Analysis tab.
2. Open Group Predictor and click **Run Models** with default settings.
3. Look for the model with the lowest permutation p-value — that is the behavioural dimension most discriminated by your manipulation.
4. Click **View** on the best model to see which clusters or transitions drive the separation.

### Improving LOO Accuracy

1. **Try Custom mode:** Test specific cluster combinations hypothesised to differ between groups. Interaction terms are especially powerful — two clusters that individually show no difference may jointly predict treatment.
2. **Use Mix or Custom with groups:** Group-level features (e.g., total locomotion time) often have lower variance than individual cluster proportions.
3. **Increase n:** LOO accuracy is most reliable with ≥10 animals per group. Small n is the single biggest driver of instability — no statistical method fully compensates.
4. **Check group assignments:** Misassigned animals are the most common reason for near-chance accuracy.
5. **SVM with many clusters:** For high cluster counts (≥12), try SVM after selecting a subset in Custom mode.
6. **Use Max Contributors to stress-test:** Re-run with Max Contributors = 1, 2, 3. If accuracy collapses when restricted to 2 clusters, discrimination depends on many subtle features which is harder to replicate. If 1–2 clusters maintain near-full accuracy, the effect is robust.

### Reading Multiple Models Together

- **High accuracy + red p-value with small N:** With 2 groups of 5 animals each (n = 10 total), LOO accuracy of 70% is not statistically significant — the permutation space is simply too small. Treat the permutation p-value as the primary evidence, not the accuracy.
- **Frequency and Transition models agree:** Convergent evidence across two feature types substantially strengthens the finding.
- **Only Transition significant:** Examine which transitions are elevated in the greedy trace — this reveals not just *what* behaviours differ but *when and in what context*.
- **High LOO vs. high balanced accuracy diverge:** If LOO = 80% but balanced = 55%, the model is exploiting group-size imbalance. Use balanced accuracy as the primary metric.

### Comparing Greedy Order to Shapley Values

- Clusters selected early by greedy should have the highest Shapley values.
- A cluster selected early by greedy but with a low or negative Shapley value may only help in combination with the specific clusters selected at later steps — not universally useful.
- A cluster with positive Shapley but not selected by greedy is often informative individually but redundant given the already-selected clusters.

### Re-run After Removing Outlier Animals

If one animal is a clear behavioural outlier (visible in the UMAP or heatmap panels), temporarily remove it and re-run. If accuracy jumps dramatically, the result depends heavily on that single animal — not a robust population-level effect.

### Publication Checklist

- Run 999 permutations for final results (resolution 0.001).
- Run the Nested Permutation Test to confirm cluster selection stability.
- Report both LOO accuracy and balanced accuracy, plus Cohen's κ and the permutation p-value.
- Indicate the number of permutations and whether conditional or nested.
- Note which Feature Source and Max Contributors were used.
- If only one of the three models is significant, acknowledge the multiple-testing consideration.
