# Next steps: lca_algebraic (Option A) after the 16 Jul 2026 offshore + bug-fix session

This is the follow-up to `PLAN_lca_algebraic.md`'s "Completed" sections; read those first,
especially "Completed 16 Jul 2026: DK offshore (Monopile) model + two major bug fixes" for
full detail on what changed and why.

Status: DK onshore and DK offshore (Monopile) are both built, wired, and matching the baseline
to ~0.15-0.19% abs mean/max error on a small smoke test. The next big step is expanding to
other countries and confirming the fixes at full fleet scale.

---

## 1. RESOLVED 16 Jul 2026: the residual ~0.5-1% (and the original ~6%) error

**Root cause found and fixed: `sympy_interp` extrapolated instead of clamping.** The helper's
last `Piecewise` segment used `cond=True` (a catch-all) with the final segment's linear
formula, extrapolating for any x beyond the table's last breakpoint instead of clamping to
`ys[-1]` like `np.interp` actually does (the function's own docstring claimed clamping; the
code didn't do it). This affected **every** percentage-split table in the model for any
turbine with P beyond that table's last breakpoint (many DK turbines are P>800 or P>2000),
almost certainly what "Lead 1" (the `InterpolatedUnivariateSpline` mismatch, originally
suspected too small to matter) was a symptom of; the real bug was much bigger and simpler.
Fixed by adding an explicit right-clamp piece after the loop.

**A second bug found alongside it: the onshore transformer exchange was wrong.**
`built_inventory.py:340-344` (`if offshore==False:`) only ever adds `MV_transfo`, hardcoded
`19/35`, never `HV_transfo` at all (that's offshore-only). The algebraic model had this
backwards (added both, using `LT/35`=20/35).

**Combined effect:** a small smoke test (2 onshore + 2 offshore turbines, 2 methods) now shows
**~0.15-0.19% abs mean/max error** for both models, down from >1% (onshore) and >30%
(offshore, before further offshore-specific fixes; see PLAN doc). Confirming this at full
fleet scale required rerunning the full 50-turbine baseline and full-fleet algebraic
validation; the smoke-test result turned out to be representative (see the confirmation
below).

**RESOLVED 26 Jul 2026: Disposal stage discrepancy.** Root cause found: `built_inventory.py`
(via `scaling.py:404-429`'s `add_to_dict_2`) only nests by component/sub_comp for
`phase=='Input'`; every other phase does a flat `dict[phase][key] = value` with **no
accumulation**. `'Steel, inert waste'` and `'Chromium Steel waste'` are cloned to the same
ecoinvent activity (`prepare_inventories.py:228-230`, confirmed via `activities_and_uuids.pkl`;
both point at `market for scrap steel`), and `'Chromium Steel waste'` is processed last, so
baseline's Disposal exchange silently drops the (usually much larger) low-alloy-steel mass and
keeps only chromium steel's. The algebraic model's `disposal_exc`/`disposal_exc_off` were
*summing* the two instead (the physically sensible thing, but wrong for matching baseline);
fixed to replicate baseline's overwrite instead. Disposal abs-mean error dropped from 62-98%
to ≤0.21% across all 5 validated countries (all onshore + every offshore bucket). Full
writeup, three solution options considered, and per-country before/after numbers in
`PLAN_lca_algebraic.md`, "Completed 26 Jul 2026: Disposal-stage bug, root cause found and
fixed." See item 2b below for the baseline bug report this implies.

**1b. RESOLVED 27 Jul 2026 (DK) / EXPLAINED, not a bug (DE): the Total-stage outlier
turbines flagged above.** Both root-caused using the same trace-to-stage-then-read-the-code
method as the Disposal bug; full writeup in `PLAN_lca_algebraic.md`, "Completed 27 Jul 2026":

- **DK onshore (abs-max 5.72%→0.209%, FIXED):** `built_inventory.py`'s land-use construction
  doesn't give its "land use for onshore wind turbines" activity a P-dependent amount. It
  rewrites the activity's *own internal exchanges* per turbine via `np.interp(P, ...)` on 6
  land-transformation datasets. The algebraic model froze this activity's recipe at whatever
  `P_REF` built the one-time reference inventory, so any turbine whose P differs a lot from
  `P_REF` got the wrong land-use recipe, worst for DK's smallest turbines (P=10-25kW, table
  breakpoints only go up to 800kW). Fixed by expanding each land-use dataset as its own
  `sympy_interp(P,...)/park_size` term directly in `input_exc`, bypassing the frozen activity.
- **DE offshore Monopile/Semi-submersible (abs-max 35.5%/17.7%, UNCHANGED, not a bug):**
  found and fixed a real, separate bug along the way (`sympy_interp` was wrongly clamping
  instead of extrapolating for the one code path that actually uses
  `InterpolatedUnivariateSpline` in baseline, Tower galvanizing/welding; new `sympy_extrap()`
  helper added), but that turned out not to be the actual driver for these 3 turbines. The real
  cause: baseline's per-turbine location lookup resolves this specific offshore turbine's
  electricity market to `location='NL'` (it's geographically closer to Dutch infrastructure
  than to Germany's own reference point), while the algebraic model, by design, freezes one
  background per country and always uses Germany's electricity activity. This is the
  documented "one background per country" simplification breaking down for a border-adjacent
  offshore turbine, not a coding defect; not worth fixing without giving up the country-level
  speedup architecture. Affects 3 turbines total, minor impact categories only (not the
  headline GWP100 Total, which is 1-3% off on these same turbines). Recommend documenting as a
  known limitation in the paper's validation/limitations section.

**Still open: the `Foundation`/`Diesel` bug** (found earlier; confirmed to cancel out
identically between baseline and algebraic model, so not a source of *this* error, but still a
real bug in the original codebase worth reporting): see item 2 below.

**Confirmed at full scale:** `fleet_evaluation_v03_elie.py` (the real 50-turbine + all-methods
baseline) and `fleet_evaluation_lca_algebraic.py` were both rerun after these fixes. The
`fleet_impacts_DK_validation_summary_by_{stage,method,turbine}.csv` numbers confirm the
smoke-test result held at full scale; see item 9's checklist and `PLAN_lca_algebraic.md` for
the full per-country numbers.

---

## 2. Report (and optionally fix) the `Foundation`/`Diesel` bug in `built_inventory.py`

**Why:** independent of the residual-error investigation above, this is a real correctness
bug in the *original*, non-algebraic codebase. The Foundation-phase Diesel assembly exchange
is silently dropped from every single fleet run (onshore and offshore), including the "exact"
baseline everyone validates against. It should be reported to Dominik/the advisor even though
it isn't a source of algebraic-vs-baseline error (both sides inherit the same omission).

**Steps:**
1. Confirm the bug is real: check `built_inventory.py` lines 289-300 (the
   `if component == 'Foundation':` branch of the Assembly non-kg loop, starting around line 275,
   search for `#Assembly activities not in kg`) against the `else` branch immediately after
   (lines 301-306) which *does* call `add_to_dict_2(...)`. The `Foundation` branch computes
   `np.interp(P, df_inv_not_kg_i.index.values, df_inv_not_kg_i.values)` but only uses it inside
   an `if print_details:` block; it never calls `add_to_dict_2()` to actually add the exchange
   to `dict_activities`.
2. Quantify the impact: how much GWP100 does the missing Diesel exchange represent, as a % of
   total turbine GWP? (Multiply the discarded interpolated value by the diesel-burning
   ecoinvent activity's climate-change characterization factor, using
   `bd.projects.set_current('wimby')` and the existing `wimby` project.)
3. Decide with the advisor whether to fix it. If fixed, every existing "exact" baseline
   number would shift slightly and need re-generating. This affects every existing baseline CSV
   in the repo, so it needs sign-off before regenerating those.

---

## 2b. Report (and optionally fix) the `add_to_dict_2` non-Input overwrite bug in `scaling.py`

**Why:** a second, independent real bug in the *original* baseline codebase, same category as
the Foundation/Diesel bug above (found while root-causing the Disposal-stage discrepancy; see
item 1 and `PLAN_lca_algebraic.md`, "Completed 26 Jul 2026"). `scaling.py:404-429`'s
`add_to_dict_2()` nests by `component`/`sub_comp` for `phase=='Input'` only; every other phase
(`Assembly`, `Transport`, `Maintenance`, `Disposal`) does a flat `dict[phase][key] = value`
with no `+=`, so if two source materials ever route to the same downstream ecoinvent activity
under a non-Input phase, the second one silently overwrites (not adds to) the first. Currently
this only bites once in practice (`'Steel, inert waste'`/`'Chromium Steel waste'` sharing
`market for scrap steel` in Disposal), but it's a latent defect that would silently corrupt any
*future* case where two Assembly/Transport/Maintenance/Disposal datasets are mapped to the same
activity, not just this one. Affects every historical "exact" baseline fleet CSV, same
consequence as item 2's bug.

**Steps:**
1. Confirm scope: grep `built_inventory.py` for every `add_to_dict_2` call with `phase` not
   `'Input'`, and check whether any *other* pair (besides Steel/Chromium) shares a UUID within
   the same phase, using `activities_and_uuids.pkl`, `.duplicated('UUID', keep=False)` grouped
   by `Phase`, the same check used to confirm the Disposal case.
2. Quantify: for Disposal, the missing low-alloy-steel mass is 3-11x the kept chromium-steel
   mass depending on turbine size (see `PLAN_lca_algebraic.md` for the `df_perc.pkl`-based
   calculation). Translate to a GWP100 % of Total using the "market for scrap steel"
   characterization factor.
3. Decide with the advisor whether to fix `add_to_dict_2` itself (adding `+=` for the non-Input
   branch). If fixed, every existing "exact" baseline number shifts slightly (Disposal-stage
   GWP100 would increase, since the currently-dropped low-alloy-steel mass would count) and
   needs regenerating, same tradeoff as item 2.

---

## 3. DONE: Monopile (16 Jul) and Semi-submersible (17 Jul) offshore models, validated for real

Both are built and confirmed against real baseline runs (not just smoke tests); see
`PLAN_lca_algebraic.md`'s "Completed" sections for full detail:

- **Monopile**: validated on DK (604 turbines, 100% Monopile) and again on BE (270 of 401
  offshore turbines): 0.16% abs mean error on both.
- **Semi-submersible**: validated on BE (131 of 401 offshore turbines, sea_depth 9-36m),
  **0.31% abs mean error, first real test, no bugs found** (the numerically-derived formulas
  were correct on the first attempt).
- **Spar buoy**: still only numerically verified against `scaling.py`, never run against a real
  turbine. Checked EU-wide: **only Norway has Spar-buoy-range turbines (sea_depth >60m), and
  only 2 of them** (up to 207m, Norway's real floating wind projects). Every other country
  tops out at ≤56m. So this one specifically needs a Norway run, and even there it's a
  2-turbine edge case, worth doing, but should not be expected to produce a large validation
  sample.
- **Known bug in the baseline itself, reproduced as-is:** `scaling.spar_buoy_floating_foundation()`
  returns mass in tonnes but `built_inventory.py` uses it directly as a kg amount (~1000x too
  little). Not our approximation; matched for comparability. Flag as a separate bug report.

**Norway/Spar buoy validation, when it's done:** set `COUNTRY = 'NO'`, run
`precompute_geo_columns.py` for Norway, confirm the bucket distribution includes those 2 Spar
buoy turbines, run the real baseline (`fleet_evaluation_v03_elie.py`) and the algebraic model,
and report the validation error the same way DK/BE were validated (see `PLAN_lca_algebraic.md`,
"Completed 16-17 Jul 2026"). The known tonnes-vs-kg bug in
`scaling.spar_buoy_floating_foundation()` should be reproduced as-is, not silently fixed, since
that would break comparability with the baseline.

---

## 4. DONE (mechanism + confirmation batch), IN PROGRESS (full coverage): generalized to any country

**Update 21 Jul 2026 (confirmation):** confirmed on a 3-country batch (NO/DE/GB, ~148 turbines)
chosen to cover all 4 test cases (onshore + Monopile + Semi-submersible + Spar buoy) at once;
see `PLAN_lca_algebraic.md`'s "Completed 21 Jul 2026" section for full detail, including a real
`lca_algebraic_bw25` library bug found and worked around (batch size of exactly 1 turbine).
Combined with DK/BE, all 4 test cases are now validated on ≥2 countries each, ~0.1-0.7% abs
error.

**Update 21 Jul 2026 (full rollout): DONE, all 38 countries now have algebraic fleet results.**
`fleet_evaluation_lca_algebraic.py` was changed to accept the country as `argv[1]`
(`COUNTRY = sys.argv[1] if len(sys.argv) > 1 else 'GB'`), and `precompute_geo_columns.py`'s
`validate_against_originals` sample size was capped to `min(15, len(pool))` (several
micro-countries have fewer than 15 onshore turbines, Iceland and Slovenia have only 2 each,
and the old hardcoded `n_sample=15` would have crashed `pandas.sample()`). Geo columns were
precomputed for all 33 remaining countries, then `run_remaining_countries.sh` drove
`fleet_evaluation_lca_algebraic.py` once per country (one subprocess each, so Brightway/
lca_algebraic state never leaks across countries): **0 failures across all 33**, ~15-20s
setup + <1s compute each. All `fleet_impacts_<ISO>_lca_algebraic.csv` (38 total, one per
country) and per-bucket offshore files (wherever that country has any: FI/IE/SE/NO/NL/DE/GB/
BE/DK have Monopile, DE/GB/NL/BE/NO/FR have Semi-submersible, only NO has Spar buoy) now exist
in `REWIND/REWIND/data/`. Sanity-checked: every country's GWP100 mean falls in the
0.0135-0.0345 kg CO2eq/kWh range, consistent with the validated countries; no zeros, no
NaNs, no outlier countries. **Full EU fleet is now covered; this item is closed.**

Both scripts are now parametrized by country instead of hardcoded to DK (see
`PLAN_lca_algebraic.md`, "Completed 17 Jul 2026"):

- `fleet_evaluation_lca_algebraic.py`: a single `COUNTRY = 'DK'` variable (section 0) drives
  everything: file names, foreground DB name, the reference point (`LON_REF`/`LAT_REF`, now
  computed dynamically as the mean lon/lat of that country's own onshore turbines, not
  hand-picked per country).
- `fleet_evaluation_v03_elie.py`: needed almost no change (every per-turbine call already used
  real lon/lat), just widen `selected_countries`. Its offshore sampling was also improved to
  stratify by foundation-type bucket (`OFFSHORE_SAMPLE_PER_BUCKET`, default 20 per bucket)
  instead of a blind `.head(N)`, which under-samples rare buckets (BE's first 20 offshore rows
  had only 1 of 131 Semi-submersible turbines).
- **Confirmed working on a second country (BE)**: see item 3, all three models (onshore,
  Monopile, Semi-submersible) matched their real baselines within 0.1-0.31% abs mean error.

**The general recipe for adding a country** (`ISO_code`, sizes vary from 2 turbines to 27,916
for Germany, but sample sizes stay constant regardless of country size, so runtime per
country is similar to DK/BE, roughly 90 min for baseline + ~1 min algebraic):

1. Set `COUNTRY` in `fleet_evaluation_lca_algebraic.py` and add the country to
   `selected_countries` in `fleet_evaluation_v03_elie.py`.
2. Run `precompute_geo_columns.py` for that country.
3. Check its offshore bucket distribution (printed automatically); if it includes Spar buoy,
   see item 3's note (only Norway does, and only 2 turbines).
4. Run the real baseline, then the algebraic model, and confirm the validation numbers.

This recipe has already been applied to all 38 countries (see the full rollout above); this
section is kept as reference for anyone re-deriving a country's results from scratch.

---

## 5. Unresolved: DK onshore fleet stats vs. the paper's original numbers (lower priority, not a correctness blocker)

**Not urgent, this is a discrepancy against an external, non-reproducible historical result,
not a bug in the current pipeline.** Flagging it so it isn't forgotten, not because it needs
fixing before anything else above. Note: given item 1's fixes, this comparison should be
rerun too; the +8%/-37% numbers below predate the `sympy_interp` and transformer fixes.

**What was found (16 Jul 2026):** `Shared_Rewind/3_submission/Scripts_evaluation/Fleet_basic_charts_validation.ipynb`
contains Dominik's own summary statistics for the DK onshore fleet (from the CSVs behind the
paper). Same population size (4,328 turbines) as the current fleet, but the distributions
didn't match exactly (numbers below are from *before* the sympy_interp/transformer fixes in
item 1, worth recomputing once the fleet-wide rerun in item 1 is done):

| | count | mean | std | min | 25% | median | 75% | max |
|---|---|---|---|---|---|---|---|---|
| Paper (original) | 4,328 | 0.030875 | 0.037010 | 0.007120 | 0.012482 | 0.014282 | 0.019504 | **0.190980** |
| Ours (pre-fix)    | 4,328 | 0.031913 | 0.036895 | 0.007786 | 0.013533 | 0.015470 | 0.021083 | **0.119493** |
| Diff | (n/a) | +3.4% | -0.3% | +9.4% | +8.4% | +8.3% | +8.1% | **-37.4%** |

(units: kg CO2-eq/kWh, EF v3.1 GWP100; same method tuple confirmed on both sides.)

**What was ruled out:**
- Not a sample-size artifact: both sides are the full 4,328-turbine fleet, not the 50-turbine
  baseline subset.
- Not a different codebase: `REWIND/REWIND/` **is** Dominik's own live code (confirmed by him),
  not a separate reimplementation. So this is "the same code at two points in its history," not
  two different methods disagreeing.

**What was found while investigating, but not resolved:**
- Tried to diff against the exact code version that produced the paper's numbers:
  `Shared_Rewind/Model/ReWind/REWIND/{built_inventory,scaling,calculations,...}.py` looks like
  the right historical snapshot, but every one of those files is **0 bytes** (alongside
  `___All_Errors.txt` / `____init__.py_Error.txt`, also empty), a failed/incomplete copy, not
  a usable snapshot. No pinned `bw2data`/`bw2calc`/ecoinvent versions anywhere in that folder's
  `pyproject.toml`/`PKG-INFO` either, so a library-version explanation can't be checked that way.
- Found real evidence that this codebase has a history of transient failures needing manual
  patching: the same notebook shows Dominik identifying **27 turbines that silently failed**
  in his original full-EU-fleet run (`Total == 0`, caught by the same bare `except: stage=0`
  fallback still in `v02.py`/`v03_elie.py` today; see item 11) and recomputing them separately
  (`fleet_impacts_uncalculated_normalized_kWh.csv`). **None of the 27 are DK turbines** (all
  `NL_*`/`PT_*`), so this specific incident doesn't explain the discrepancy directly, but it
  does confirm the pipeline's results have never been from one single clean deterministic run,
  which is why an exact match isn't necessarily expected.
- Leading hypothesis for the systematic **+8%** across most of the distribution (may now be
  substantially explained by the sympy_interp/transformer fixes in item 1, recompute this
  comparison after the fleet-wide rerun): the documented bug fixes applied to this codebase on
  13 Jul 2026 (land-use activity `type='production'`→`'process'`, transformer lookup
  `IndexError` fix), or a `bw2data`/`bw2calc` version difference between Dominik's original
  environment and this one.
- The **max discrepancy (-37%, opposite direction from everything else)** is still unexplained.
  Likely driven by one or a few specific outlier turbines rather than a systematic cause;
  Dominik's raw per-turbine CSV (only accessible on his machine) would be needed to identify
  which.

**To actually resolve this:** ask Dominik directly (a) what code state/environment produced
the `Fleet_results` CSVs behind the paper, and (b) for the raw per-turbine CSV (not just this
notebook's aggregate stats) so a turbine-by-turbine diff becomes possible, matching on
Longitude/Latitude (not index, since these may not align).

---

## 6. DONE 26 Jul 2026: generate maps of the fleet / impact results

`generate_paper_figures.py` (repo root) produces 84 PNGs into `figures/`: 9 headline
GWP100 figures adapted from `Shared_Rewind/4_submission`'s Figure5/Figure6 (country ranking,
NUTS3 subnational choropleth using `Shared_Rewind/Fleet_results/NUTS_RG_20M_2024_4326.gpkg`,
turbine scatter map, offshore foundation-type map, plus 5 validation-error figures), and the
same country-ranking/choropleth/turbine-scatter trio repeated for all 25 EF v3.1 impact
categories in `figures/by_metric/`. Re-run any time the underlying
`fleet_impacts_*_lca_algebraic.csv` data changes (e.g. after the Disposal-stage fix; see item
1). It reloads everything from disk and takes under a minute.

---

## 7. TODO: generate summary statistics for the paper

**Why:** the fleet-wide algebraic results (38 countries, 25 EF v3.1 methods, per-stage
breakdowns) are sitting in `REWIND/REWIND/data/results/fleet_impacts_<ISO>_lca_algebraic*.csv`
but haven't been aggregated into paper-ready summary tables/figures yet.

**Candidate stats to compute (this list is a starting menu, not a fixed spec; confirm scope
before generating everything):**
- EU-wide GWP100 distribution (mean/median/std/min/max), and the same broken down by
  onshore vs offshore and by foundation-type bucket.
- Per-country ranking (best/worst mean GWP100). DE's huge fleet (36% of the EU total) will
  dominate any unweighted EU-wide average, so decide whether to report a simple mean, a
  fleet-size-weighted mean, or both.
- Stage contribution breakdown (Input/Assembly/Transport/Maintenance/Disposal as % of Total)
  EU-wide and by country, using the `_by_stage` CSVs.
- Correlation checks: GWP100 vs rated power, vs turbine age/commissioning date, vs park size,
  useful for the discussion section.
- Cross-check EU-wide summary stats against the "Unresolved" comparison in item 5 above (the
  paper's original DK numbers) now that the full fleet is available, if useful context.

**Before building:** same as item 6, confirm which stats matter most for the paper structure
before generating everything speculatively.

---

## 8. TODO: clean up repo bloat before pushing

**Why:** the repo is currently 3.3G locally (`REWIND/REWIND/data/` alone is 2.8G), almost
entirely from files that are either regenerable, intermediate/debug output, or shouldn't be
committed at all (reference PDFs). Only a fraction of this is needed for someone else to
reproduce or continue the work.

**Findings from a 21 Jul 2026 audit, concrete cleanup candidates:**

- **`REWIND/REWIND/data/datasets/` (2.6G)**: the extracted ecoinvent spold files. Already
  `.gitignore`d, not a git problem, but worth confirming it's genuinely excluded before any
  `git add -A`.
- **`fleet_impacts_*_lca_algebraic_by_stage.csv` (253M combined, e.g. `DE` alone is 73M+15M)**:
  full per-turbine × per-stage × per-method breakdowns for the whole EU fleet. Likely too
  granular to commit to git as-is. Options: keep only country-level aggregates in git and
  publish the full per-turbine files via a data-release mechanism (Zenodo, institutional
  storage) referenced from the README; or `git lfs` if these should live in the repo itself.
  Decide before committing, don't just git-add everything.
- **`*_validation_errors.csv` (8.3M) and `*_baseline_all_methods.csv` (1.9M)**: intermediate
  validation debug output from the DK/BE/NO/DE/GB confirmation batch. Small enough to keep as
  evidence for the paper's validation appendix, but confirm that's actually wanted rather than
  assuming.
- **Two reference PDFs at repo root** (`IJLCA_ReWind_Huber_2025_manuscript_5.pdf` 1.1M,
  `s11367-021-01993-z.pdf` 2.3M): third-party copyrighted papers currently untracked. Should
  probably stay untracked/gitignored and be cited/linked instead of committed.
- **`logs/` (548K)**: run logs (baseline + algebraic batch stdout). Pure scratch output, not
  source of truth for anything (the actual results are the CSVs), safe to gitignore or delete
  entirely.
- **Stale staged deletions already in the git index**: `REWIND/REWIND.egg-info/*`,
  `REWIND/REWIND/__pycache__/*.pyc`, and four root-level `.pkl` files show as staged `D`
  (deleted). These were presumably removed once `.gitignore` started excluding them, but the
  removal was never committed. Needs a cleanup commit to actually finalize it.
- **Untracked pipeline scripts** (`fleet_evaluation_lca_algebraic.py`,
  `fleet_evaluation_v03_elie.py`, `precompute_geo_columns.py`, `run_remaining_countries.sh`,
  `REWIND/REWIND/temp.py`, `brightway_example_for_elie.ipynb`): these ARE the actual pipeline
  code and should very likely be committed, unlike the data above. `temp.py` is a one-time
  ecoinvent-import setup script (see `PLAN_lca_algebraic.md`), worth keeping for
  reproducibility even though it's only run once.

**Before executing any deletion/gitignore/commit here, decide which of the above should be
kept vs. dropped vs. moved to external storage.** This is a judgment call about what
"necessary" means for the intended destination (paper submission repo? working copy? public
GitHub?), not something to decide unilaterally.

---

## 9. TODO: fix the significance-filter design in the validation summary (found/agreed 28 Jul 2026, current design is a bad idea)

**Why:** `validate_against_baseline()` in `fleet_evaluation_lca_algebraic.py` (see
`_err_summary` and the `significant` column) excludes any (turbine, stage, method) row where
`|baseline_value| < 0.1% * |that turbine+method's Total|` from every %-based summary stat
(`mean_error_pct`, `abs_mean_error_pct`, `abs_max_error_pct` in the
`*_validation_summary_by_*.csv` files). The underlying motivation is real: relative error is
mathematically meaningless when dividing by a near-zero baseline (some Disposal recycling
credits are ~1e-14), but the mechanism chosen (drop the row from the summary) is the wrong
fix. Confirmed on the actual data on 28 Jul 2026:

- **Doesn't affect the headline number.** Total-stage rows are always 100% "significant" by
  construction (`baseline_value == total_baseline_value` when `stage=='Total'`), so the
  0.018–1.09% abs-mean/abs-max numbers actually used to claim the model's accuracy are
  unaffected, computed over the full, unfiltered turbine × method population, no exceptions.
- **But it silently deletes an entire stage's row from 4 of 5 validated countries' reports.**
  Measured: Maintenance rows excluded = 90% (DK), **100% (BE, DE, GB, NO)**. When a groupby
  produces zero significant rows for a stage, that stage's row simply doesn't appear in
  `fleet_impacts_<ISO>_validation_summary_by_stage.csv` at all, not a small/near-zero number,
  an absent line, indistinguishable from "this stage was never checked" to anyone reading the
  CSV without also reading the code.
- **Disposal is also thinned substantially:** 51–66% of rows excluded across the 5 countries.
  Recomputed the filtered vs. unfiltered Disposal abs-mean/abs-max side by side (28 Jul 2026);
  today's difference is small (a few hundredths of a percentage point, worst case GB's
  abs-max 1.53%→1.59%), because the bugs that used to cause 500%+ blowups on these rows are
  already fixed (item 1). But the *mechanism*, silently vanishing a stage's row when its
  denominator is small, has no way to catch a *future* regression in Maintenance-stage code
  for BE/DE/GB/NO specifically, since there would be no row left to look suspicious.

**RESOLVED 28 Jul 2026: fix applied and confirmed.**

1. `mean_abs_error` (absolute units, already computed **unfiltered** on every row) is now the
   primary per-stage metric alongside a filtered percentage; it was never subject to the
   near-zero-division problem in the first place.
2. Added `median_error_pct`/`abs_median_error_pct`, computed over **all** rows (robust to a
   handful of blown-up outliers without deciding in advance which rows to exclude).
3. The only rows now excluded from the %-based stats are the literally degenerate case,
   `baseline_value == 0` (true division by zero), not the old arbitrary "<0.1% of Total"
   threshold.
4. Added `n` (row count) to every summary row, so a thin sample is visible rather than a
   silently missing stage.

**Scope:** `_err_summary()` and `validate_against_baseline()` in
`fleet_evaluation_lca_algebraic.py`. All 5 validated countries' summary CSVs
(`fleet_impacts_<ISO>_validation_summary_by_{stage,method,turbine}.csv`) were regenerated after
the fix; Maintenance now appears in all 5 countries' reports, and the Total-stage/headline
GWP100 numbers are unchanged, as expected since Total rows were never filtered.

---

## 10. TODO: checklist to review every change and fix made this session

**Why:** many independent fixes have accumulated across ~2 weeks (13–28 Jul 2026) touching
both the algebraic model and the validation infrastructure. Before the paper write-up locks
in these numbers, each item below should be re-verified against the *current* code/data, not
just trusted from memory of when it was first fixed, and checked off only once confirmed
still correct in the live repo.

**Environment/setup fixes (13 Jul 2026), one-time, unlikely to regress**
- [ ] `bw2setup()` list-vs-tuple LCIA key patch still present in `prepare_inventories.py` / `temp.py`
- [ ] `ecoinvent_setup()`'s nested-`datasets/`-folder auto-detection still present
- [ ] `power_transformer.py`'s `transfo_500mva()`/`transfo_10mva()` existence check uses the list-comprehension form (not the `[0]`-indexing form that crashes on an empty list)
- [ ] Land-use activities in `built_inventory.py` still created with `type='process'`, not `'production'`

**Model-accuracy bugs (fixed in the algebraic model), re-verify against current code**
- [ ] `sympy_interp` clamps (not extrapolates) past the last breakpoint, right-clamp piece present
- [ ] `sympy_extrap` (the deliberate exception, for the one `InterpolatedUnivariateSpline` code path: Tower galvanizing/welding) is applied only where documented, not accidentally to every table
- [ ] Onshore transformer: only `MV_transfo` at `19/35`, no `HV_transfo`, confirm still true in `input_exc`
- [ ] Offshore transformer: both `MV_transfo` and `HV_transfo` at `LT/35`, confirm still true in `build_offshore_model()`
- [ ] Offshore `M_nacelle_off_sym`/`M_rotor_off_sym` use the offshore-specific coefficients, not the onshore ones
- [ ] `disposal_exc`/`disposal_exc_off` overwrite (not sum) when two datasets share an ecoinvent activity, matches baseline's bug, not the "corrected" physically-sensible version
- [ ] DK/EU onshore land-use: each of the 6 land-use datasets is its own `sympy_interp(P,...)/park_size` term in `input_exc`, not a frozen reference-activity amount

**Deliberate approximations tightened (v1→v3), confirm nothing regressed to an older version**
- [ ] Material percentages use exact `sympy_interp` Piecewise (not `np.polyfit`) for every component, including `M_reinf`/`M_elec`
- [ ] Transport distances (`dist_rotor`/`dist_nacelle`/`dist_tower`/`dist_to_grid`) are real per-turbine values from `<iso>_geo_precomputed.csv`, not fleet-average constants
- [ ] Cable mass uses the real per-turbine `dist_to_grid`, not the old `P/P_REF` proxy

**Real baseline bugs, confirmed but NOT fixed (status should stay "reported, not silently fixed" unless the advisor has signed off)**
- [ ] Foundation/Diesel omission (`built_inventory.py`, Assembly non-kg loop): still reproduced as-is, not fixed; check it hasn't been "fixed" without advisor sign-off (would silently change every historical baseline CSV)
- [ ] `add_to_dict_2` non-Input overwrite bug (`scaling.py`): still reproduced as-is in the algebraic model's Disposal logic; same sign-off caveat
- [ ] Spar-buoy tonnes-vs-kg unit bug (`scaling.spar_buoy_floating_foundation`): still reproduced as-is, not "fixed"

**Explained outliers, confirm they're still just documented, not chased into further "fixes"**
- [ ] DE offshore Monopile/Semi-submersible's NL-border electricity-market outlier: still documented as a known "one background per country" limitation, not turned into a per-turbine background lookup (which would undo the whole speedup)
- [ ] Offshore Transport's use of the onshore-style default foundation mass: still reproduced as-is (matches baseline), not "corrected" to the real offshore mass

**Validation-methodology fixes, re-verify**
- [ ] `assert_same_turbines()` still called before every cross-file comparison and still raises (not warns) on mismatch
- [ ] Method-name normalization (`method_key`) still used instead of trusting column order between baseline and algebraic method lists
- [x] Significance-filter redesign (item 9): done, unfiltered `mean_abs_error`/median in place, all 5 countries' validation summaries regenerated

**Library workarounds (lca_algebraic_bw25), re-verify still needed/present**
- [ ] `clear_caches()` still called at the top of every run
- [ ] `.reset_index(drop=True)` still applied after every `compute_impacts()` call
- [ ] Per-stage `compute_impacts()` calls (not `axis="phase"`) still used for the stage breakdown
- [ ] Batch-size-1 float workaround still present in the offshore driver loop

**Coverage / rollout**
- [ ] All 38 countries' `fleet_impacts_<ISO>_lca_algebraic.csv` (+ offshore bucket files) are from the *post*-27-Jul-fixes run, not a stale pre-fix batch (check file mtimes against the fix dates)
- [ ] All 84 figures (`figures/`) were regenerated after the same fixes (check mtimes)

---

## 11. TODO: the silent-fail-to-zero fallback in the baseline scripts (found 28 Jul 2026)

**Why:** `fleet_evaluation_v02.py`, `fleet_evaluation_v03_elie.py` (the actual "exact" ground
truth every algebraic-model validation is measured against), and `fleet_evaluation_redo_lci.py`
all wrap their per-turbine loop in a broad `except Exception as e:` that prints the error and
then, for `v02.py`/`v03_elie.py`, writes a fabricated `0` for every lifecycle stage of that
turbine into the output CSV, indistinguishable from a genuine (if implausible) zero-impact
result. `redo_lci.py`'s version just drops the turbine from its output list, which is milder
(no fake data) but still leaves no trace in the final CSV beyond a stdout line. This is the
same mechanism already documented in `PLAN_lca_algebraic.md`'s item-5 discussion as having
caused 27 turbines to silently fail in Dominik's original full-EU run; it's still live in the
current scripts, unchanged.

**Checked 28 Jul 2026, confirmed clean today:** queried every one of the 175 turbines actually
used to validate the algebraic model (DK/BE/DE/GB/NO, onshore + every offshore bucket) for
`Total == 0` in their `*_baseline_all_methods.csv`; zero hits in all 5 countries. So today's
validated numbers are not affected. But the mechanism itself is a live risk: any *future*
baseline rerun (or a first-time baseline run for any of the 33 non-validated countries) could
silently inject fake zeros with nothing in the output CSV to flag it, and a downstream
comparison (this script's `assert_same_turbines()`/`validate_against_baseline()`) would happily
treat that zero as ground truth rather than catching the failure.

**Recommended fix:** stop writing a fabricated `0` on exception. Instead, write the failing
turbine's index/error message to a separate `<output>_failed_turbines.csv` (or an explicit
`status` column on the main output, e.g. `'ok'`/`'failed: <error>'`) and leave that turbine's
stage columns as `NaN`, not `0`, so a failure is visibly distinguishable from a real result at
every downstream step, instead of silently blending in as ground truth.

**Scope:** the per-turbine `try/except` blocks in `fleet_evaluation_v02.py`,
`fleet_evaluation_v03_elie.py`, and `fleet_evaluation_redo_lci.py`. Since these are the
"baseline"/"exact" scripts (not the algebraic model), any change here needs the same care as
item 2/2b's baseline-code changes: confirm with the advisor before altering the actual
baseline-generation logic, since a rerun would be needed to produce the new failure-tracking
columns.

---

## 12. DONE 31 Jul 2026: buses.csv had zero coverage for 5 countries (BY, CY, FO, IS, XK)

`buses.csv` is PyPSA-Eur's transmission-bus extraction, and PyPSA-Eur documents excluding
non-synchronous/isolated grids, so Belarus, Cyprus, the Faroe Islands, Iceland, and Kosovo had
no points at all. Cable length for turbines in these countries was really "distance to the
nearest bus in a different country" (86-1047 km).

Tried Gridfinder's `grid.gpkg` first, rejected it (84% systematic gap vs. `buses.csv` since it
mixes in all-voltage lines, not just transmission). Replaced with a direct Overpass query for
OSM `power=substation` points at 220 kV and above (falling back to a country's own top tier for
grids that never reach 220 kV), validated to within a few percent of `buses.csv` for most of the
33 already-covered countries. Rolled this out to all 38 fleet countries for consistency, not
just the 5 missing ones.

Along the way found that the exact baseline (`fleet_evaluation_v03_elie.py`) never actually used
any of this: `calculate_closest_distance()` in `prepare_inventories.py` reads `buses.csv`
directly and doesn't go through `geo_precomputed` at all, so the first baseline regeneration
attempt changed nothing. Fixed by pointing that function at the new file too. Full writeup,
numbers, and the false start with Gridfinder are in `PLAN_lca_algebraic.md`, "Completed 31 Jul
2026."

---

## Quick reference: files this work touches

```
precompute_geo_columns.py                                    : covers onshore+offshore per country
  writes -> REWIND/REWIND/data/geo_precomputed/<iso>_geo_precomputed.csv
fleet_evaluation_v03_elie.py                                  : baseline, onshore + offshore, SAMPLE_CONFIG per country
  writes -> REWIND/REWIND/data/baseline/fleet_impacts_<ISO>[_offshore][_baseline_all_methods].csv
fleet_evaluation_lca_algebraic.py <ISO>                       : algebraic, onshore + offshore (all 3 buckets), COUNTRY via argv[1]
  writes -> REWIND/REWIND/data/results/fleet_impacts_<ISO>[_offshore_<bucket>]_lca_algebraic[_by_stage].csv
  writes -> REWIND/REWIND/data/validation/fleet_impacts_<ISO>[_offshore_<bucket>]_validation_*.csv
run_remaining_countries.sh                                    : drives fleet_evaluation_lca_algebraic.py across a country list
generate_paper_figures.py                                     : reads results/ + validation/, writes figures/
PLAN_lca_algebraic.md                                         : "Completed" sections, full history
NEXT_STEPS_lca_algebraic.md                                   : this file
Shared_Rewind/wind_fleet_data_incl_sea_depths_corrected.csv   : advisor-confirmed sea depth source

Note (28 Jul 2026): REWIND/REWIND/data/ was reorganized from one flat folder into
geo_precomputed/, results/ (+ results/by_stage/), validation/, and baseline/ subfolders.
EU_turbines_input_data.xlsx, buses.csv, and the ecoinvent datasets/ folder stayed at the top
level as reference inputs. All scripts above were updated to match.
```
