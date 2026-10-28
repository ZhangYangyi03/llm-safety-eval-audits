# Measurement audits of LLM safety evaluation

Three manuscripts and the raw data behind every number in them. Each is currently
under technical check at the journal named below. No paper here has been through
peer review; none is a published result.

The common subject is measurement validity: in each case an evaluation instrument
was taken on faith, and re-deriving its numbers from the stored evidence did not
reproduce what the instrument reported.

## 1. `01_lsdc_parsing/` — How Response Parsing Manufactures Scaling Laws

**Target:** Discover Artificial Intelligence (submission `7d81ba60`), submitted 2 Oct 2026.

A response parser fell back to scanning the entire model response for the tokens
`yes`/`no` whenever a line-wise parse recovered too few answers. The fallback fired
in 28 of 120 trials, and the scoring denominator counted only the entries the
fallback filled. The collapse phenomenon replicates under a strict parser; the
scaling exponent reported on top of it does not.

Data: `data/api_800_trials_raw.jsonl` (944 scored API trials, 8 model families,
k = 1..20), the two local-model logs, the decomposition run, and the failed local
ladder (models that would not load). Every table in the paper is recomputable from
these files alone.

Note on the convention: `k*_(tau=0.95)` is the **largest** k whose accuracy is at
least 0.95, not the first k at which it drops below. The two readings give
different tables.

## 2. `02_judge_instrument_validity/` — Neither the Judge nor the Rules

**Target:** Machine Learning (submission `0e20dc87`), submitted 2 Oct 2026.

Safety-response labelling is done either by an LLM judge or by a hand-written
rule-based classifier. This paper audits both against a hand-labelled benchmark and
finds neither is accurate: the judge recalls none of the genuinely unsafe
responses, and the rule classifier that was treated as ground truth is correct on
53% of audited items. Agreement between the two is 61.6%, but the chance-corrected
agreement is kappa = 0.115 — the raw figure is a majority-class effect.

Data: `data/hand_labelled_benchmark_32.json` (the 32 hand labels plus both
instruments' outputs on each item), `data/judge_rerun_99_raw.jsonl` (the 99-item
judge re-run), `data/summary_statistics.json`.

## 3. `03_attribution_floor_effect/` — Attribution Under a Floor

**Target:** Discover Artificial Intelligence (submission `71026654`), submitted 2 Oct 2026.

Single-component ablation is the standard way to locate safety-critical layers and
neurons in a fine-tuned model. In the audited run three of four prompts were
already unsafe before any ablation, so the per-layer unsafe count saturates and the
reported set of "critical" layers is whichever ones flip the single remaining
prompt. The same run reports 205 safety-critical neurons above an impact threshold
of 0.1; the stored attribution results contain one, with a largest value of 0.156.

Data: `data/attention_ablation_per_prompt.json` (the 32-layer x 4-prompt grid with
baselines) and `data/neuron_attribution.json`. Every recomputed statistic in the
paper is derived from these two files.

## Reproducing

Each `data/` directory is self-contained JSON/JSONL. Figures and tables were
computed with Python standard library only; no modelling code is needed to check
any number claimed in the three papers.

    python scripts/verify_readme_numbers.py

That script recomputes, from the committed data, every figure quoted above and
exits non-zero on any drift: 944 scored API trials over 8 model families, the
fallback firing in 28 of 120 local trials, k* = 1 and 13 under a strict parser,
21/32 and 17/32, 0 of 4 judge recall, 61.6% raw agreement against kappa 0.115,
3 of 4 prompts already unsafe, 205 listed safety-critical neurons above an
impact threshold of 0.1 while the stored attribution results contain one, with a
largest value of 0.156. It runs in CI on every push. A repository whose subject
is unverified numbers should not itself be a claim taken on faith.

## Contact

Yangyi Zhang — Department of Computer Science, Georgetown University —
yangyi.zhang@georgetown.edu
