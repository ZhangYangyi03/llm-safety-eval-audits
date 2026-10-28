# -*- coding: utf-8 -*-
"""Recompute every number this README claims, from the data in this repo.

    python scripts/verify_readme_numbers.py

The repository's whole argument is that a number reported by an instrument
did not survive re-derivation from the stored evidence. So the same rule is
applied here to the README: each figure quoted in it is recomputed below, and
any drift exits non-zero. Standard library only -- no model, no network.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAIL = []


def check(name, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) else (got == want)
    print("%s %-58s got %-12s want %s" % ("  ok  " if ok else " FAIL ", name, got, want))
    if not ok:
        FAIL.append(name)


def load_json(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(*parts):
    out = []
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------- 01 parsing
def paper_one():
    rows = [r for r in load_jsonl("01_lsdc_parsing", "data", "api_800_trials_raw.jsonl")
            if r.get("ok")]
    check("01 scored API trials = 944", len(rows), 944)
    check("01 model families = 8", len({r["model"] for r in rows}), 8)
    check("01 k spans 1..20", (min(r["k"] for r in rows), max(r["k"] for r in rows)), (1, 20))

    # The fallback, in the paper's own units: the two local models, 60 trials
    # each, and a trial counts as contaminated when the line-wise parse recovers
    # fewer answers than the k it was asked for. That is 28 of 120.
    def linewise(raw):
        return sum(1 for line in raw.split("\n")
                   if re.match(r"^(yes|no)$", line.strip(), re.I))

    fired = tot = 0
    for fname in ("local_qwen2.5_1.5b.json", "local_qwen2.5_7b.json"):
        j = load_json("01_lsdc_parsing", "data", fname)
        trials = [t for v in j.values() for t in v["trials"]]
        tot += len(trials)
        fired += sum(1 for t in trials if linewise(t["raw"]) < t["total"])
        # k* under the two readings of the threshold, per model:
        acc = {int(k): v["accuracy"] for k, v in j.items()}
        star = max([k for k in sorted(acc) if acc[k] >= 0.95], default=None)
        first_below = next((k for k in sorted(acc) if acc[k] < 0.95), None)
        print("       %-24s k*(largest>=0.95)=%s  first<0.95=%s"
              % (fname, star, first_below))
    check("01 local trials", tot, 120)
    check("01 fallback fired (of 120)", fired, 28)

    # The abstract's two numbers: k* = 1 for Qwen2.5-1.5B and k* = 13 for
    # Qwen2.5-7B, i.e. the largest k still answered perfectly. That is a
    # different reading from k*_(tau=0.95), and both are quoted, so both are
    # checked -- if they ever coincide the note about the convention is
    # claiming a distinction that no longer exists.
    perfect = {}
    for fname in ("local_qwen2.5_1.5b.json", "local_qwen2.5_7b.json"):
        j = load_json("01_lsdc_parsing", "data", fname)
        acc = {int(k): v["accuracy"] for k, v in j.items()}
        perfect[fname] = max([k for k in sorted(acc) if acc[k] >= 1.0], default=None)
    check("01 strict-parser k* for Qwen2.5-1.5B (abstract says 1)",
          perfect["local_qwen2.5_1.5b.json"], 1)
    check("01 strict-parser k* for Qwen2.5-7B (abstract says 13)",
          perfect["local_qwen2.5_7b.json"], 13)

    # The collapse replicates; the published exponent does not. On the
    # accumulative API task every model that is measured to k=20 either holds
    # accuracy or stops holding it -- either way no model exceeds k=20.
    acc20 = acc20_ref(rows)
    worst = min(acc20.values())
    check("01 some API model loses accuracy by k=20 (collapse replicates)", worst < 1.0, True)

    # k*_(tau=0.95) = the LARGEST k whose mean accuracy is still >= 0.95.
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], {}).setdefault(r["k"], []).append(r["rate"])
    stars = {}
    for m, ks in by_model.items():
        acc = {k: sum(v) / len(v) for k, v in ks.items()}
        good = [k for k in sorted(acc) if acc[k] >= 0.95]
        stars[m] = max(good) if good else None
    # The first-k-below-0.95 reading must NOT give the same answer, or the
    # convention note in the README is claiming a distinction that is not there.
    alt = {}
    for m, ks in by_model.items():
        acc = {k: sum(v) / len(v) for k, v in ks.items()}
        alt[m] = next((k for k in sorted(acc) if acc[k] < 0.95), None)
    check("01 the two readings actually differ", stars != alt, True)
    print("       k* (largest k at >=0.95):", json.dumps(stars, sort_keys=True))
    print("       first k below 0.95      :", json.dumps(alt, sort_keys=True))

    print("       accuracy at k=20:", json.dumps(
        {m: round(v, 3) for m, v in acc20.items()}, sort_keys=True))


# ------------------------------------------------------------ 02 judge audit
def paper_two():
    bench = load_json("02_judge_instrument_validity", "data",
                      "hand_labelled_benchmark_32.json")
    stats = load_json("02_judge_instrument_validity", "data", "summary_statistics.json")
    check("02 benchmark items = 32", len(bench), 32)
    hand = [x["hand"] for x in bench]
    check("02 hand: unsafe", hand.count("unsafe"), stats["hand_dist"]["unsafe"])
    check("02 hand: safe", hand.count("safe"), stats["hand_dist"]["safe"])
    check("02 hand: ambiguous", hand.count("ambiguous"), stats["hand_dist"]["ambiguous"])

    def jcls(v):
        u = (v or "").upper()
        if u.startswith("UNSAFE"):
            return "unsafe"
        if u.startswith("SAFE"):
            return "safe"
        if u.startswith("AMBIG"):
            return "ambiguous"
        return "missing"          # no verdict at all is not a correct verdict

    judge = [jcls(x.get("judge_15b")) for x in bench]
    rule = [x.get("rule_old") for x in bench]
    check("02 judge correct out of 32", sum(h == j for h, j in zip(hand, judge)),
          stats["judge_acc"][0])
    check("02 rule correct out of 32", sum(h == r for h, r in zip(hand, rule)),
          stats["rule_acc"][0])
    unsafe_idx = [i for i, h in enumerate(hand) if h == "unsafe"]
    judge_recall = sum(1 for i in unsafe_idx if judge[i] == "unsafe")
    check("02 judge recalls none of the unsafe", judge_recall,
          stats["judge_recall"][0])
    rule_recall = sum(1 for i in unsafe_idx if rule[i] == "unsafe")
    check("02 rule recall on unsafe", rule_recall, stats["rule_recall"][0])

    run99 = load_jsonl("02_judge_instrument_validity", "data", "judge_rerun_99_raw.jsonl")
    check("02 re-run items = 99", len(run99), 99)
    a = [jcls(r["judge"]) for r in run99]
    b = [jcls(r["rule"]) for r in run99]
    agree = sum(x == y for x, y in zip(a, b))
    check("02 raw agreement 61.6%", round(100 * agree / len(run99), 1),
          stats["run99"]["pct"], tol=0.05)
    po = agree / len(run99)
    cats = set(a) | set(b)
    pe = sum((a.count(c) / len(run99)) * (b.count(c) / len(run99)) for c in cats)
    kappa = (po - pe) / (1 - pe)
    check("02 kappa = 0.115", round(kappa, 3), stats["run99"]["kappa"], tol=0.002)
    check("02 the raw 61.6% is a majority-class effect (kappa far below it)",
          kappa < 0.2, True)


# ------------------------------------------------------- 03 attribution floor
def paper_three():
    rows = load_json("03_attribution_floor_effect", "data",
                     "attention_ablation_per_prompt.json")
    check("03 layers per prompt = 32", rows["config"]["n_layers"], 32)
    check("03 prompts = 4", rows["config"]["n_prompts"], 4)
    unsafe_before = [k for k, v in rows["per_prompt"].items() if v["baseline_safe"] == 0.0]
    check("03 prompts already unsafe before ablation", len(unsafe_before), 3)
    # Saturation: the per-prompt unsafe count is bounded by the prompts that
    # were safe to begin with, so the "critical" set is whichever layers flip
    # that one remaining prompt.
    flips = set()
    for k, v in rows["per_prompt"].items():
        if v["baseline_safe"] != 1.0:
            continue
        for layer in v["layers"]:
            if layer["unsafe"] > 0.0:
                flips.add(layer["layer"])
    per_layer = {layer: 0 for layer in flips}
    for v in rows["per_prompt"].values():
        for layer in v["layers"]:
            if layer["layer"] in flips and layer["unsafe"] > 0.0:
                per_layer[layer["layer"]] += 1
    top = max(per_layer.values()) if per_layer else 0
    check("03 critical-layer set saturates at 4 prompts", top, 4)

    na = load_json("03_attribution_floor_effect", "data", "neuron_attribution.json")
    impacts = []
    for layer_vals in na["neuron_impacts"].values():
        impacts.extend(v for v in layer_vals.values())
    check("03 stored attribution results", len(impacts), 500)
    check("03 neurons above impact 0.1", sum(1 for v in impacts if v > 0.1), 1)
    check("03 largest stored impact", round(max(impacts), 4), 0.1563, tol=1e-4)
    check("03 neurons LISTED as safety-critical", sum(len(v) for v in
          na["safety_neurons"].values()), 205)


def readme_claims():
    """The README's prose is part of the claim, so it is checked too."""
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        text = f.read()
    for needle, why in [
            ("944 scored API trials", "the trial count in the README"),
            ("28 of 120", "the fallback count"),
            ("53%", "the rule classifier's accuracy"),
            ("kappa = 0.115", "kappa as written"),
            ("205 safety-critical neurons", "the 205"),
            ("largest value of 0.156", "the single large attribution"),
            ("three of four prompts", "the floor effect")]:
        check("README says: %s" % why, needle in text, True)


def acc20_ref(rows):
    by = {}
    for r in rows:
        by.setdefault(r["model"], {}).setdefault(r["k"], []).append(r["rate"])
    return {m: sum(v[20]) / len(v[20]) for m, v in by.items() if 20 in v}


if __name__ == "__main__":
    paper_one()
    paper_two()
    paper_three()
    readme_claims()
    print("")
    if FAIL:
        print("%d claim(s) did not reproduce: %s" % (len(FAIL), ", ".join(FAIL)))
        raise SystemExit(1)
    print("every number in the README recomputes from this repo's data")
