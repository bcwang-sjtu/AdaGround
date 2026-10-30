"""Compose AdaGround's final answers from its stage outputs and score them.

Each run directory holds the stage outputs of one backbone on (a part of) one benchmark:
  c4x.json             fixed chain c_4x: single pass on the whole screen (p1, tok1), then a 1288x728 window around that
                       answer at 4x area (final, vis_tokens)
  exist.json           existence query e(S, I) on the whole screen (p_exist, img_tokens); OSWorld-G only
  csg_cell.json        instruction program; for cell(c, r) the column- and row-header points and the header check
  tnz_t3.json          TNZ chain c_TNZ (tau = 3)
  arb_t3.json          verifier arbitration of c_TNZ and c_4x
  rg_t3_tau0.99.json   negative-feedback re-grounding of the arbitrated answer (theta_r = 0.99); verifier scores v(p),
                       v(p'') in its "retry" record
Arbitration and re-grounding are applied by their stages. compose() applies the rest of the method per sample:
    p <- re-grounded answer
    if the program is cell(c, r) and the header check holds (x_col > x_row and y_col < y_row): p <- (x_col, y_row)
    on OSWorld-G, unless p was composed by CSG: abstain iff (max(v(p), v(p'')) + e(S, I)) / 2 < theta_e = 0.03
Visual tokens per sample: the re-grounding file (both chains, arbitration, v(p) and the re-grounding passes when
triggered) + the header passes of a cell program (csg_cell - c4x) + the existence query on OSWorld-G when the answer is
not composed. Abstention applies only to OSWorld-G, whose protocol contains instructions without a target.

Several run directories (e.g. shards of one benchmark) are merged per sample before composition and scored once.

usage:
  python -m adaground.compose --run_dir OUT [--run_dir OUT2 ...] --benchmark {sspro,osworldg,ssv2,uivision}
         [--osworldg_labels OSWorld-G_refined.json] [--osworldg_classes classification_result.json] [--out pred.json]
"""
import argparse
import json
import os
import sys
from fractions import Fraction

from .scoring import Scorer, as_answer, fmt1, ratio

BENCHMARKS = ("sspro", "osworldg", "ssv2", "uivision")
TAU = 3                                  # TNZ target span (tokens)
THETA_R = 0.99                           # re-grounding threshold on v(p)
THETA_E = 0.03                           # abstention threshold of the joint rule
ABSTENTION_BENCHMARKS = ("osworldg",)
STAGE_FILES = {"c4x": "c4x.json", "exist": "exist.json", "csg": "csg_cell.json", "tnz": f"tnz_t{TAU}.json",
               "arb": f"arb_t{TAU}.json", "rg": f"rg_t{TAU}_tau{float(THETA_R)}.json"}
# fields of a stage record that composition may read; labels (gt, correct, refusal, ...) are dropped on loading
PRED_FIELDS = ("id", "final", "vis_tokens", "choose", "retry", "program", "p1", "tok1", "p_exist", "img_tokens")


def fail(msg):
    sys.exit(f"error: {msg}")


# ----------------------------------------------------------------------------- loading
def load_stage(path):
    """(benchmark named in the summary or None, {id: record projected onto PRED_FIELDS})."""
    J = json.load(open(path))
    recs = {}
    for d in J["details"]:
        if d["id"] in recs:
            fail(f"{path}: duplicate id {d['id']}")
        recs[d["id"]] = {k: d[k] for k in PRED_FIELDS if k in d}
    return J.get("summary", {}).get("dataset"), recs


def load_runs(run_dirs, benchmark):
    """stage records of every run directory merged per sample: ({stage: {id: record}}, sorted ids)."""
    stages = [s for s in STAGE_FILES if s != "exist" or benchmark in ABSTENTION_BENCHMARKS]
    S = {s: {} for s in stages}
    for d in run_dirs:
        for s in stages:
            path = os.path.join(d, STAGE_FILES[s])
            if not os.path.exists(path):
                fail(f"missing {path}")
            ds, recs = load_stage(path)
            if ds is not None and ds != benchmark:
                fail(f"{path} was produced for {ds}, not {benchmark}")
            overlap = set(recs) & set(S[s])
            if overlap:
                fail(f"{path}: {len(overlap)} ids also appear in another run directory")
            S[s].update(recs)
    ids = sorted(S["c4x"])
    for s in stages:
        if set(S[s]) != set(ids):
            fail(f"{STAGE_FILES[s]} covers {len(S[s])} samples, c4x.json {len(ids)}: the stages differ in their ids")
    return S, ids


# ----------------------------------------------------------------------------- composition (no labels)
def verifier_score(v_old, v_new):
    """max(v(p), v(p'')) of the re-grounding record; a missing v(p) (no answer to verify) counts as 1.0, a missing
    v(p'') (re-grounding not triggered or without a point) as 0.0."""
    return max(1.0 if v_old is None else v_old, 0.0 if v_new is None else v_new)


def same_point(a, b, tol=0.051):
    if a is None or b is None:
        return a is None and b is None
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def compose(S, ids, benchmark):
    """per sample: final answer (None when abstaining or without a point), abstention, CSG composition and tokens."""
    gate = benchmark in ABSTENTION_BENCHMARKS
    out = {}
    for i in ids:
        rg, arb, c4x, csg = S["rg"][i], S["arb"][i], S["c4x"][i], S["csg"][i]
        retry = rg.get("retry") or {}
        v_old, v_new = retry.get("p_old"), retry.get("p_new")
        p, tok, composed = rg["final"], rg["vis_tokens"], False
        prog = csg["program"]
        cell = prog["prog"]["op"] == "cell"
        if cell:
            tok += csg["vis_tokens"] - c4x["vis_tokens"]          # header passes
            ex = prog.get("exec") or {}
            if ex.get("geo_ok"):
                p, composed = [round(ex["col_pt"][0], 1), round(ex["row_pt"][1], 1)], True
        p_csg, tok_csg = p, tok
        signal, abstain = None, False
        if gate and not composed:
            e = S["exist"][i]
            signal = (verifier_score(v_old, v_new) + e["p_exist"]) / 2
            tok += e["img_tokens"]
            abstain = signal < THETA_E
        out[i] = dict(id=i, final=None if abstain else p, abstain=abstain, composed=composed, vis_tokens=tok,
                      gate_signal=signal, csg_final=p_csg, csg_tokens=tok_csg, cell_program=cell,
                      arbitrated=arb.get("choose") is not None, rg_triggered=bool(retry.get("triggered")),
                      rg_changed=not same_point(rg["final"], arb["final"]))
    return out


# ----------------------------------------------------------------------------- scoring
def method_rows(S, comp, ids, benchmark):
    """{row name: {id: (answer, visual tokens)}} in reporting order: baselines, then the cumulative component ladder."""
    gate = benchmark in ABSTENTION_BENCHMARKS
    R = {"single pass": {i: (as_answer(S["c4x"][i].get("p1")), S["c4x"][i].get("tok1") or 0) for i in ids},
         "fixed 4x crop": {i: (as_answer(S["c4x"][i]["final"]), S["c4x"][i]["vis_tokens"]) for i in ids},
         "c_TNZ": {i: (as_answer(S["tnz"][i]["final"]), S["tnz"][i]["vis_tokens"]) for i in ids},
         "+arbitration": {i: (as_answer(S["arb"][i]["final"]), S["arb"][i]["vis_tokens"]) for i in ids},
         "+re-grounding": {i: (as_answer(S["rg"][i]["final"]), S["rg"][i]["vis_tokens"]) for i in ids},
         "+CSG": {i: (as_answer(comp[i]["csg_final"]), comp[i]["csg_tokens"]) for i in ids}}
    if gate:
        R["+abstention"] = {i: (as_answer(comp[i]["final"], comp[i]["abstain"]), comp[i]["vis_tokens"]) for i in ids}
    return R, ("+abstention" if gate else "+CSG")


def evaluate(S, comp, ids, benchmark, scorer):
    R, final_row = method_rows(S, comp, ids, benchmark)
    ok = {name: {i: bool(scorer.ok(i, a)) for i, (a, _) in r.items()} for name, r in R.items()}
    absent = scorer.absent(ids)
    present = [i for i in ids if i not in absent]
    cats = scorer.categories(ids)
    rows = {}
    for name, r in R.items():
        c = sum(ok[name][i] for i in ids)
        row = dict(correct=c, n=len(ids), acc=ratio(c, len(ids)), tokens=round(sum(t for _, t in r.values()) / len(ids), 1),
                   categories={cname: (sum(ok[name][i] for i in cids), len(cids)) for cname, cids in cats})
        if absent:
            row["present"] = (sum(ok[name][i] for i in present), len(present))
        rows[name] = row
    ab = [i for i in ids if comp[i]["abstain"]]
    stats = dict(abstain=len(ab), abstain_on_absent=sum(i in absent for i in ab),
                 abstain_on_present=sum(i not in absent for i in ab), absent_targets=len(absent),
                 cell_programs=sum(comp[i]["cell_program"] for i in ids), composed=sum(comp[i]["composed"] for i in ids),
                 composed_correct=sum(ok[final_row][i] for i in ids if comp[i]["composed"]),
                 arbitrations=sum(comp[i]["arbitrated"] for i in ids), rg_triggered=sum(comp[i]["rg_triggered"] for i in ids),
                 rg_changed=sum(comp[i]["rg_changed"] for i in ids),
                 no_answer=sum(comp[i]["final"] is None and not comp[i]["abstain"] for i in ids))
    return dict(rows=rows, final_row=final_row, ok=ok, stats=stats)


def task_mean(row):
    """UI-Vision: unweighted mean of the per-task accuracies (exact)."""
    accs = [ratio(c, n) for c, n in row["categories"].values()]
    return sum(accs, Fraction(0)) / len(accs) if accs and None not in accs else None


# ----------------------------------------------------------------------------- report
def show(res, benchmark, n_dirs):
    rows, final_row = res["rows"], res["final_row"]
    gate = benchmark in ABSTENTION_BENCHMARKS
    n = next(iter(rows.values()))["n"]
    print(f"AdaGround on {benchmark}: {n} samples from {n_dirs} run director{'y' if n_dirs == 1 else 'ies'} "
          f"(tau={TAU}, theta_r={THETA_R}, " + (f"abstention: joint rule, theta_e={THETA_E})" if gate else "no abstention)"))
    has_present = any("present" in r for r in rows.values())
    head = f"  {'':<28} {'acc':>6} {'correct/n':>11} {'tokens/sample':>14}" + ("   present targets" if has_present else "")
    labels = {"single pass": "single pass (c4x pass 1)", "fixed 4x crop": "fixed 4x crop (c4x final)"}
    print(head)
    for name, r in rows.items():
        if name == "c_TNZ":
            print("  component ladder")
        label = labels.get(name, name) + (" (= AdaGround)" if name == final_row else "")
        pr = ""
        if "present" in r:
            pc, pn = r["present"]
            pr = f"   {fmt1(ratio(pc, pn)):>6} ({pc}/{pn})"
        cn = f"{r['correct']}/{r['n']}"
        print(f"  {label:<28} {fmt1(r['acc']):>6} {cn:>11} {r['tokens']:>14.1f}{pr}")
    cats = list(next(iter(rows.values()))["categories"])
    if cats:
        extra = ["Avg"] if benchmark == "uivision" else []
        print("  accuracy per category (%)" + ("; Avg = mean of the three tasks" if benchmark == "uivision" else ""))
        w = max(8, max(len(c) for c in cats) + 1)
        print(f"  {'':<28} " + " ".join(f"{c:>{w}}" for c in cats + extra) + f" {'Overall':>{w}}")
        for name, r in rows.items():
            vals = [fmt1(ratio(c, k)) for c, k in r["categories"].values()]
            if benchmark == "uivision":
                vals.append(fmt1(task_mean(r)))
            label = labels.get(name, name) + (" (= AdaGround)" if name == final_row else "")
            print(f"  {label:<28} " + " ".join(f"{v:>{w}}" for v in vals) + f" {fmt1(r['acc']):>{w}}")
        sizes = ", ".join(f"{c} {k}" for c, (_, k) in next(iter(rows.values()))["categories"].items())
        print(f"  category sizes: {sizes}")
    elif benchmark == "osworldg":
        print("  (OSWorld-G categories: pass --osworldg_classes classification_result.json)")
    s = res["stats"]
    line = (f"  cell programs {s['cell_programs']}, composed by CSG {s['composed']} (correct {s['composed_correct']}); "
            f"arbitrations {s['arbitrations']}; re-grounding triggered {s['rg_triggered']}, changed {s['rg_changed']}; "
            f"no answer {s['no_answer']}")
    if gate:
        line = (f"  abstentions {s['abstain']} (on absent targets {s['abstain_on_absent']} of {s['absent_targets']}, "
                f"on present targets {s['abstain_on_present']});\n" + line)
    print(line)


def write_predictions(path, comp, ids, benchmark):
    gate = benchmark in ABSTENTION_BENCHMARKS
    details = [{k: comp[i][k] for k in ("id", "final", "abstain", "composed", "vis_tokens", "gate_signal")} for i in ids]
    out = dict(benchmark=benchmark, n=len(ids), tau=TAU, theta_r=THETA_R,
               abstention=dict(enabled=gate, rule="(max(v_old, v_new) + p_exist) / 2 < theta_e, not on a CSG-composed answer",
                               theta_e=THETA_E),
               details=details)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", action="append", required=True, help="directory with the stage outputs; repeat to merge shards")
    ap.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    ap.add_argument("--osworldg_labels", default="", help="OSWorld-G: OSWorld-G_refined.json (target boxes, refusal instructions)")
    ap.add_argument("--osworldg_classes", default="", help="OSWorld-G: official classification_result.json for the per-category "
                    "accuracy (default: classification_result.json next to --osworldg_labels, if present)")
    ap.add_argument("--out", default="", help="write the per-sample final predictions here (JSON)")
    a = ap.parse_args(argv)
    dirs = [d.rstrip("/") for d in a.run_dir]
    if len(set(dirs)) != len(dirs):
        fail("a run directory is given twice")
    S, ids = load_runs(dirs, a.benchmark)
    comp = compose(S, ids, a.benchmark)
    classes = a.osworldg_classes
    if a.benchmark == "osworldg" and not classes and a.osworldg_labels:
        classes = os.path.join(os.path.dirname(os.path.abspath(a.osworldg_labels)), "classification_result.json")
    scorer = Scorer(a.benchmark, dirs, a.osworldg_labels or None, classes or None)
    missing = [i for i in ids if (scorer.raw is not None and i not in scorer.raw) or (scorer.raw is None and i not in scorer.gt)]
    if missing:
        fail(f"{len(missing)} samples have no label (e.g. {missing[0]})")
    res = evaluate(S, comp, ids, a.benchmark, scorer)
    show(res, a.benchmark, len(dirs))
    if a.out:
        write_predictions(a.out, comp, ids, a.benchmark)
        print(f"  predictions -> {a.out}")
    return res, comp


if __name__ == "__main__":
    main()
