"""Benchmark scoring for AdaGround predictions.

ScreenSpot-Pro, ScreenSpot-v2 and UI-Vision: a point is correct iff it lies inside the target box [x1, y1, x2, y2]
(boundaries included); an abstention or a missing point is wrong.
OSWorld-G: the official rule (xlang-ai/OSWorld-G, evaluation/eval.py). Box targets [x, y, w, h] and polygon targets are
point-in-region; an instruction without a target (box_type "refusal") is correct only when the method abstains; an
abstention on a present target and a missing point are wrong.

Labels are read only here: the target boxes and ScreenSpot categories from the "gt" and "group" fields of c4x.json, the
OSWorld-G targets from the refined instruction file (OSWorld-G_refined.json) and its categories from the official
classification file (classification_result.json).
"""
import json
import os
from fractions import Fraction

ABSTAIN = "ABSTAIN"

# OSWorld-G categories of the official classification file (overlapping subsets) and their display names
OSWORLDG_CATEGORIES = (("text_matching", "Text"), ("element_recognition", "Element"), ("layout_understanding", "Layout"),
                       ("fine_grained_manipulation", "Fine-grained"), ("refusal", "Refusal"))
SSPRO_CATEGORIES = ("Dev", "Creative", "CAD", "Scientific", "Office", "OS")
UIVISION_TASKS = (("basic", "Basic"), ("functional", "Functional"), ("spatial", "Spatial"))


def in_box(pt, box):
    """box = [x1, y1, x2, y2]."""
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def in_polygon(pt, coords):
    """even-odd ray casting; coords = [x1, y1, x2, y2, ...]."""
    poly = list(zip(coords[0::2], coords[1::2]))
    x, y, inside = pt[0], pt[1], False
    for k in range(len(poly)):
        (x1, y1), (x2, y2) = poly[k], poly[k - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def as_answer(final, abstain=False):
    """the answer a scorer sees: ABSTAIN, None (no point) or an (x, y) tuple."""
    if abstain:
        return ABSTAIN
    return tuple(final) if final else None


class Scorer:
    """ok(sample_id, answer) -> bool for one benchmark.

    benchmark: sspro | ssv2 | uivision | osworldg. run_dirs: the directories holding c4x.json (its "gt" field is the target
    box on the point-in-box benchmarks). osworldg_labels: OSWorld-G_refined.json. osworldg_classes: the official
    classification file (optional; needed only for the OSWorld-G categories)."""

    def __init__(self, benchmark, run_dirs, osworldg_labels=None, osworldg_classes=None):
        self.benchmark = benchmark
        self.gt, self.group, self.raw, self.classes = {}, {}, None, None
        if benchmark == "osworldg":
            if not osworldg_labels:
                raise SystemExit("OSWorld-G needs --osworldg_labels (OSWorld-G_refined.json)")
            self.raw = {r["id"]: r for r in json.load(open(osworldg_labels))}
            if osworldg_classes and os.path.exists(osworldg_classes):
                cls = json.load(open(osworldg_classes))["classified"]
                self.classes = {c: {x["id"] if isinstance(x, dict) else x for x in cls[c]} for c, _ in OSWORLDG_CATEGORIES}
        else:
            for d in run_dirs:
                for r in json.load(open(os.path.join(d, "c4x.json")))["details"]:
                    self.gt[r["id"]] = r["gt"]
                    self.group[r["id"]] = r.get("group")

    def ok(self, sample_id, ans):
        if self.raw is None:
            if ans is None or ans == ABSTAIN:
                return False
            return in_box(ans, self.gt[sample_id])
        r = self.raw[sample_id]
        if r["box_type"] == "refusal":
            return ans == ABSTAIN
        if ans is None or ans == ABSTAIN:
            return False
        c = r["box_coordinates"]
        if r["box_type"] == "bbox":
            return c[0] <= ans[0] <= c[0] + c[2] and c[1] <= ans[1] <= c[1] + c[3]
        return in_polygon(ans, c)

    def absent(self, ids):
        """OSWorld-G instructions without a target (empty elsewhere)."""
        if self.raw is None:
            return set()
        return {i for i in ids if self.raw[i]["box_type"] == "refusal"}

    def categories(self, ids):
        """[(name, ids of the category)] in reporting order; empty when the benchmark has no categories here."""
        ids = list(ids)
        if self.benchmark == "osworldg":
            if self.classes is None:
                return []
            return [(name, [i for i in ids if i in self.classes[c]]) for c, name in OSWORLDG_CATEGORIES]
        if self.benchmark == "uivision":
            return [(name, [i for i in ids if i.split("/", 1)[0] == t]) for t, name in UIVISION_TASKS]
        if not all(self.group.get(i) for i in ids):
            return []
        names = SSPRO_CATEGORIES if self.benchmark == "sspro" else sorted({self.group[i] for i in ids})
        return [(g, [i for i in ids if self.group[i] == g]) for g in names]


def ratio(c, n):
    """100 * c / n as an exact fraction (None when n = 0)."""
    return Fraction(100 * c, n) if n else None


def fmt1(x):
    """an exact percentage rounded once, half up, to one decimal ('--' for None)."""
    if x is None:
        return "--"
    v = Fraction(x) * 10
    q = (v.numerator * 2 + v.denominator) // (2 * v.denominator)
    return f"{'-' if q < 0 else ''}{abs(q) // 10}.{abs(q) % 10}"
