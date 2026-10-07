"""
Module 5 - risk and priority: a weighted formula, not a trained model.

    risk = sum_k w_k * f_k / sum_k w_k        (over the factors that are enabled)

Every factor f_k is scaled to [0, 1]:

  size       people in the group / `size_full`            more lives at one stop
  severity   flooded share around the group (Model 2)     how much water is around them
  stranded   seconds not moving / `stranded_full_s`       stuck, not walking out (memory)
  isolation  water between them and dry land (Model 2)  how hard they are to reach
  trend      rise in severity per minute / `trend_full`   water rising around them (memory)

Levels follow the project document: Critical > 0.8, High 0.5-0.8,
Medium 0.3-0.5, Low < 0.3.

Ablations (the planned "with vs without" comparisons) switch a module off
and re-normalise the remaining weights:
  no memory    -> stranded and trend dropped
  no severity  -> severity, isolation and trend dropped (all come from Model 2)

The weights are a design choice, stated as such - there is no ground truth
to fit them to. They put the water first (severity + trend + isolation = 0.55)
because that is what turns a group of people into an emergency.
"""

WEIGHTS = dict(size=0.20, severity=0.30, stranded=0.25, isolation=0.10, trend=0.15)
SCALE = dict(size_full=10, stranded_full_s=300.0, trend_full=0.10)
LEVELS = [(0.8, "Critical"), (0.5, "High"), (0.3, "Medium"), (0.0, "Low")]
MEMORY_FACTORS = {"stranded", "trend"}
SEVERITY_FACTORS = {"severity", "isolation", "trend"}


def factors(count, severity=0.0, stranded_s=0.0, isolation=0.0, trend_per_min=0.0, scale=SCALE):
    clip = lambda v: float(min(1.0, max(0.0, v)))
    return dict(size=clip(count / scale["size_full"]),
                severity=clip(severity),
                stranded=clip(stranded_s / scale["stranded_full_s"]),
                isolation=clip(isolation),
                trend=clip(trend_per_min / scale["trend_full"]))


def score(f, use_memory=True, use_severity=True, weights=WEIGHTS):
    off = (set() if use_memory else MEMORY_FACTORS) | (set() if use_severity else SEVERITY_FACTORS)
    w = {k: v for k, v in weights.items() if k not in off}
    total = sum(w.values())
    return sum(w[k] * f[k] for k in w) / total if total else 0.0


def level(risk):
    for th, name in LEVELS:
        if risk > th or th == 0.0:
            return name
