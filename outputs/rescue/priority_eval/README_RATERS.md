# Rater instructions (about 15 minutes)

You are checking whether our system ranks rescue groups the way a person would.
**Do this on your own. Don't discuss your answers with the other raters until everyone is done.**

## What to do

1. Open the `sheet/` folder. Each of the 40 photos has white boxes with numbers (1, 2, 3, …).
   Each box is a group of people the system found.
2. For each photo, ask yourself: **"I have one rescue boat. Which numbered group do I go to FIRST?"**
3. Open `labels.csv` in Excel. Find the photo's row (the `photo` column is the file name without `.jpg`).
4. Type that number in **your** column:
   - Teammate 1 → `rater1`
   - Teammate 2 → `rater2`
   - Teammate 3 → `rater3`
5. Save as **CSV** (not .xlsx), keeping the same file name.

## Rules

- Judge only what you can see: how many people, how much water is around them, and whether
  they can get to dry land on their own.
- The numbers are in **random order**. Number 1 is not "most important".
- If a box contains **no real person** (a rock, a shadow, a helicopter), don't pick it.
- If **no box** on the photo contains a real person, write **0**.
- Don't overthink it: 10–20 seconds per photo, first instinct.

## When all three columns are filled

```
python scripts/eval_priority.py score
```

It writes `priority_eval.json`. This contains:

| Field | Meaning |
|---|---|
| `full` | how often our system's top group matches the raters' majority choice |
| `no_severity` | the same, ranking by people count only (Model 2 switched off) |
| `random_top1` | what a random pick would score |
| `rater_pairwise_agreement` | how often two raters agree with each other: the best any method can realistically reach |
