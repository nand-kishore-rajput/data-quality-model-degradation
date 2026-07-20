"""
Diagnostic script: is -9 (HELOC) / -1 (bank-marketing V14) a synchronized
sentinel code, or just an independently-occurring negative value?

Usage:
    python inspect_sentinels.py --heloc /path/to/heloc.csv --bank /path/to/bank-marketing.csv

If you don't have local CSVs, this will try openml.org directly
(requires `pip install openml --break-system-packages`).
    - HELOC: OpenML dataset id 46932 (per project's canonical source decision)
    - bank-marketing: OpenML dataset id 1461 (adjust if yours differs)

What it checks and why it matters for the no_negative vs none decision:

1. CO-OCCURRENCE: FICO's documented codes distinguish "no bureau record at all"
   (-9, which should hit EVERY feature in that row simultaneously) from
   "no usable trade of this specific type" (-8/-7, which would hit only
   the features that depend on that specific trade type). If -9 rows are
   synchronized across most/all columns, that's strong evidence it's a
   deliberate record-level sentinel, not scattered noise -> supports
   treating the *domain* constraint as no_negative (i.e. -9 is a known,
   removable defect, not a real observation).

2. ISOLATION FROM THE REST OF THE DISTRIBUTION: if -9 sits far below the
   next-smallest real value (e.g. real values are 0-100, and the only
   negative value is exactly -9, appearing nowhere else in that shape),
   that's a classic sentinel signature vs. a naturally continuous negative
   tail.

3. For bank-marketing V14 (pdays): pdays=-1 should almost perfectly line up
   with V15 (previous contacts) == 0, i.e. "never contacted before" logically
   implies "days since last contact" is undefined. If that logical
   consistency holds near-perfectly, it's a sentinel, not a real observation.
"""
import argparse
import pandas as pd

HELOC_COLS = [
    'ExternalRiskEstimate','MSinceOldestTradeOpen','MSinceMostRecentTradeOpen','AverageMInFile',
    'NumSatisfactoryTrades','NumTrades60Ever2DerogPubRec','NumTrades90Ever2DerogPubRec',
    'PercentTradesNeverDelq','MSinceMostRecentDelq','MaxDelq2PublicRecLast12M','MaxDelqEver',
    'NumTotalTrades','NumTradesOpeninLast12M','PercentInstallTrades','MSinceMostRecentInqexcl7days',
    'NumInqLast6M','NumInqLast6Mexcl7days','NetFractionRevolvingBurden','NetFractionInstallBurden',
    'NumRevolvingTradesWBalance','NumInstallTradesWBalance','NumBank2NatlTradesWHighUtilization',
    'PercentTradesWBalance'
]

def _read_any(path):
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def load_heloc(path):
    if path:
        return _read_any(path)
    import openml
    ds = openml.datasets.get_dataset(46932)
    X, *_ = ds.get_data()
    return X

def load_bank(path):
    if path:
        return _read_any(path)
    import openml
    ds = openml.datasets.get_dataset(1461)
    X, *_ = ds.get_data()
    return X

def inspect_heloc(df):
    print("="*70)
    print("HELOC: -9 co-occurrence and distribution-isolation check")
    print("="*70)
    cols = [c for c in HELOC_COLS if c in df.columns]
    missing = [c for c in HELOC_COLS if c not in df.columns]
    if missing:
        print("WARNING - columns not found in file:", missing)

    n = len(df)
    print(f"\nTotal rows: {n}\n")

    # Per-column -9 rate
    print("--- Per-column -9 rate ---")
    rate = {}
    for c in cols:
        cnt = (df[c] == -9).sum()
        rate[c] = cnt / n
        print(f"{c:40s} -9 count={cnt:6d}  ({cnt/n:6.2%})")

    # Co-occurrence: for rows where ExternalRiskEstimate == -9,
    # what fraction of the OTHER columns are also -9 in that same row?
    anchor = 'ExternalRiskEstimate'
    if anchor in cols:
        anchor_mask = df[anchor] == -9
        print(f"\n--- Rows where {anchor} == -9: n={anchor_mask.sum()} ---")
        if anchor_mask.sum() > 0:
            sub = df.loc[anchor_mask, [c for c in cols if c != anchor]]
            frac_also_sentinel = (sub == -9).mean().mean()
            print(f"Average fraction of OTHER columns also == -9 in those rows: {frac_also_sentinel:.2%}")
            print("(close to 100% => record-level sentinel / synchronized 'no bureau record'")
            print(" close to individual per-column rates above => scattered, not synchronized)")

    # Distribution isolation: gap between -9 and the next smallest real value
    print("\n--- Distribution gap check (is -9 isolated from real values?) ---")
    for c in cols:
        vals = df[c]
        real_vals = vals[vals != -9]
        if len(real_vals) == 0:
            continue
        next_min = real_vals.min()
        print(f"{c:40s} next-smallest non(-9) value = {next_min}")

def inspect_bank(df):
    print("\n" + "="*70)
    print("bank-marketing: V14 (pdays) == -1 vs V15 (previous) == 0 consistency")
    print("="*70)
    # your processed file may have renamed V14/V15 to their semantic names
    # (pdays / previous) rather than the raw OpenML V-codes -- try both.
    col14 = 'V14' if 'V14' in df.columns else ('pdays' if 'pdays' in df.columns else None)
    col15 = 'V15' if 'V15' in df.columns else ('previous' if 'previous' in df.columns else None)
    if col14 is None or col15 is None:
        print("WARNING: expected columns V14/V15 (or pdays/previous) not found; found:", list(df.columns))
        return
    print(f"(using columns: pdays-equivalent='{col14}', previous-equivalent='{col15}')")
    n = len(df)
    v14_neg1 = df[col14] == -1
    v15_zero = df[col15] == 0
    both = (v14_neg1 & v15_zero).sum()
    print(f"n={n}")
    print(f"V14==-1 count: {v14_neg1.sum()} ({v14_neg1.mean():.2%})")
    print(f"V15==0  count: {v15_zero.sum()} ({v15_zero.mean():.2%})")
    print(f"V14==-1 AND V15==0: {both} "
          f"({both / max(v14_neg1.sum(),1):.2%} of V14==-1 rows also have V15==0)")
    print("(close to 100% => -1 is a logical 'never contacted' sentinel tied to V15,")
    print(" not an independent negative observation)")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--heloc', default=None, help='path to local HELOC csv (optional, else tries OpenML)')
    ap.add_argument('--bank', default=None, help='path to local bank-marketing csv (optional, else tries OpenML)')
    args = ap.parse_args()

    heloc_df = load_heloc(args.heloc)
    inspect_heloc(heloc_df)

    bank_df = load_bank(args.bank)
    inspect_bank(bank_df)