"""
run_complete_system.py
=======================
Master entry point for the complete pharmacy optimization system:

  Part 2:  ZCTA-level profit scoring    (profit_model_v2.py)
     ↓
  Part 2b: ML enhancement layer          (profit_ml_layer.py)
     ↓
  Part 3:  Walgreens store optimization   (walgreens_optimizer_v2.py)

Usage:
  # Full pipeline
  python run_complete_system.py --npi data/pharmacy_data.csv

  # Full pipeline with LLM briefs
  python run_complete_system.py --npi data/pharmacy_data.csv --api-key sk-ant-xxx

  # Part 3 only (requires existing Part 2 results)
  python run_complete_system.py --npi data/pharmacy_data.csv --walgreens-only

  # Skip ML layers for faster iteration
  python run_complete_system.py --npi data/pharmacy_data.csv --skip-ml

Outputs:
  results_v2/                    ← Part 2 ZCTA scores + ML enhancements
  results_walgreens/             ← Part 3 store-level Walgreens analysis
"""

import argparse
import sys
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Complete Pharmacy Optimization System")
    parser.add_argument("--npi", default="data/pharmacy_data.csv",
                        help="Path to pharmacy NPI data CSV (buddy's export with one-hot chain columns)")
    parser.add_argument("--out-zcta", default="results_v2")
    parser.add_argument("--out-walgreens", default="results_walgreens")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-pop", type=int, default=500)
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key for LLM-generated store briefs (optional)")
    parser.add_argument("--wag-stores", default=None,
                        help="Optional CSV of Walgreens stores with store_id and ZIP/ZCTA (enables true store-level)")
    parser.add_argument("--n-briefs", type=int, default=50)
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--skip-ml", action="store_true",
                        help="Skip Part 2b ML enhancement layer")
    parser.add_argument("--walgreens-only", action="store_true",
                        help="Skip Part 2, run only Part 3 (requires existing ZCTA scores)")
    args = parser.parse_args()

    # Validate NPI data exists
    npi_path = Path(args.npi)
    if not npi_path.exists():
        print(f"ERROR: NPI data not found at {npi_path}")
        print(f"  Expected: buddy's pharmacy export with one-hot chain columns")
        print(f"  Run:  python run_complete_system.py --npi <path_to_pharmacy_data.csv>")
        sys.exit(1)

    # ── Part 2: ZCTA Profit Scoring ──
    if not args.walgreens_only:
        print("\n" + "█" * 70)
        print("  PART 2: ZCTA-LEVEL PROFIT SCORING")
        print("█" * 70 + "\n")

        from profit_model_v2 import run_pipeline
        config = {
            "out_dir": args.out_zcta,
            "top_k": args.top_k,
            "min_pop": args.min_pop,
            "pharmacy_csv_candidates": [
                args.npi,
                "data/pharmacy_data.csv",
                "data/pharmacy.csv",
                "data/pharmacies.csv",
                "data/npi_tagged.csv",
            ],
        }
        df, result, summary = run_pipeline(config=config)

        # ── Part 2b: ML Enhancement ──
        if not args.skip_ml:
            print("\n" + "█" * 70)
            print("  PART 2b: ML ENHANCEMENT LAYER")
            print("█" * 70 + "\n")

            try:
                from profit_ml_layer import enhance_scores
                df_enhanced, ml_artifacts = enhance_scores(
                    df, result,
                    chronic_cols=summary.get("chronic_disease_cols_used", []),
                    api_key=args.api_key,
                    n_briefs=args.n_briefs,
                    output_dir=args.out_zcta,
                )
            except ImportError:
                print("  profit_ml_layer.py not found — skipping ML enhancement")
            except Exception as e:
                print(f"  ML enhancement failed: {e} — continuing with base scores")
    else:
        # Verify ZCTA scores exist
        zcta_path = Path(args.out_zcta) / "profit_scores.csv"
        if not zcta_path.exists():
            print(f"ERROR: {zcta_path} not found. Run Part 2 first (remove --walgreens-only).")
            sys.exit(1)

    # ── Part 3: Walgreens Distribution Optimization ──
    print("\n" + "█" * 70)
    print("  PART 3: WALGREENS DISTRIBUTION OPTIMIZATION")
    print("█" * 70 + "\n")

    # Load NPI data
    npi_df = pd.read_csv(args.npi, dtype=str, low_memory=False)

    from walgreens_optimizer_v2 import run_walgreens_pipeline

    wag_results = run_walgreens_pipeline(
        zcta_scores_path=f"{args.out_zcta}/profit_scores.csv",
        npi_data=npi_df,
        output_dir=args.out_walgreens,
        n_clusters=args.n_clusters,
        n_briefs=args.n_briefs,
        api_key=args.api_key,
        walgreens_stores=args.wag_stores,
    )

    # ── Final Summary ──
    print("\n" + "=" * 70)
    print("  COMPLETE SYSTEM SUMMARY")
    print("=" * 70)

    stores = wag_results["stores"]
    summary = wag_results["summary"]

    print(f"\n  ZCTA scores:           {args.out_zcta}/profit_scores.csv")
    print(f"  Store scores:          {args.out_walgreens}/store_viability_scores.csv")
    print(f"  Consolidation pairs:   {args.out_walgreens}/consolidation_pairs.csv")
    print(f"  Archetype centroids:   {args.out_walgreens}/archetype_centroids.csv")
    print(f"  Store briefs:          {args.out_walgreens}/store_briefs.csv")
    print(f"  Portfolio summary:     {args.out_walgreens}/portfolio_summary.json")

    print(f"\n  Total Walgreens analyzed:  {len(stores):,}")
    print(f"  ├─ Protect & Invest:       {summary.get('protect_invest', 0):,}")
    print(f"  ├─ Monitor:                {summary.get('monitor', stores[stores['action']=='MONITOR'].shape[0] if 'action' in stores.columns else 0):,}")
    print(f"  ├─ Consolidate:            {summary.get('consolidation_pairs', 0):,}")
    print(f"  ├─ Closure Candidates:     {summary.get('closure_candidates', 0):,}")
    print(f"  ├─ Hidden Gems (ML):       {summary.get('anomaly_hidden_gems', 0):,}")
    print(f"  └─ Hidden Risks (ML):      {summary.get('anomaly_hidden_risks', 0):,}")

    # Archetype breakdown
    if "archetype_name" in stores.columns:
        print(f"\n  Archetypes:")
        for name, count in stores["archetype_name"].value_counts().items():
            print(f"    {name:30s} {count:,}")

    print(f"\n✓ Complete system finished. All outputs ready.")


if __name__ == "__main__":
    main()
