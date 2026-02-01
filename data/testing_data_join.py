# ...existing code...
if __name__ == '__main__':
    # Automatically discover and load all CSVs under data/
    csvs = discover_csvs('data')
    if not csvs:
        print("No CSV files found under data/; aborting.")
        raise SystemExit(1)

    known_id_names = ['zip','zipcode','zip_code','zcta','zcta5','zcta_5','geoid','postal_code']
    loaded = []
    skipped = []

    for rel in csvs:
        p = PROJECT_ROOT / rel
        try:
            df = pd.read_csv(p, dtype=str, low_memory=False)
        except Exception as e:
            skipped.append((rel, f"read error: {e}"))
            continue

        # normalize column names
        df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)

        # try to find a known id column (case-insensitive)
        col_map = {c.lower(): c for c in df.columns}
        id_col = None
        for name in known_id_names:
            if name in col_map:
                id_col = col_map[name]
                break

        # if found, standardize; if not, attempt to coerce any column containing a 5+ digit pattern
        if id_col:
            df = standardize_zip_column(df, id_col)
        else:
            # attempt: find the first column where >30% of values contain 5 consecutive digits
            candidate = None
            for c in df.columns:
                s = df[c].astype(str).fillna('')
                prop = s.str.contains(r'\d{5}').mean()
                if prop >= 0.3:
                    candidate = c
                    break
            if candidate:
                df = standardize_zip_column(df, candidate)

        # keep only dataframes that now have at least one non-null 'zip' value
        if 'zip' in df.columns and df['zip'].notna().any():
            # ensure zip is string and drop rows without a zip (optional)
            df['zip'] = df['zip'].astype(str)
            loaded.append(df)
            print(f"Loaded {rel} -> normalized zip column present ({df['zip'].notna().sum()} non-null)")
        else:
            skipped.append((rel, "no zip-like column found / coerced"))

    if not loaded:
        print("None of the CSVs could be coerced to a 'zip' column. Inspect skipped files:")
        for rel, reason in skipped:
            print(f"  - {rel}: {reason}")
        raise SystemExit(1)

    # Merge all loaded datasets on 'zip' (outer join to preserve rows from all sources)
    merged = merge_datasets(loaded, on='zip', how='outer')
    out_path = PROJECT_ROOT / 'data' / 'merged_data.csv'
    merged.to_csv(out_path, index=False)
    print(f"Merged {len(loaded)} datasets -> {out_path} ({len(merged)} rows)")