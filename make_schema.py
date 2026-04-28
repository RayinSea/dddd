import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd


def is_list_value(x):
    return isinstance(x, (list, tuple, np.ndarray))


def get_fid(col_name):
    m = re.search(r"_(\d+)$", col_name)
    if m is None:
        raise ValueError(f"Cannot parse fid from column name: {col_name}")
    return int(m.group(1))


def get_dim(series):
    max_len = 1
    for x in series.dropna():
        if is_list_value(x):
            max_len = max(max_len, len(x))
    return int(max_len)


def get_vocab_size(series):
    max_val = 0

    for x in series.dropna():
        if is_list_value(x):
            arr = np.asarray(x)
            if arr.size == 0:
                continue
            arr = arr.astype(np.int64, copy=False)
            arr = arr[arr > 0]
            if arr.size > 0:
                max_val = max(max_val, int(arr.max()))
        else:
            try:
                v = int(x)
            except Exception:
                continue
            if v > 0:
                max_val = max(max_val, v)

    return int(max_val + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default="./data/schema.json")
    parser.add_argument("--ts_fid", type=int, default=46)
    args = parser.parse_args()

    parquet_files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {args.data_dir}")

    dfs = []
    for path in parquet_files:
        print(f"Reading {path}")
        dfs.append(pd.read_parquet(path))

    df = pd.concat(dfs, ignore_index=True)
    print("Loaded shape:", df.shape)

    user_int = []
    item_int = []
    user_dense = []

    seq_groups = {}

    for col in df.columns:
        if col.startswith("user_int_feats_"):
            fid = get_fid(col)
            vocab_size = get_vocab_size(df[col])
            dim = get_dim(df[col])
            user_int.append([fid, vocab_size, dim])

        elif col.startswith("item_int_feats_"):
            fid = get_fid(col)
            vocab_size = get_vocab_size(df[col])
            dim = get_dim(df[col])
            item_int.append([fid, vocab_size, dim])

        elif col.startswith("user_dense_feats_"):
            fid = get_fid(col)
            dim = get_dim(df[col])
            user_dense.append([fid, dim])

        else:
            m = re.match(r"(domain_[a-z]+_seq)_(\d+)$", col)
            if m:
                prefix = m.group(1)
                fid = int(m.group(2))
                seq_groups.setdefault(prefix, []).append((fid, col))

    user_int.sort(key=lambda x: x[0])
    item_int.sort(key=lambda x: x[0])
    user_dense.sort(key=lambda x: x[0])

    seq = {}
    for prefix, fid_cols in sorted(seq_groups.items()):
        # domain_a_seq -> seq_a
        domain_letter = prefix.split("_")[1]
        seq_name = f"seq_{domain_letter}"

        features = []
        fids = [fid for fid, _ in fid_cols]
        has_ts = args.ts_fid in fids

        for fid, col in sorted(fid_cols, key=lambda x: x[0]):
            if has_ts and fid == args.ts_fid:
                # 时间戳列不作为 embedding 特征使用，这里给 1 即可
                vocab_size = 1
            else:
                vocab_size = get_vocab_size(df[col])
            features.append([fid, vocab_size])

        seq[seq_name] = {
            "prefix": prefix,
            "ts_fid": args.ts_fid if has_ts else None,
            "features": features,
        }

    schema = {
        "user_int": user_int,
        "item_int": item_int,
        "user_dense": user_dense,
        "seq": seq,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"Saved schema to {args.output}")
    print("user_int:", len(user_int))
    print("item_int:", len(item_int))
    print("user_dense:", len(user_dense))
    print("seq domains:", list(seq.keys()))
    for name, cfg in seq.items():
        print(name, "prefix:", cfg["prefix"], "ts_fid:", cfg["ts_fid"], "num_features:", len(cfg["features"]))


if __name__ == "__main__":
    main()
