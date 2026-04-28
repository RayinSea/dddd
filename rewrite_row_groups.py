import os
import shutil

import pyarrow.parquet as pq


src_parquet = "./data/demo_1000.parquet"
src_schema = "./data/schema.json"

dst_dir = "./data_rg"
dst_parquet = os.path.join(dst_dir, "demo_1000.parquet")
dst_schema = os.path.join(dst_dir, "schema.json")

os.makedirs(dst_dir, exist_ok=True)

table = pq.read_table(src_parquet)

# 每 100 行写成一个 Row Group
pq.write_table(table, dst_parquet, row_group_size=100)

shutil.copyfile(src_schema, dst_schema)

pf = pq.ParquetFile(dst_parquet)
print("saved to:", dst_parquet)
print("num_rows =", pf.metadata.num_rows)
print("num_row_groups =", pf.metadata.num_row_groups)

for i in range(pf.metadata.num_row_groups):
    print(i, pf.metadata.row_group(i).num_rows)
