import numpy as np
import pandas as pd
import time
import duckdb
import os
import argparse
import struct

N_DIM = 128
DATA_SIZE = 4

def fvecs_read(filename, c_contiguous=True, record_count=-1, line_offset=0, record_dtype=np.int32):
    if record_count > 0:
        record_count *= N_DIM + 1
    if line_offset > 0:
        line_offset *= (N_DIM + 1) * DATA_SIZE
    fv = np.fromfile(filename, dtype=record_dtype, count=record_count, offset=line_offset)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
        fv = fv.copy()
    return fv

def load_truthset(bin_file, k):
    actual_file_size = os.path.getsize(bin_file)

    with open(bin_file, 'rb') as f:
        npts_i32 = struct.unpack('i', f.read(4))[0]
        dim_i32 = struct.unpack('i', f.read(4))[0]
        npts = npts_i32
        dim = dim_i32

        truthset_type = -1  # 1 means truthset has ids and distances, 2 means only ids, -1 is error
        expected_file_size_with_dists = 2 * npts * dim * 4 + 2 * 4
        expected_file_size_just_ids = npts * dim * 4 + 2 * 4

        if actual_file_size == expected_file_size_with_dists:
            truthset_type = 1
        elif actual_file_size == expected_file_size_just_ids:
            truthset_type = 2

        if truthset_type == -1:
            raise ValueError(f"Error. File size mismatch. File should have bin format, with "
                             f"npts followed by ngt followed by npts*ngt ids and optionally "
                             f"followed by npts*ngt distance values; actual size: "
                             f"{actual_file_size}, expected: {expected_file_size_with_dists} or "
                             f"{expected_file_size_just_ids}")

        ids = np.fromfile(f, dtype=np.uint32, count=npts * dim)
        dists = None

        if truthset_type == 1:
            dists = np.fromfile(f, dtype=np.float32, count=npts * dim)

    return ids.reshape(npts, k)

def main(args):
    con = duckdb.connect(args.db_path) # :memory:

    con.install_extension("vss")
    con.load_extension("vss")

    con.execute("UPDATE EXTENSIONS (vss);") 

    con.execute("SET hnsw_enable_experimental_persistence = 'true'")
    con.execute("PRAGMA enable_optimizer") 
    con.execute('SET threads = 8')
    con.execute('SET worker_threads = 8')
    con.execute("SET memory_limit = '10GB';")
    con.execute("SET max_memory = '10GB';")
    
    # build hnsw index
    con.execute("DROP INDEX IF EXISTS hnsw_idx")
    s = time.time()
    con.execute("CREATE INDEX hnsw_idx ON data_table USING HNSW (data) WITH (metric = 'l2sq',m = 32,ef_construction = 128)") 
    e = time.time()
    print("Index build taken: "+ str(e-s))

    # query = fvecs_read("/home/anns/ANNS/dataset/sift1m/sift_query.fvecs", record_dtype=np.float32)
    con.execute("SET hnsw_ef_search = ?;",parameters=[args.l]) # k=100
    q = con.execute('SELECT data FROM query_table;').fetchall()
    dim = len(q[0][0])
    
    sql = 'SELECT id FROM data_table ORDER BY array_distance(data_table.data, ?::FLOAT[' + str(dim) + ']) limit ;' + str(args.k)
    res = []
    start_time_total = time.time()
    for i in range(args.querysize):
      res.append(con.execute(sql,parameters=[q[i][0]]).fetchnumpy()['id'])
    end_time_total = time.time()

    # to load groundtruth vectors
    if args.gt_type == '.fvecs':
        ground_truth = fvecs_read(args.gt_path)
    elif args.gt_type == '.bin':
        ground_truth = load_truthset(args.gt_path, args.k)
    
    ground_truth = ground_truth[:,:args.k]

    intersection = 0
    for i in range(len(res)):
        intersection += len(np.intersect1d(ground_truth[i], res[i]))

    print("Recall: " + str(intersection/(args.k*args.querysize)))
    print("Total Time Taken: " + str(end_time_total - start_time_total))
    print("QPS: " + str(1/((end_time_total - start_time_total)/args.querysize)))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='baseline_duckdb_vss')
    parser.add_argument('--db_path', type=str)
    parser.add_argument('--k', type=int, default=100, help='top-k search, k number')
    parser.add_argument('--l', type=int, default=100, help='the search list length')
    parser.add_argument('--querysize', type=int, default=10000, help='the query size of datasets') 
    parser.add_argument('--gt_path', type=str, help='the path of groundtruth file')
    parser.add_argument('--gt_type', type=str, default='.fvecs', help='the type of groundtruth file')
    args = parser.parse_args()

    main(args)