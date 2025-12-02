import numpy as np
import pandas as pd
import pickle
import time
import duckdb
import struct
import argparse
import os

N_DIM = 128
DATA_SIZE = 4

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

def fvecs_read(filename, c_contiguous=True, record_count=-1, line_offset=0, record_dtype=np.int32):
    if record_count > 0:
        record_count *= N_DIM + 1
    if line_offset > 0:
        line_offset *= (N_DIM + 1) * DATA_SIZE
    fv = np.fromfile(filename, dtype=record_dtype, count=record_count, offset=line_offset)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    #print(dim)
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def main(args):
    con = duckdb.connect(args.db_path)

    con.execute("PRAGMA enable_optimizer") 
    con.execute('SET threads = 8')
    con.execute('SET worker_threads = 8')
    con.execute("SET memory_limit = '10GB';")
    con.execute("SET max_memory = '10GB';")

    enter_node = con.execute('SELECT data FROM data_table WHERE id = ?',parameters=[args.enter_node]).fetchall()[0][0]
    dim = len(enter_node)
    con.execute('CREATE TEMP TABLE L_table(id INTEGER, dist DOUBLE, batch INTEGER)') # L
    sql = 'INSERT INTO L_table SELECT ?,array_distance(?::FLOAT[' + str(dim) + '],data),num as batch from query_table'
    con.execute(sql,parameters=[args.enter_node,enter_node]) 
    con.execute('CREATE TEMP TABLE V_table(id INTEGER, batch INTEGER)') # V
    con.execute('CREATE INDEX vid ON V_table(id,batch);')

    start_time_total = time.time()
    cnt = 9
    it = 0
    while cnt != 0:
        it+=1
        con.execute('CREATE TEMP TABLE tp1 AS SELECT L_table.* FROM L_table LEFT JOIN V_table ON L_table.id = V_table.id AND L_table.batch = V_table.batch WHERE V_table.id IS NULL;')
        cnt = con.execute('SELECT count(*) FROM tp1').fetchall()[0][0]

        con.execute('CREATE TEMP TABLE tpv AS SELECT t1.* FROM tp1 t1 JOIN (SELECT batch, min(dist) AS mind FROM tp1 GROUP BY batch) t2 ON t1.batch = t2.batch AND t1.dist = t2.mind')   
        
        con.execute('CREATE TEMP TABLE tp2 AS SELECT batch,unnest(nbs) as neighbor FROM tpv JOIN Graph_index_table USING(id)') 
        
        con.execute('CREATE TEMP TABLE tp3 AS SELECT neighbor as id, array_distance(data_table.data,query_table.data) as dist,tp2.batch FROM data_table,query_table,tp2 WHERE tp2.neighbor = data_table.id AND query_table.num = tp2.batch') 

        con.execute('INSERT INTO V_table SELECT id,batch FROM tpv') 
        con.execute('DROP TABLE tp1')
        con.execute('DROP TABLE tp2')
        con.execute('DROP TABLE tpv')
        
        con.execute('CREATE TABLE tp_table AS SELECT d.id,d.dist,d.batch FROM (SELECT l1.*, row_number() over (PARTITION BY batch ORDER BY dist) AS class_rank FROM (SELECT * FROM L_table UNION SELECT * FROM tp3) l1) d WHERE d.class_rank <= ? order by batch,dist',parameters=[args.l]) 
        con.execute('DROP TABLE L_table')
        con.execute('DROP TABLE tp3')
        con.execute('ALTER TABLE tp_table RENAME TO L_table') 
    end_time_total = time.time()

    
    # to load groundtruth vectors
    if args.gt_type == '.fvecs':
        ground_truth = fvecs_read(args.gt_path)
    elif args.gt_type == '.bin':
        ground_truth = load_truthset(args.gt_path, args.k)
    
    ground_truth = ground_truth[:,:args.k]
    
    res_all = con.execute('SELECT d.id,d.batch FROM (SELECT l1.*, row_number() over (PARTITION BY batch ORDER BY dist) AS class_rank FROM L_table l1) d WHERE d.class_rank <= ? order by batch,dist',parameters=[args.k]).fetchnumpy() 
    batch_dict = {}
    for id, batch in zip(res_all['id'], res_all['batch']):
        if batch in batch_dict:
            batch_dict[batch].append(id)
        else:
            batch_dict[batch] = [id]
        
    result = list(batch_dict.values())
    intersection = 0
    for i in range(len(result)):
        intersection += len(np.intersect1d(ground_truth[i], result[i]))

    print("Iteration: " + str(it))
    print("Recall: " + str(intersection/(args.k*args.querysize)))
    print("Total Time Taken: " + str(end_time_total - start_time_total))
    print("QPS: " + str(1/((end_time_total - start_time_total)/args.querysize)))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SQLVec batched query-optimized version')
    parser.add_argument('--db_path', type=str)
    parser.add_argument('--enter_node', type=int, default=123742)
    parser.add_argument('--k', type=int, default=100, help='top-k search, k number')
    parser.add_argument('--l', type=int, default=100, help='the search list length')
    parser.add_argument('--querysize', type=int, default=10000, help='the query size of datasets') 
    parser.add_argument('--gt_path', type=str, help='the path of groundtruth file')
    parser.add_argument('--gt_type', type=str, default='.fvecs', help='the type of groundtruth file')
    args = parser.parse_args()

    main(args)

