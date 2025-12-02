import psycopg2
import time
import numpy as np
import os
import struct
import argparse

def fvecs_read(dim, filename, c_contiguous=True, record_count=-1, line_offset=0, record_dtype=np.int32):
    if record_count > 0:
        record_count *= dim + 1
    if line_offset > 0:
        line_offset *= (dim + 1) * 4
    fv = np.fromfile(filename, dtype=record_dtype, count=record_count, offset=line_offset)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def load_truthset(bin_file,k):
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
    conn = psycopg2.connect(database=args.db, user="postgres", password="pw", host="127.0.0.1", port="5432")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')

    cur.execute('SELECT data FROM data_table WHERE id = %s', (args.enter_node,))
    enter_node = cur.fetchone()
    # L_table,V_table
    cur.execute('CREATE TEMP TABLE L_table (id INTEGER, dist DOUBLE PRECISION, batch INTEGER)')
    cur.execute('INSERT INTO L_table SELECT %s, %s <-> data as dist, num AS batch FROM query_table', (args.enter_node, enter_node))
    cur.execute('CREATE TEMP TABLE V_table (id INTEGER, batch INTEGER)')
    cur.execute('CREATE INDEX vid ON V_table USING btree (id,batch);')

    cur.execute("SELECT pg_backend_pid();")
    backend_pid = cur.fetchone()[0]
    print(f"PostgreSQL backend id (PID): {backend_pid}")
    time.sleep(30)
    
    cnt = 9
    it = 0
    start_time_total = time.time()
    while cnt != 0:
        it += 1
        cur.execute('CREATE TEMP TABLE tp1 AS SELECT L.* FROM L_table L WHERE (L.id, L.batch) NOT IN (SELECT V.id, V.batch FROM V_table V);')
        cur.execute('SELECT count(*) FROM tp1')
        cnt = cur.fetchone()[0]
  
        if it>= args.t:
            cur.execute('CREATE TEMP TABLE tpv AS SELECT d.id,d.dist,d.batch FROM (SELECT t1.*, row_number() over (PARTITION BY batch ORDER BY dist) AS class_rank FROM tp1 t1) d WHERE d.class_rank <= %s order by batch,dist',(args.w,)) # windows para = 8   
        else:
            cur.execute('CREATE TEMP TABLE tpv AS SELECT DISTINCT t1.* FROM tp1 t1 JOIN (SELECT batch, min(dist) AS mind FROM tp1 GROUP BY batch) t2 ON t1.batch = t2.batch AND t1.dist = t2.mind') 

        cur.execute('CREATE TEMP TABLE tp2 AS SELECT DISTINCT batch, UNNEST(nbs) AS neighbor FROM tpv JOIN Graph_index_table ON tpv.id = Graph_index_table.id;')  

        cur.execute('CREATE TEMP TABLE tp3 AS SELECT neighbor AS id, data_table.data <-> query_table.data AS dist, tp2.batch FROM data_table, query_table, tp2 WHERE tp2.neighbor = data_table.id AND query_table.num = tp2.batch')

        cur.execute('INSERT INTO V_table SELECT id,batch FROM tpv')
        cur.execute('DROP TABLE tp1')
        cur.execute('DROP TABLE tp2')
        cur.execute('DROP TABLE tpv')

        cur.execute('CREATE TEMP TABLE tp_table AS SELECT d.id, d.dist, d.batch FROM (SELECT l1.*, row_number() OVER (PARTITION BY batch ORDER BY dist) AS class_rank FROM (SELECT * FROM L_table UNION SELECT * FROM tp3) l1) d WHERE d.class_rank <= %s ORDER BY batch, dist',(args.l,))
        cur.execute('DROP TABLE L_table')
        cur.execute('DROP TABLE tp3')
        cur.execute('ALTER TABLE tp_table RENAME TO L_table')
    end_time_total = time.time()

    cur.execute('SELECT d.id, d.batch FROM (SELECT l1.*, row_number() OVER (PARTITION BY batch ORDER BY dist) AS class_rank FROM L_table l1) d WHERE d.class_rank <= %s ORDER BY batch, dist',(args.k,))
    res_all = cur.fetchall()

    conn.commit()
    conn.close()

    # to load groundtruth vectors
    if args.gt_type == '.fvecs':
        ground_truth = fvecs_read(args.dim, args.gt_path)
    elif args.gt_type == '.bin':
        ground_truth = load_truthset(args.gt_path, args.k)
    
    ground_truth = ground_truth[:,:args.k]

    # to get the result vector of each batch
    batch_dict = {}
    for i in range(len(res_all)):
        if res_all[i][1] in batch_dict:
            batch_dict[res_all[i][1]].append(res_all[i][0])
        else:
            batch_dict[res_all[i][1]] = [res_all[i][0]]
        
    result = list(batch_dict.values())

    intersection = 0
    for i in range(len(result)):
        intersection += len(np.intersect1d(ground_truth[i], result[i]))
    
    print("Iteration: " + str(it))
    print("Average Recall: " + str(intersection/(args.k*args.querysize)))
    print("Total Time Taken: " + str(end_time_total - start_time_total))
    print("QPS: " + str(1/(end_time_total - start_time_total)*args.querysize))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='batched query-optimized')
    parser.add_argument('--db', type=str, default='sift')
    parser.add_argument('--enter_node', type=int, default=123742)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--k', type=int, default=100, help='top-k search, k number')
    parser.add_argument('--l', type=int, default=100, help='the search list length')
    parser.add_argument('--w', type=int, default=4, help='the expasion width of muti-node expansion')
    parser.add_argument('--t', type=int, default=10, help='the threshold of two-phase search')
    parser.add_argument('--querysize', type=int, default=10000, help='the query size of datasets') 
    parser.add_argument('--gt_path', type=str, help='the path of groundtruth file')
    parser.add_argument('--gt_type', type=str, default='.fvecs', help='the type of groundtruth file')
    args = parser.parse_args()
    main(args)