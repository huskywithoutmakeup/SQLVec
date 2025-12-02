import psycopg2
import time
import numpy as np
import pandas as pd
import argparse
import pickle

def main(args):
    conn = psycopg2.connect(database=args.db, user="postgres", password="pw", host="127.0.0.1", port="5432")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    start_time_total = time.time()
    # init node
    cur.execute('SELECT data FROM data_table WHERE id = %s', (args.enter_node,))
    enter_node = cur.fetchone()
    # L_table,V_table
    cur.execute('CREATE TEMP TABLE L_table (id INTEGER, dist DOUBLE PRECISION, batch INTEGER)')
    cur.execute('INSERT INTO L_table SELECT %s, %s <-> data as dist, num AS batch FROM query_table', (args.enter_node, enter_node))
    cur.execute('CREATE TEMP TABLE V_table (id INTEGER, batch INTEGER)')
    cur.execute('CREATE INDEX vid ON V_table USING btree (id,batch);')

    cnt = 9
    it = 0
    while cnt != 0:
        it += 1
        cur.execute('CREATE TEMP TABLE tp1 AS SELECT L_table.* FROM L_table LEFT JOIN V_table ON L_table.id = V_table.id AND L_table.batch = V_table.batch WHERE V_table.id IS NULL;')
        cur.execute('SELECT count(*) FROM tp1')
        cnt = cur.fetchone()[0]
  
        cur.execute('CREATE TEMP TABLE tpv AS SELECT t1.* FROM tp1 t1 JOIN (SELECT batch, min(dist) AS mind FROM tp1 GROUP BY batch) t2 ON t1.batch = t2.batch AND t1.dist = t2.mind')

        cur.execute('''CREATE TEMP TABLE tp2 AS SELECT DISTINCT t1.batch AS batch, unnest(t2.nbs) AS neighbor
                                                    FROM (SELECT batch, unnest(nbs) AS id FROM tpv JOIN graph_table USING(id)) t1 
                                                    JOIN graph_table t2 USING(id);''')
        
        cur.execute('''CREATE TEMP TABLE tp3 AS SELECT neighbor AS id, data_table.data <-> query_table.data AS dist, tp2.batch
                                        FROM tp2
                                        JOIN data_table ON tp2.neighbor = data_table.id
                                        JOIN query_table ON query_table.num = tp2.batch
                                        WHERE data_table.a1 = query_table.a1
                                    ''') 

        cur.execute('INSERT INTO V_table SELECT id,batch FROM tpv')
        cur.execute('DROP TABLE tp1')
        cur.execute('DROP TABLE tp2')
        cur.execute('DROP TABLE tpv')

        cur.execute('CREATE TEMP TABLE tp_table AS SELECT d.id, d.dist, d.batch FROM (SELECT l1.*, row_number() OVER (PARTITION BY batch ORDER BY dist) AS class_rank FROM (SELECT DISTINCT * FROM L_table) l1) d WHERE d.class_rank <= %s ORDER BY batch, dist', (args.l,))
        cur.execute('DROP TABLE L_table')
        cur.execute('DROP TABLE tp3')
        cur.execute('ALTER TABLE tp_table RENAME TO L_table')
    end_time_total = time.time()

    cur.execute('SELECT d.id, d.batch FROM (SELECT l1.*, row_number() OVER (PARTITION BY batch ORDER BY dist) AS class_rank FROM L_table l1) d WHERE d.class_rank <= %s ORDER BY batch, dist', (args.k,))
    res_all = cur.fetchall()

    conn.commit()
    conn.close()

    # to load groundtruth vectors
    if args.gt_type == '.pkl':
        with open(args.gt_path,'rb') as f:
            ground_truth = pickle.load(f)
    else:
        print("Groundtruth type error! Please use .pkl file.")

    ground_truth = ground_truth[:,:args.k]

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
    parser = argparse.ArgumentParser(description='hybrid query')
    parser.add_argument('--db', type=str, default='sift')
    parser.add_argument('--enter_node', type=int, default=123742)
    parser.add_argument('--k', type=int, default=100, help='top-k search, k number')
    parser.add_argument('--l', type=int, default=100, help='the search list length')
    parser.add_argument('--querysize', type=int, default=10000, help='the query size of datasets') 
    parser.add_argument('--gt_path', type=str, help='the path of groundtruth file')
    parser.add_argument('--gt_type', type=str, default='.pkl', help='the type of groundtruth file')
    args = parser.parse_args()

    main(args)

