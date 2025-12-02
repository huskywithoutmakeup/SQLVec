import numpy as np
import pickle
import time
import duckdb
import argparse


def main(args):
    con = duckdb.connect(args.db_path)
    con.execute("PRAGMA enable_optimizer") 

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

        if it>= args.t:
            con.execute('CREATE TEMP TABLE tpv AS SELECT d.id,d.dist,d.batch FROM (SELECT t1.*, row_number() over (PARTITION BY batch ORDER BY dist) AS class_rank FROM tp1 t1) d WHERE d.class_rank <= ? order by batch,dist',parameters=[args.w]) # windows para = 8   
        else:
            con.execute('CREATE TEMP TABLE tpv AS SELECT DISTINCT t1.* FROM tp1 t1 JOIN (SELECT batch, min(dist) AS mind FROM tp1 GROUP BY batch) t2 ON t1.batch = t2.batch AND t1.dist = t2.mind')   
        
        con.execute('CREATE TEMP TABLE tp2 AS SELECT DISTINCT batch,unnest(neighbor) as neighbor FROM tpv JOIN Graph_index_table USING(id)') 
        
        con.execute('CREATE TEMP TABLE tp3 AS SELECT neighbor as id, array_distance(data_table.data,query_table.data) as dist,tp2.batch FROM data_table,query_table,tp2 WHERE tp2.neighbor = data_table.id AND query_table.num = tp2.batch') 

        con.execute('INSERT INTO V_table SELECT id,batch FROM tpv') 
        con.execute('DROP TABLE tp1')
        con.execute('DROP TABLE tp2')
        con.execute('DROP TABLE tpv')
        
        con.execute('CREATE TABLE tp_table AS SELECT DISTINCT d.id,d.dist,d.batch FROM (SELECT l1.*, row_number() over (PARTITION BY batch ORDER BY dist) AS class_rank FROM (SELECT distinct * from L_table) l1) d WHERE d.class_rank <= ? order by batch,dist',parameters=[args.l])  # L para = 100  
        con.execute('DROP TABLE L_table')
        con.execute('DROP TABLE tp3')
        con.execute('ALTER TABLE tp_table RENAME TO L_table') 
    end_time_total = time.time()

    res_all = con.execute('SELECT d.id,d.batch FROM (SELECT l1.*, row_number() over (PARTITION BY batch ORDER BY dist) AS class_rank FROM L_table l1) d WHERE d.class_rank <= ? order by batch,dist',parameters=[args.k]).fetchnumpy() 
    batch_dict = {}
    for id, batch in zip(res_all['id'], res_all['batch']):
        if batch in batch_dict:
            batch_dict[batch].append(id)
        else:
            batch_dict[batch] = [id]
        
    result = list(batch_dict.values())

    # to load groundtruth vectors
    if args.gt_type == '.pkl':
        with open(args.gt_path,'rb') as f:
            ground_truth = pickle.load(f)
    else:
        print("Groundtruth type error! Please use .pkl file.")
        return 
    
    ground_truth = ground_truth[:,:args.k]

    intersection = 0
    for i in range(len(result)):
        intersection += len(np.intersect1d(ground_truth[i], result[i]))

    print("Iteration: " + str(it))
    print("Average Recall: " + str(intersection/(args.k*args.querysize)))
    print("Total Time Taken: " + str(end_time_total - start_time_total))
    print("QPS: " + str(1/(end_time_total - start_time_total)*args.querysize))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='baseline_post_filter')
    parser.add_argument('--db_path', type=str)
    parser.add_argument('--enter_node', type=int, default=123742)
    parser.add_argument('--k', type=int, default=100, help='top-k search, k number')
    parser.add_argument('--l', type=int, default=200, help='the search list length, post-filtering need larger L')
    parser.add_argument('--w', type=int, default=4, help='the expasion width of muti-node expansion')
    parser.add_argument('--t', type=int, default=10, help='the threshold of two-phase search')
    parser.add_argument('--querysize', type=int, default=10000, help='the query size of datasets') 
    parser.add_argument('--gt_path', type=str, help='the path of groundtruth file')
    parser.add_argument('--gt_type', type=str, default='.pkl', help='the type of groundtruth file')
    args = parser.parse_args()

    main(args)

