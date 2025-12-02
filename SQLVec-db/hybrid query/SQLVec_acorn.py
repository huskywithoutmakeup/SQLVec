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
    con.execute('CREATE TEMP TABLE L_table(id INTEGER, dist DOUBLE, batch INTEGER)') 
    sql = 'INSERT INTO L_table SELECT ?,array_distance(?::FLOAT[' + str(dim) + '],data),num as batch from query_table'
    con.execute(sql,parameters=[args.enter_node,enter_node]) 
    con.execute('CREATE TEMP TABLE V_table(id INTEGER, batch INTEGER)') # V
    
    start_time_total = time.time()
    cnt = 9
    it = 0
    while cnt != 0:
        it+=1
        con.execute('CREATE TEMP TABLE tp1 AS SELECT L_table.* FROM L_table LEFT JOIN V_table ON L_table.id = V_table.id AND L_table.batch = V_table.batch WHERE V_table.id IS NULL;')
        cnt = con.execute('SELECT count(*) FROM tp1').fetchall()[0][0]

        con.execute('CREATE TEMP TABLE tpv AS SELECT t.id,t.dist,t.batch FROM (SELECT tp1.*, MIN(dist) OVER (PARTITION BY batch) AS md FROM tp1) t WHERE t.dist = t.md;') 

        # the modified operator to find 2-hop neighbors
        con.execute('''CREATE TEMP TABLE tp2 AS 
                            SELECT DISTINCT t1.batch AS batch, unnest(t2.neighbor) AS neighbor
                            FROM (SELECT batch, unnest(neighbor) AS id FROM tpv JOIN Graph_index_table USING(id)) t1 
                            JOIN Graph_index_table t2 USING(id);''') 
        

        con.execute('''CREATE TEMP TABLE tp3 AS SELECT neighbor AS id,
                                                array_distance(data_table.data, query_table.data) AS dist,
                                                tp2.batch
                                                FROM tp2
                                                JOIN data_table ON tp2.neighbor = data_table.id
                                                JOIN query_table ON query_table.num = tp2.batch
                                                WHERE data_table.a1 = query_table.a1
                                            ''') 

        con.execute('INSERT INTO V_table SELECT id,batch FROM tpv') 
        con.execute('DROP TABLE tp1')
        con.execute('DROP TABLE tp2')
        con.execute('DROP TABLE tpv')

        con.execute('CREATE TABLE tp_table AS SELECT d.id,d.dist,d.batch FROM (SELECT l1.*, row_number() over (PARTITION BY batch ORDER BY dist) AS class_rank FROM (SELECT * FROM L_table UNION SELECT * FROM tp3) l1) d WHERE d.class_rank <= ? order by batch,dist',parameters=[args.l])  # L para = 100  
        con.execute('DROP TABLE tp3')
        con.execute('DROP TABLE L_table')
        con.execute('ALTER TABLE tp_table RENAME TO L_table')

    end_time_total = time.time()

    res_all = con.execute('SELECT d.id,d.batch FROM (SELECT l1.*, row_number() over (PARTITION BY batch ORDER BY dist) AS class_rank FROM L_table l1) d WHERE d.class_rank <= ? order by batch,dist;',parameters=[args.k]).fetchnumpy() 
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
    parser = argparse.ArgumentParser(description='hybrid query')
    parser.add_argument('--db_path', type=str)
    parser.add_argument('--enter_node', type=int, default=123742)
    parser.add_argument('--k', type=int, default=100, help='top-k search, k number')
    parser.add_argument('--l', type=int, default=100, help='the search list length')
    parser.add_argument('--querysize', type=int, default=10000, help='the query size of datasets') 
    parser.add_argument('--gt_path', type=str, help='the path of groundtruth file')
    parser.add_argument('--gt_type', type=str, default='.pkl', help='the type of groundtruth file')
    args = parser.parse_args()

    main(args)


