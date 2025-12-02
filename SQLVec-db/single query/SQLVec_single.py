import numpy as np
import pandas as pd
import time
import duckdb
import argparse

def main(args):
    con = duckdb.connect(args.db_path)
    con.execute("PRAGMA enable_optimizer") 

    enter_node = con.execute('SELECT data FROM data_table WHERE id = ?',parameters=[args.enter_node]).fetchall()[0][0]
    query_node = con.execute('SELECT data FROM query_table WHERE num = ? ',parameters=[0]).fetchall()[0][0] 
    dim = len(enter_node)

    start_time_total = time.time()
    con.execute('CREATE TEMP TABLE L_table(id INTEGER, dist DOUBLE)') # L
    con.execute(f'INSERT INTO L_table VALUES (?,array_distance(?::FLOAT[{dim}],?::FLOAT[{dim}]))',parameters=[args.enter_node,enter_node,query_node])  
    con.execute('CREATE TEMP TABLE V_table(id INTEGER, dist DOUBLE)') # V
       
    cnt = 9
    it = 0
    while True:
        con.execute('CREATE TEMP TABLE tp1 AS SELECT * FROM L_table EXCEPT SELECT * FROM V_table')
        cnt = con.execute('SELECT count(*) FROM tp1').fetchall()[0][0]
        if cnt == 0:
            break
        cpoint = con.execute('SELECT id,dist FROM tp1 ORDER BY dist LIMIT 1').fetchall()[0]

        con.execute(f'INSERT INTO L_table SELECT id, array_distance(data,?::FLOAT[{dim}]) as dist FROM data_table WHERE id IN (SELECT UNNEST(nbs) FROM Graph_index_table WHERE id = ?)',parameters=[query_node, cpoint[0]])      

        con.execute('INSERT INTO V_table VALUES (?, ?)',parameters=[cpoint[0],cpoint[1]])
        con.execute("DROP TABLE tp1") 
             
        con.execute(f'CREATE TABLE tp_table AS SELECT DISTINCT id,dist FROM L_table ORDER BY dist LIMIT ?',parameters=[args.l]) 
        con.execute('DROP TABLE L_table')
        con.execute('ALTER TABLE tp_table RENAME TO L_table') 
        it+=1

    res = con.execute(f'SELECT id FROM L_table LIMIT ?',parameters=[args.k]).fetchnumpy()
    con.execute('DROP TABLE L_table')
    con.execute('DROP TABLE V_table')
    end_time_total = time.time()

    # result and performance
    print("Result IDs: " + str(res))
    print("Iterations: " + str(it))
    print("Total Time Taken: " + str(end_time_total - start_time_total))
    print("QPS: " + str(1/(end_time_total - start_time_total)))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='single query')
    parser.add_argument('--db_path', type=str)
    parser.add_argument('--enter_node', type=int, default=123742)
    parser.add_argument('--k', type=int, default=100, help='top-k search, k number')
    parser.add_argument('--l', type=int, default=100, help='the search list length')
    args = parser.parse_args()

    main(args)
