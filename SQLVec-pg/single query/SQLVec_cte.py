import psycopg2
import time
import numpy as np
import pandas as pd
import argparse


def main(args):
    conn = psycopg2.connect(database=args.db, user="postgres", password="pw", host="127.0.0.1", port="5432")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # to get the first query and its vector and dist
    cur.execute('SELECT data FROM query_table WHERE num = %s',(0,))
    xq = cur.fetchone()[0]
    cur.execute('SELECT data <-> %s as dist FROM data_table WHERE id = %s', (xq,args.enter_node))
    dist = cur.fetchone()[0]
    
    start_time_total = time.time()
    # core CTE
    cur.execute('''   
    WITH RECURSIVE search(L, V, dist) AS (  
        SELECT 
            ARRAY[%s]::integer[] AS L, 
            ARRAY[]::integer[] AS V,
            ARRAY[%s]::float8[] AS dist
        UNION ALL
                
        SELECT 
            lxv.newid AS L,
            ARRAY_APPEND(s.V, p_info.p_star) AS V,
            lxv.newdist AS dist
        FROM 
            search s, 
            
            LATERAL (SELECT lv AS p_star, MIN(dv) AS md
                FROM (SELECT UNNEST(s.L) as lv, UNNEST(s.dist) AS dv) AS tp
                WHERE lv NOT IN (SELECT UNNEST(s.V))
                GROUP BY lv
                ORDER BY md
                LIMIT 1
            ) AS p_info,
            
            LATERAL (
                SELECT ARRAY_AGG(neighbor) as neighbors
                FROM graph_index_table
                WHERE id = p_info.p_star
            ) AS p_neighbors,
                      
            LATERAL (
                SELECT ARRAY_AGG(l2.id) AS newid, ARRAY_AGG(l2.dist) AS newdist
                FROM (select l1.id, data_table.data <-> %s AS dist FROM (SELECT unnest(s.L) UNION SELECT unnest(neighbors)) AS l1(id)
                JOIN data_table 
                ON l1.id = data_table.id
                ORDER BY dist
                LIMIT %s) as l2
            ) AS lxv
                
        WHERE NOT s.V @> s.L
    )
    SELECT L FROM search;
    ''',(args.enter_node,dist,xq,args.l))
    end_time_total = time.time()
    
    resall = cur.fetchall()
    print(resall[:])
    print("Iteration: " + str(len(resall)))
    print("Total Time Taken: " + str(end_time_total - start_time_total))
    print("QPS: " + str(1/(end_time_total - start_time_total)))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='output')
    parser.add_argument('--db', type=str, default='sift')
    parser.add_argument('--enter_node', type=int, default=123742)
    parser.add_argument('--k', type=int, default=100, help='top-k search, k number')
    parser.add_argument('--l', type=int, default=100, help='the search list length')
    parser.add_argument('--querysize', type=int, default=10000, help='the query size of datasets') 
    parser.add_argument('--gt_path', type=str, help='the path of groundtruth file')
    parser.add_argument('--gt_type', type=str, default='.fvecs', help='the type of groundtruth file')
    args = parser.parse_args()

    main(args)
