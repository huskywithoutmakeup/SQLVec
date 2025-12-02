import numpy as np
import pandas as pd
import pickle
import time
import duckdb
import argparse

def main(args):
    con = duckdb.connect(args.db_path)
    con.execute("PRAGMA enable_optimizer") 

    query_node = con.execute('SELECT data FROM query_table WHERE num = ?',parameters=[0]).fetchall()[0][0]
    enter_node = con.execute('SELECT data FROM data_table WHERE id = ?',parameters=[args.enter_node]).fetchall()[0][0]
    dim = len(enter_node)

    start_time = time.time()
    res = con.execute(f'''   
    WITH RECURSIVE search(L, V) AS (
        SELECT 
            [?] AS L, 
            [] AS V,
            [array_distance(?::FLOAT[{dim}], ?::FLOAT[{dim}])] AS dist
        UNION ALL
                
        SELECT 
            lxv.newid AS L,
            list_append(s.V, p_info.p_star) AS V,
            lxv.newdist AS dist
        FROM 
            search s, 

            LATERAL (SELECT lv AS p_star, MIN(dv) AS md
                FROM (SELECT UNNEST(s.L) as lv, UNNEST(s.dist) AS dv)
                WHERE lv NOT IN (SELECT UNNEST(s.V))
                GROUP BY lv
                ORDER BY md
                LIMIT 1
            ) AS p_info,
            
            LATERAL (
                SELECT neighbor
                FROM graph_index_table
                WHERE id = p_info.p_star
            ) AS p_neighbors,
                      
            LATERAL (
                SELECT list(l2.id) AS newid, list(l2.dist) AS newdist
                FROM (select l1.id,array_distance(data_table.data,?::FLOAT[{dim}]) AS dist FROM (SELECT unnest(s.L) UNION SELECT unnest(neighbor)) AS l1(id)
                JOIN data_table ON l1.id = data_table.id
                ORDER BY dist
                LIMIT ?) as l2
            ) AS lxv
                
        WHERE NOT list_has_all(s.V, s.L)
    )
    SELECT L FROM search;
    ''',parameters=[args.enter_node,enter_node,query_node,query_node,args.l]).fetchall()
    end_time = time.time()
    n = len(res)

    # result and performance
    print("Result IDs: " + str(res[n-1][0]))
    print("Iterations: " + str(n))
    print("Total Time Taken: " + str(end_time - start_time))
    print("QPS: " + str(1/(end_time - start_time)))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='single query cte')
    parser.add_argument('--db_path', type=str)
    parser.add_argument('--enter_node', type=int, default=123742)
    parser.add_argument('--k', type=int, default=100, help='top-k search, k number')
    parser.add_argument('--l', type=int, default=100, help='the search list length')
    args = parser.parse_args()

    main(args)



