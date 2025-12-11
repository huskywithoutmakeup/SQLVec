# kuzu-0.8.2
import numpy as np
import pandas as pd
import pickle
import time
import kuzu
import os
import struct
import argparse


def main(args):
    db = kuzu.Database(args.db_path)
    con = kuzu.Connection(db)

    res = con.execute("MATCH (d1:data_table) WHERE d1.id = $enter_id RETURN d1.data;",parameters={"enter_id": args.enter_node})
    enter_data = res.get_next()[0]
    res = con.execute("MATCH (q1:query_table) WHERE q1.num = $enter_id RETURN q1.data;",parameters={"enter_id": 0})
    query_data = res.get_next()[0]


    # create temp tables


    # con.execute('DROP TABLE L_table;')
    # con.execute('DROP TABLE V_table;')
    # con.execute('DROP TABLE temp_table;')
    con.execute('CREATE NODE TABLE L_table (id INT64, dist FLOAT, PRIMARY KEY (id))')
    con.execute('CREATE NODE TABLE V_table (id INT64, PRIMARY KEY (id))')
    con.execute('CREATE NODE TABLE temp_table (id INT64, dist FLOAT, PRIMARY KEY (id));')
    
    # clear old data and insert init data
    con.execute('MATCH (t:temp_table) DELETE t')
    con.execute('MATCH (v:V_table) DELETE v')
    con.execute('MATCH (l:L_table) DELETE l')
    con.execute("CREATE (l:L_table {id: $enter_id, dist: array_distance(CAST($enter_data, 'FLOAT[128]'), CAST($query_data, 'FLOAT[128]'))})",parameters={"enter_id": args.enter_node,"enter_data": enter_data,"query_data": query_data})  
    
    
    start_time_total = time.time()
    it = 0
    while True:
        it+=1
        print(it)

        # LOCATE
        res = con.execute('''MATCH (l:L_table)
                            WHERE NOT EXISTS {
                            MATCH (v:V_table {id: l.id})
                            }
                            WITH l  
                            ORDER BY l.dist ASC
                            LIMIT 1
                            RETURN l.id, l.dist ''')
        if res.has_next():
            point = res.get_next()[0]
        else:
            break

        # EXPAND & CALUCULATE & MERGE
        con.execute('''MATCH (d1:data_table {id: $enter_id})-[g:graph_index_table]->(d2:data_table) 
                            WITH d2.id AS nid, array_distance(d2.data, CAST($query_data, 'FLOAT[128]')) AS dist
                            WHERE NOT EXISTS {
                                MATCH (l:L_table {id: nid})
                            }
                            CREATE (l:L_table {id: nid, dist: dist});''',parameters={"enter_id": point,"query_data": query_data}) 
        con.execute("CREATE (v:V_table {id: $enter_id})",parameters={"enter_id": point})

        # LIMIT
        res = con.execute(''' MATCH (l:L_table)
                        WITH l
                        ORDER BY l.dist ASC  
                        SKIP $len
                        DELETE l
        ''',parameters={"len": args.l})  
    end_time_total = time.time()

    res = con.execute('MATCH (l:L_table) RETURN l.id')
    df = res.get_as_df()

    print("Total Time Taken: " + str(end_time_total - start_time_total))
    print("QPS: " + str(1/(end_time_total - start_time_total)))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='single query')
    parser.add_argument('--db_path', type=str,required=True)
    parser.add_argument('--enter_node', type=int, default=123742)
    parser.add_argument('--k', type=int, default=100, help='top-k search, k number')
    parser.add_argument('--l', type=int, default=100, help='the search list length')
    args = parser.parse_args()

    main(args)


