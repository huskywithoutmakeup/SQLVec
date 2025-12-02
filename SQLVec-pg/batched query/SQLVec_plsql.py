import psycopg2
import time
import numpy as np
import pandas as pd
import pickle
import argparse

def main(args):
    conn = psycopg2.connect(database=args.db, user="postgres", password="pw", host="127.0.0.1", port="5432")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # function
    cur.execute('DROP FUNCTION IF EXISTS batchsearch')
    cur.execute('''CREATE OR REPLACE FUNCTION batchsearch(enter_node_id INTEGER, L INTEGER,K INTEGER)
                    RETURNS TABLE(ids INTEGER[], bid INTEGER) AS $$
                    DECLARE
                        enter_node vector;
                        cnt INTEGER := 9;
                    BEGIN                   
                        SELECT data INTO enter_node FROM data_table WHERE id = enter_node_id;

                        CREATE TEMP TABLE L_table (id INTEGER, dist DOUBLE PRECISION, batch INTEGER);
                        CREATE TEMP TABLE V_table (id INTEGER, batch INTEGER);

                        INSERT INTO L_table SELECT enter_node_id, enter_node <-> data as dist, num AS batch FROM query_table;
                        
                        WHILE cnt != 0 LOOP
                            CREATE TEMP TABLE tp1 AS SELECT L_table.* FROM L_table LEFT JOIN V_table ON L_table.id = V_table.id AND L_table.batch = V_table.batch WHERE V_table.id IS NULL;
                            SELECT count(*) INTO cnt FROM tp1;

                            CREATE TEMP TABLE tpv AS SELECT t1.* FROM tp1 t1 JOIN (SELECT batch, min(dist) AS mind FROM tp1 GROUP BY batch) t2 ON t1.batch = t2.batch AND t1.dist = t2.mind;

                            CREATE TEMP TABLE tp2 AS SELECT batch, UNNEST(nbs) as neighbor FROM tpv JOIN Graph_index_table ON tpv.id = Graph_index_table.id;
                            
                            CREATE TEMP TABLE tp3 AS SELECT neighbor AS id, data_table.data <-> query_table.data AS dist, tp2.batch FROM data_table, query_table, tp2 WHERE tp2.neighbor = data_table.id AND query_table.num = tp2.batch;

                            INSERT INTO V_table SELECT id,batch FROM tpv;
                            DROP TABLE tp1;
                            DROP TABLE tp2;
                            DROP TABLE tpv;

                            CREATE TEMP TABLE tp_table AS SELECT d.id, d.dist, d.batch FROM (SELECT l1.*, row_number() OVER (PARTITION BY batch ORDER BY dist) AS class_rank FROM (SELECT * FROM L_table UNION SELECT * FROM tp3) l1) d WHERE d.class_rank <= L ORDER BY batch, dist;
                            DROP TABLE L_table;
                            DROP TABLE tp3; 
                            ALTER TABLE tp_table RENAME TO L_table;
                        END LOOP;

                        RETURN QUERY SELECT ARRAY_AGG(d.id), d.batch FROM (SELECT l1.*, row_number() OVER (PARTITION BY batch ORDER BY dist) AS rank FROM L_table l1) d WHERE d.rank <= K GROUP BY d.batch ORDER BY d.batch;
                    END;
                    $$ LANGUAGE plpgsql;''')

    start_time_total = time.time()
    cur.execute('SELECT batchsearch(%s,%s,%s)', (args.enter_node, args.l, args.k))
    end_time_total = time.time()
    
    # res = cur.fetchall()
    # print(res)
    print("Total Time Taken: " + str(end_time_total - start_time_total))
    print("QPS: " + str(1/(end_time_total - start_time_total)*args))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='batched query-plsql')
    parser.add_argument('--enter_node', type=int, default=123742)
    parser.add_argument('--k', type=int, default=100, help='top-k search, k number')
    parser.add_argument('--l', type=int, default=100, help='the search list length')
    parser.add_argument('--querysize', type=int, default=10000, help='the query size of datasets') 
    parser.add_argument('--gt_path', type=str, help='the path of groundtruth file')
    parser.add_argument('--gt_type', type=str, default='.fvecs', help='the type of groundtruth file')
    args = parser.parse_args()

    main(args)


