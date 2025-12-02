import psycopg2
import time
import argparse


def main(args):
    conn = psycopg2.connect(database=args.db, user="postgres", password="229229", host="127.0.0.1", port="5432")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # function
    cur.execute('DROP FUNCTION IF EXISTS sqlvec')
    cur.execute('''CREATE OR REPLACE FUNCTION sqlvec(enter_node_id INTEGER, query_node_num INTEGER, L INTEGER,K INTEGER)
                    RETURNS SETOF INTEGER[] AS $$
                    DECLARE
                        enter_node vector;
                        query_node vector;
                        cpoint INTEGER;
                        cnt INTEGER := 9;
                    BEGIN                   

                        SELECT data INTO enter_node FROM data_table WHERE id = enter_node_id;
                        SELECT data INTO query_node FROM query_table WHERE num = query_node_num;

                        CREATE TEMP TABLE L_table (id INTEGER, dist DOUBLE PRECISION);
                        CREATE TEMP TABLE V_table (id INTEGER);

                        INSERT INTO L_table SELECT enter_node_id, enter_node <-> query_node as dist;

                        CREATE VIEW tp1 AS SELECT L_table.* FROM L_table LEFT JOIN V_table ON L_table.id = V_table.id WHERE V_table.id IS NULL;
                        
                        WHILE cnt != 0 LOOP
                            WITH tp1_data AS (SELECT id, dist FROM tp1), 
                                count AS (SELECT COUNT(*) AS cnt FROM tp1_data)
                            SELECT tp1_data.id, count.cnt INTO cpoint,cnt FROM tp1_data, count WHERE dist = (SELECT MIN(dist) FROM tp1_data);

                            INSERT INTO L_table SELECT id, data <-> query_node as dist FROM data_table WHERE id IN (SELECT neighbor FROM Graph_index_table WHERE id = cpoint);
                            INSERT INTO V_table VALUES (cpoint);

                            CREATE TEMP TABLE tp_table AS SELECT DISTINCT id, dist FROM L_table ORDER BY dist LIMIT L;
                            TRUNCATE TABLE L_table;
                            INSERT INTO L_table SELECT * FROM tp_table;
                            DROP TABLE tp_table;
                        END LOOP;
                        DROP VIEW IF EXISTS tp1;

                        RETURN QUERY SELECT ARRAY_AGG(id) FROM L_table LIMIT K;
                    END;
                    $$ LANGUAGE plpgsql;''')

    start_time_total = time.time()
    cur.execute('SELECT sqlvec(%s,%s,%s,%s)',(args.enter_node,0,args.l,args.k))
    end_time_total = time.time()

    res = cur.fetchall()
    print(res)
    print("Total Time Taken: " + str(end_time_total - start_time_total))
    print("QPS: " + str(1/(end_time_total - start_time_total)))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sqlvec_plsql')
    parser.add_argument('--db', type=str, default='sift')
    parser.add_argument('--enter_node', type=int, default=123742)
    parser.add_argument('--k', type=int, default=100, help='top-k search, k number')
    parser.add_argument('--l', type=int, default=100, help='the search list length')
    parser.add_argument('--querysize', type=int, default=10000, help='the query size of datasets') 
    parser.add_argument('--gt_path', type=str, help='the path of groundtruth file')
    parser.add_argument('--gt_type', type=str, default='.fvecs', help='the type of groundtruth file')
    args = parser.parse_args()

    main(args)
