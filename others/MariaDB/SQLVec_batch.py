import mariadb 
import time
import numpy as np
import pandas as pd
import pickle

def main():
    conn = mariadb.connect(database="test", user="root", password="", host="localhost", port=3306, unix_socket="/tmp/mysql.sock")
    conn.autocommit = True

    cur = conn.cursor()
    cur.execute("DROP PROCEDURE IF EXISTS batchsearch") 
    cur.execute('''
                    CREATE PROCEDURE batchsearch(IN enter_node_id INT, IN query_node_num INT, IN L INT, IN R INT)
                    BEGIN
                        DECLARE enter_node vector(128);
                        DECLARE cnt INT DEFAULT 9;

                        SELECT data INTO enter_node FROM data_table WHERE id = enter_node_id;

                        CREATE TEMPORARY TABLE L_table (id INT, dist DOUBLE, batch INT);
                        CREATE TEMPORARY TABLE V_table (id INT, batch INT);

                        INSERT INTO L_table SELECT enter_node_id, VEC_DISTANCE_EUCLIDEAN(enter_node, data) AS dist, num AS batch FROM query_table;

                        WHILE cnt != 0 DO
                            CREATE TEMPORARY TABLE tp1 AS SELECT L_table.* FROM L_table LEFT JOIN V_table ON L_table.id = V_table.id WHERE V_table.id IS NULL;
                            SELECT count(*) INTO cnt FROM tp1;

                            CREATE TEMPORARY TABLE tpv AS SELECT t1.* FROM tp1 t1 JOIN (SELECT batch, min(dist) AS mind FROM tp1 GROUP BY batch) t2 ON t1.batch = t2.batch AND t1.dist = t2.mind;

                            CREATE TEMPORARY TABLE tp2 AS SELECT batch, neighbor FROM tpv JOIN graph_index_table ON tpv.id = graph_index_table.id;
                            
                            INSERT INTO L_table SELECT neighbor AS id, VEC_DISTANCE_EUCLIDEAN(data_table.data, query_table.data) AS dist, tp2.batch FROM data_table, query_table, tp2 WHERE tp2.neighbor = data_table.id AND query_table.num = tp2.batch;

                            INSERT INTO V_table SELECT id,batch FROM tpv;
                            DROP TEMPORARY TABLE tp1;
                            DROP TEMPORARY TABLE tp2;
                            DROP TEMPORARY TABLE tpv;

                            CREATE TEMPORARY TABLE tp_table AS SELECT d.id, d.dist, d.batch FROM (SELECT l1.*, row_number() OVER (PARTITION BY batch ORDER BY dist) AS class_rank FROM (SELECT DISTINCT * FROM L_table) l1) d WHERE d.class_rank <= L ORDER BY batch, dist;
                            DROP TEMPORARY TABLE L_table;
                            ALTER TABLE tp_table RENAME TO L_table;
                        END WHILE;

                        SELECT JSON_ARRAYAGG(d.id), d.batch FROM (SELECT l1.*, row_number() OVER (PARTITION BY batch ORDER BY dist) AS rank FROM L_table l1) d WHERE d.rank <= R GROUP BY d.batch ORDER BY d.batch;

                        DROP TEMPORARY TABLE IF EXISTS L_table;
                        DROP TEMPORARY TABLE IF EXISTS V_table;
                    END ;''')

    start_time_total = time.time()
    cur.execute("CALL batchsearch(123742,0,100,100);")   
    end_time_total = time.time()        
    print(cur.fetchall()[0])
    print("Total Time Taken: " + str(end_time_total - start_time_total))
    print("QPS: " + str(1/(end_time_total - start_time_total)))

    conn.close()
    
    

if __name__ == '__main__':
    main()
