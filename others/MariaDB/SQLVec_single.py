import mariadb 
import time
import numpy as np
import pandas as pd
import pickle

def main():
    conn = mariadb.connect(database="test", user="root", password="", host="localhost", port=3306, unix_socket="/tmp/mysql.sock")
    conn.autocommit = True

    cur = conn.cursor()
    cur.execute("DROP PROCEDURE IF EXISTS singlesearch") 
    cur.execute('''
                    CREATE PROCEDURE singlesearch(IN enter_node_id INT, IN query_node_num INT, IN L INT, IN R INT)
                    BEGIN
                        DECLARE enter_node vector(128);
                        DECLARE query_node vector(128);
                        DECLARE cpoint_id INT;
                        DECLARE cnt INT DEFAULT 9;

                        SELECT data INTO enter_node FROM data_table WHERE id = enter_node_id;
                        SELECT data INTO query_node FROM query_table WHERE num = query_node_num;

                        CREATE TEMPORARY TABLE L_table (id INT, dist DOUBLE);
                        CREATE TEMPORARY TABLE V_table (id INT);

                        INSERT INTO L_table SELECT enter_node_id, VEC_DISTANCE_EUCLIDEAN(enter_node, query_node) AS dist;

                        WHILE cnt != 0 DO
                            CREATE TEMPORARY TABLE tp1 AS SELECT L_table.* FROM L_table LEFT JOIN V_table ON L_table.id = V_table.id WHERE V_table.id IS NULL;
                            SELECT id INTO cpoint_id FROM tp1 WHERE dist = (SELECT MIN(dist) FROM tp1);
                            SELECT COUNT(*) INTO cnt FROM tp1;
                            DROP TEMPORARY TABLE IF EXISTS tp1;

                            INSERT INTO L_table
                            SELECT id, VEC_DISTANCE_EUCLIDEAN(data, query_node) AS dist
                            FROM data_table
                            WHERE id IN (SELECT neighbor FROM graph_index_table WHERE id = cpoint_id);

                            INSERT INTO V_table VALUES (cpoint_id);

                            CREATE TEMPORARY TABLE tp_table AS SELECT DISTINCT id, dist FROM L_table ORDER BY dist LIMIT L;             
                            TRUNCATE TABLE L_table;
                            INSERT INTO L_table SELECT * FROM tp_table;
                            DROP TEMPORARY TABLE tp_table;
                        END WHILE;

                        SELECT JSON_ARRAYAGG(id) FROM L_table LIMIT R;

                        DROP TEMPORARY TABLE IF EXISTS L_table;
                        DROP TEMPORARY TABLE IF EXISTS V_table;
                    END ;''')
    
    
    start_time_total = time.time()
    cur.execute("CALL singlesearch(123742,0,100,100);")   
    end_time_total = time.time()        
    print(cur.fetchall())
    print("Total Time Taken: " + str(end_time_total - start_time_total))
    print("QPS: " + str(1/(end_time_total - start_time_total)))

    conn.close()
    

if __name__ == '__main__':
    main()
