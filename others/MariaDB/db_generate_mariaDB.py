import mariadb 
import time
import numpy as np
import pandas as pd
import pickle
import argparse

def main(args):
    conn = mariadb.connect(database=args.db, user="root", password="", host="localhost", port=3306, unix_socket="/tmp/mysql.sock")
    conn.autocommit = True

    cur = conn.cursor()

    cur.execute("CREATE TABLE graph_index_table(id INT,neighbor INT);")
    cur.execute(f"CREATE TABLE data_table(id INT,data vector({args.dim}),a1 INT);")
    cur.execute(f"CREATE TABLE query_table(num INT,data vector({args.dim}),a1 INT);")

    # --------------------------------------------------
    # load index
    cur.execute(f''' LOAD DATA INFILE '{args.index_path}'
                    INTO TABLE graph_index_table
                    FIELDS TERMINATED BY ',' 
                    LINES TERMINATED BY '\n'
                    IGNORE 1 LINES
                    (id, neighbor);''')

    cur.execute("DROP TABLE IF EXISTS tp1;")

    # --------------------------------------------------
    # load queryset
    cur.execute("CREATE TABLE tp1(num INT,tx TEXT,a1 INT);")
    cur.execute(f''' LOAD DATA INFILE '{args.queryset_path}'
                    INTO TABLE tp1
                    FIELDS TERMINATED BY '.' 
                    LINES TERMINATED BY '\n'
                    IGNORE 1 LINES
                    (num,tx, a1);''')
    cur.execute("INSERT INTO query_table SELECT num,VEC_FromText(tx) as data,a1 FROM tp1;") # transfer str to vector
    cur.execute("DROP TABLE IF EXISTS tp1;")

    # --------------------------------------------------
    # load dataset
    cur.execute("CREATE TABLE tp1(id INT,tx TEXT,a1 INT);")
    cur.execute(f''' LOAD DATA INFILE '{args.dataset_path}'
                    INTO TABLE tp1
                    FIELDS TERMINATED BY '.' 
                    LINES TERMINATED BY '\n'
                    IGNORE 1 LINES
                    (id, tx, a1);''')
    cur.execute("INSERT INTO data_table SELECT id,VEC_FromText(tx) as data,a1 FROM tp1;") # transfer str to vector


    # build index 
    cur.execute("ALTER TABLE data_table ADD PRIMARY KEY (id);")
    cur.execute("CREATE INDEX d_id_idx ON data_table (id);")
    cur.execute("CREATE INDEX d_a1_idx ON data_table (a1);")

    cur.execute("CREATE INDEX g_id_idx ON graph_index_table (id);")
    cur.execute("CREATE INDEX g_nb_idx ON graph_index_table (neighbor);")

    cur.execute("ALTER TABLE query_table ADD PRIMARY KEY (num);")
    cur.execute("CREATE INDEX q_id_idx ON query_table (num);")
    cur.execute("CREATE INDEX q_a1_idx ON query_table (a1);")

    conn.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mariaDB database generation for SQLVec')
    parser.add_argument('--db', type=str, default='sqlvec', help='database name')
    parser.add_argument('--dim', type=int, default=128, help='dimension of vector')
    parser.add_argument('--queryset_path', type=str, required=True, help='path to the query set file')
    parser.add_argument('--dataset_path', type=str, required=True, help='path to the dataset file')
    parser.add_argument('--index_path', type=str, required=True, help='path to the index file')
    args = parser.parse_args()
    main(args)
