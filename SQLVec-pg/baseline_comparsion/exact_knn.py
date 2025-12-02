import psycopg2
import time
import numpy as np
import argparse


def main(args):
    conn = psycopg2.connect(database=args.db, user="postgres", password="pw", host="127.0.0.1", port="5432")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # to load query vectors
    cur.execute('SELECT data FROM query_table;')
    q = cur.fetchall() 

    # to get backend pid for futher profiling
    cur.execute("SELECT pg_backend_pid();")
    backend_pid = cur.fetchone()[0]
    print(f"PostgreSQL backend id (PID): {backend_pid}")

    time.sleep(30)
    
    start_time_total = time.time()
    for i in range(args.querysize):
        cur.execute('SELECT id FROM data_table ORDER BY (%s <-> data) LIMIT %s',(q[i][0],args.k))
    end_time_total = time.time()

    print("Total Time Taken: " + str(end_time_total - start_time_total))
    print("QPS: " + str(1/(end_time_total - start_time_total)*args.querysize))

    cur.close()
    conn.close()

    print("done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exact-KNN')
    parser.add_argument('--db', type=str, default='sift')
    parser.add_argument('--k', type=int, default=100, help='top-k search, k number')
    parser.add_argument('--querysize', type=int, default=10000, help='the query size of datasets') 
    args = parser.parse_args()

    main(args)