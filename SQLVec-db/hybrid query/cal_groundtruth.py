import numpy as np
import pandas as pd
import time
import duckdb
import struct
import argparse
import os

def main(args):
    con = duckdb.connect(args.db)
    con.execute("PRAGMA enable_optimizer") 

    q = con.execute('SELECT data, a1 FROM query_table;').fetchall()

    res = []
    start_time_total = time.time()
    for i in range(args.querysize):
        res.append(con.execute("""
                        SELECT id 
                        FROM data_table 
                        where data_table.a1 = ?
                        ORDER BY array_distance(data_table.data, ?::FLOAT[128])  
                        limit ?"""
            ,parameters=[q[i][1],q[i][0],args.k]).fetchnumpy()['id'])
    end_time_total = time.time()
    print("QPS: " + str(args.querysize/(end_time_total-start_time_total)))

    np.save(args.gt_path, res)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate groundtruth on filtered search')
    parser.add_argument('--db', type=str, required=True)
    parser.add_argument('--querysize', type=int, default=10000, help='the query size of datasets') 
    parser.add_argument('--k', type=int, default=100, help='top-k search, k number')
    parser.add_argument('--gt_path', type=str, help='the path of groundtruth file', required=True)
    args = parser.parse_args()

    main(args)
