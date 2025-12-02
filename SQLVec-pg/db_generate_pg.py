import psycopg2
import time
import numpy as np
import pandas as pd
import pickle
import h5py
from io import StringIO
import os
import argparse


def fvecs_read(dim, filename, c_contiguous=True, record_count=-1, line_offset=0, record_dtype=np.int32):
    if record_count > 0:
        record_count *= dim + 1
    if line_offset > 0:
        line_offset *= (dim + 1) * 4
    fv = np.fromfile(filename, dtype=record_dtype, count=record_count, offset=line_offset)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    #print(dim)
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def main(args):
    conn = psycopg2.connect(database=args.db, user="postgres", password="pw", host="127.0.0.1", port="5432")
    conn.autocommit = True

    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    # --------------------------------------------------
    # create tables
    cur.execute(f'CREATE TABLE IF NOT EXISTS data_table (id INTEGER PRIMARY KEY ,data VECTOR({args.dim}));')  
    cur.execute(f'CREATE TABLE IF NOT EXISTS query_table (num INTEGER PRIMARY KEY, data VECTOR({args.dim}));') 
    cur.execute('CREATE TABLE IF NOT EXISTS graph_index_table (id INTEGER PRIMARY KEY, nbs INTEGER[]);') 

    # --------------------------------------------------
    # load graph index 
    cur.execute(f"COPY graph_index_table (id, nbs) FROM '{args.index_path}' WITH (FORMAT CSV, HEADER);")

    # --------------------------------------------------
    # load query_table 
    if not args.queryset_path:
        raise ValueError("No queryset_path provided.")
    file_ext = os.path.splitext(args.queryset_path)[1].lower()
    if file_ext == ".csv":
        cur.execute(f"COPY query_table (num, data) FROM '{args.queryset_path}' WITH (FORMAT CSV, HEADER);")
    else:
        if file_ext == ".fvecs":
            Q = fvecs_read(args.dim, args.queryset_path, record_dtype=np.float32) 
            Q = pd.DataFrame({'data': [row.tolist() for row in Q]})
            Q['num'] = range(len(Q))
            Q = Q[['num', 'data']]
            Q['data'] = Q['data'].apply(list)
           
        elif file_ext == ".hdf5" or file_ext == ".h5":
            with h5py.File(args.queryset_path, 'r') as f:
                print("Keys in the file:", list(f.keys()))
                # Keys in the file: ['distances', 'neighbors', 'test', 'train'] 
                Q = f['test'][:]
            Q = pd.DataFrame({'data': [row.tolist() for row in Q]})
            Q['num'] = range(len(Q))
            Q = Q[['num', 'data']]
            Q['data'] = Q['data'].apply(list)

        elif file_ext == ".pkl" or file_ext == ".pickle":
            with open(args.queryset_path,'rb') as f:
                Q = pickle.load(f)
            Q['data'] = Q['data'].apply(list)

        output = StringIO()
        Q.to_csv(output, sep='|', header=False, index=False)
        output.seek(0)
        cur.copy_from(output, 'query_table', sep='|', columns=('num', 'data'))

    # --------------------------------------------------
    # load data_table 
    if not args.dataset_path:
        raise ValueError("No dataset_path provided.")
    file_ext = os.path.splitext(args.dataset_path)[1].lower()
    if file_ext == ".csv":
        cur.execute(f"COPY data_table (id, data) FROM '{args.dataset_path}' WITH (FORMAT CSV, HEADER);")
    else:
        if file_ext == ".fvecs":
            data = fvecs_read(args.dim, args.dataset_path, record_dtype=np.float32) 
            data = pd.DataFrame({'data': [row.tolist() for row in data]})
            data['id'] = range(len(data))
            data = data[['id', 'data']]

        elif file_ext == ".hdf5" or file_ext == ".h5":
            with h5py.File(args.dataset_path, 'r') as f:
                print("Keys in the file:", list(f.keys()))
                # Keys in the file: ['distances', 'neighbors', 'test', 'train'] 
                data = f['train'][:]
            data = pd.DataFrame({'data': [row.tolist() for row in data]})
            data['id'] = range(len(data))
            data = data[['id', 'data']]

        elif file_ext == ".pkl" or file_ext == ".pickle":
            with open(args.dataset_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
 
        chunk_size = 10000 
        total_rows = len(data)
        start_idx = 0
        while start_idx < total_rows:
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = data[start_idx:end_idx]  
            chunk['data'] = chunk['data'].apply(list)
            output2 = StringIO()
            chunk.to_csv(output2, sep='|', header=False, index=False)
            output2.seek(0)
            cur.copy_from(output2, 'data_table', sep='|', columns=('id', 'data'))
            print(f"Inserted rows: {start_idx} to {end_idx}")
            start_idx = end_idx


    # --------------------------------------------------
    # to add filtered attribute 
    cur.execute('ALTER TABLE data_table ADD COLUMN a1 INTEGER;')
    cur.execute('ALTER TABLE query_table ADD COLUMN a1 INTEGER;')

    cur.execute(''' WITH numbered_rows AS (SELECT id, row_number() OVER (ORDER BY id) AS rn FROM data_table)   
                UPDATE data_table
                SET a1 = (rn - 1) % 24 + 1
                FROM numbered_rows
                WHERE data_table.id = numbered_rows.id;''') 
    
    cur.execute("UPDATE query_table SET a1 = FLOOR(random() * 24) + 1;") 

    cur.execute("CREATE INDEX d_a1_index ON public.data_table USING btree (a1);")
    cur.execute("CREATE INDEX q_a1_index ON public.query_table USING btree (a1);")

    cur.close()
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='batched query-optimized')
    parser.add_argument('--db', type=str, default='sift')
    parser.add_argument('--dim', type=int, default=128, help='the dimension of vector')
    parser.add_argument('--index_path', type=str, help='the path of index file')
    parser.add_argument('--dataset_path', type=str, help='the path of data file')
    parser.add_argument('--queryset_path', type=str, help='the path of query file')
    args = parser.parse_args()
    main(args)
