import numpy as np
import pandas as pd
import pickle
import duckdb
import os
import argparse
import h5py

def bvecs_read(fname):
    a = np.fromfile(fname, dtype=np.int32, count=1)
    b = np.fromfile(fname, dtype=np.uint8)
    d = a[0]
    return b.reshape(-1, d + 4)[:, 4:].copy()

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
    # We import the dataset, the query set, and the graph index into a single database.
    # build the connection and database
    con = duckdb.connect(args.db_path) 
    # --------------------------------------------------
    # to load proximity graph index
    if not args.index_path:
        raise ValueError("No index_path provided.")
    G_point = pd.read_csv(args.index_path)
    G = con.from_df(G_point)
    con.execute('CREATE TABLE temp_table AS SELECT * FROM G') 
    con.execute('CREATE TABLE Graph_index_table AS select id, cast(neighbor as int[]) as nbs from temp_table')
    con.execute('DROP TABLE temp_table') 
    
    # --------------------------------------------------
    # to load query_table
    if not args.queryset_path:
        raise ValueError("No queryset_path provided.")
    file_ext = os.path.splitext(args.queryset_path)[1].lower()

    if file_ext == ".bvecs":
        query = bvecs_read(args.queryset_path)
        query = pd.DataFrame({'data': [row.tolist() for row in query]})
        query['num'] = range(len(query))
        query = query[['num', 'data']]
    elif file_ext == ".fvecs":
        query = fvecs_read(args.queryset_path, record_dtype=np.float32)
        query = pd.DataFrame({'data': [row.tolist() for row in query]})
        query['num'] = range(len(query))
        query = query[['num', 'data']]
    elif file_ext == ".hdf5" or file_ext == ".h5":
        with h5py.File(args.queryset_path, 'r') as f:
            print("Keys in the file:", list(f.keys()))
            # Keys in the file: ['distances', 'neighbors', 'test', 'train'] 
            query = f['test'][:]
        query = pd.DataFrame({'data': [row.tolist() for row in query]})
        query['num'] = range(len(query))
        query = query[['num', 'data']]
    elif file_ext == ".pickle" or file_ext == ".pkl":
        with open(args.queryset_path, 'rb') as f:
            query = pickle.load(f)
            

    con.execute('CREATE TABLE temp_table AS SELECT * FROM query') 
    sql1 = 'CREATE TABLE query_table AS select num, cast(data as FLOAT[' + str(args.data_dim )+']) as data from temp_table'
    con.execute(sql1) 
    con.execute('DROP TABLE temp_table')
    
    # --------------------------------------------------
    # to load data_table
    if not args.dataset_path:
        raise ValueError("No dataset_path provided.")
    file_ext = os.path.splitext(args.dataset_path)[1].lower()

    if file_ext == ".bvecs":
        base = bvecs_read(args.dataset_path)
        base = pd.DataFrame({'data': [row.tolist() for row in base]})
        base['id'] = range(len(base))
        base = base[['id', 'data']]
    elif file_ext == ".fvecs":
        base = fvecs_read(args.dataset_path, record_dtype=np.float32)
        base = pd.DataFrame({'data': [row.tolist() for row in base]})
        base['id'] = range(len(base))
        base = base[['id', 'data']]
    elif file_ext == ".hdf5" or file_ext == ".h5":
        with h5py.File(args.dataset_path, 'r') as f:
            print("Keys in the file:", list(f.keys()))
            # Keys in the file: ['distances', 'neighbors', 'test', 'train'] 
            base = f['train'][:]
        base = pd.DataFrame({'data': [row.tolist() for row in base]})
        base['id'] = range(len(base))
        base = base[['id', 'data']] 
    elif file_ext == ".pickle" or file_ext == ".pkl":
        with open(args.dataset_path, 'rb') as f:
            base = pickle.load(f)

    con.execute('CREATE TABLE temp_table AS SELECT * FROM base') 
    sql2 = 'CREATE TABLE data_table AS select id, cast(data as FLOAT[' + str(args.data_dim )+']) as data from temp_table'
    con.execute(sql2) 
    con.execute('DROP TABLE temp_table')

    # --------------------------------------------------
    # check index sizes
    con.sql('PRAGMA database_size;').show()

    # --------------------------------------------------
    # add filtered attribute 
    con.execute("ALTER TABLE data_table ADD COLUMN a1 INTEGER;") 
    # here we generate attribute a1 with values from 1 to 24
    con.execute(''' WITH numbered_rows AS (SELECT id, row_number() OVER (ORDER BY id) AS rn FROM data_table)   
                    UPDATE data_table
                    SET a1 = (rn - 1) % 24 + 1
                    FROM numbered_rows
                    WHERE data_table.id = numbered_rows.id;''') 
    
    con.execute("ALTER TABLE query_table ADD COLUMN a1 INTEGER;") 
    con.execute("UPDATE query_table SET a1 = FLOOR(random() * 24) + 1;") 

    con.execute('CREATE INDEX d_id_index ON data_table(id)')
    con.execute('CREATE INDEX q_a1_index ON query_table(a1)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate duckdb database for SQLVec')
    parser.add_argument('--db_path', type=str, default=':memory:')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--data_dim', type=int, default=128)
    parser.add_argument('--queryset_path', type=str, required=True)
    parser.add_argument('--index_path', type=str, required=True)
    args = parser.parse_args()

    main(args)

