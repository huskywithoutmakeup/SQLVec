# kuzu-0.8.2
import numpy as np
import pandas as pd
import pickle
import time
import kuzu
import argparse
import os   
import h5py


def main(args):
    db = kuzu.Database(args.db_path)
    con = kuzu.Connection(db)

    # create tables
    con.execute(f"CREATE NODE TABLE data_table(id INT64, data FLOAT[{args.dim}], PRIMARY KEY (id))")
    con.execute(f"CREATE NODE TABLE query_table(num INT64, data FLOAT[{args.dim}], PRIMARY KEY (num))")
    con.execute("CREATE REL TABLE graph_index_table(FROM data_table TO data_table)")
    
    '''
    # --------------------------------------------------
    # query table
    '''
    file_ext = os.path.splitext(args.queryset_path)[1].lower()

    if file_ext == ".pickle" or file_ext == ".pkl":
        with open(args.queryset_path, 'rb') as f:
            query = pickle.load(f)
        query['data'] = [list(item) for item in query['data']]
        con.execute("COPY query_table FROM query")

        # check info
        res = con.execute("MATCH (n:query_table) RETURN n.num, n.data;")
        print(res.get_as_df())
        res = con.execute("CALL TABLE_INFO('query_table') RETURN *;")
        print(res.get_as_df())
    else:
        raise ValueError("Unsupported queryset_path file extension. Need .pickle or .pkl")

    '''
    # --------------------------------------------------
    # data table
    '''
    file_ext = os.path.splitext(args.dataset_path)[1].lower()

    if file_ext == ".pickle" or file_ext == ".pkl":
        with open(args.dataset_path, 'rb') as f:
            base = pickle.load(f)
        base['data'] = [list(item) for item in base['data']]
        con.execute("COPY data_table FROM base")

        # check info
        res = con.execute("MATCH (n:data_table) RETURN n.id, n.data;")
        print(res.get_as_df())
        res = con.execute("CALL TABLE_INFO('data_table') RETURN *;")
        print(res.get_as_df())
    else:
        raise ValueError("Unsupported dataset_path file extension. Need .pickle or .pkl")

    '''
    # --------------------------------------------------
    # graph index table  
    '''
    con.execute(f'''COPY graph_index_table FROM '{args.index_path}';''')

    # check info
    res = con.execute("MATCH ()-[r:graph_index_table]->() RETURN COUNT(*);")
    print(res.get_as_df())

    res = con.execute("CALL TABLE_INFO('graph_index_table') RETURN *;")
    print(res.get_as_df())

    # final check
    res = con.execute("MATCH (d1:data_table)-[:graph_index_table]->(d2:data_table) WHERE d1.id = 123472 RETURN d2.id;")
    print(res.get_as_df())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate kuzu database for SQLVec')
    parser.add_argument('--db_path', type=str, default='./demo')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--queryset_path', type=str, required=True)
    parser.add_argument('--index_path', type=str, required=True)
    args = parser.parse_args()

    main(args)

