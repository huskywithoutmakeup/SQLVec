import time
import duckdb
import argparse
import pickle

def main(args):
    con = duckdb.connect(args.db_path) 
    con.execute("PRAGMA enable_optimizer") 
   
    queries = con.execute("SELECT data,a1 FROM query_table;").fetchall() 
    dim = len(queries[0][0])
    sql = 'SELECT id, array_distance(data,?::FLOAT[' + str(dim) + ']) as dist FROM data_table WHERE a1 = ? ORDER BY dist LIMIT '+str(args.k)

    res = []
    # dont use vector index, just filter and exact knn
    start_time_total = time.time()
    for i in range(args.querysize): 
        res.append(con.execute(sql,parameters=[queries[i][0],queries[i][1]]).fetchnumpy()['id'])
    end_time_total = time.time()

    print("Total Time Taken: " + str(end_time_total - start_time_total))
    print("QPS: " + str(1/(end_time_total - start_time_total)*args.querysize))

    # save res to the groudtruth file
    with open(args.gt_path, 'wb') as f:
        pickle.dump(res, f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='baseline_pre_filter')
    parser.add_argument('--db_path', type=str)
    parser.add_argument('--k', type=int, default=100, help='top-k search, k number')
    parser.add_argument('--querysize', type=int, default=10000, help='the query size of datasets') 
    parser.add_argument('--gt_path', type=str, default='./groundtruth/gt.pkl', help='the path of groundtruth file') 
    args = parser.parse_args()

    main(args)