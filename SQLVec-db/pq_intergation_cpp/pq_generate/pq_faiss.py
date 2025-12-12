import numpy as np
import pandas as pd
import faiss
import argparse

N_DIM = 128
DATA_SIZE = 4

def fvecs_read(filename, c_contiguous=True, record_count=-1, line_offset=0, record_dtype=np.int32):
    if record_count > 0:
        record_count *= N_DIM + 1
    if line_offset > 0:
        line_offset *= (N_DIM + 1) * DATA_SIZE
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

def save_codes_to_csv(codes_array, filename):
    indices = np.arange(codes_array.shape[0])

    df = pd.DataFrame({
        'id': indices,
        'data': [row.tolist() for row in codes_array]
    })
    df.to_csv(filename, index=False)
    print(f"save to  {filename}")


def main(args):
    ksub = 1 << args.n 
    dsub = args.dim // args.m

    base = fvecs_read(args.dataset_path, record_dtype=np.float32) 

    pq = faiss.ProductQuantizer(args.dim, args.m, args.n)
    pq.train(base)

    codes = pq.compute_codes(base)     

    query = fvecs_read(args.queryset_path, record_dtype=np.float32)
    query_codes = pq.compute_codes(query)
    
    centroids = faiss.vector_to_array(pq.centroids).reshape(args.m, ksub, dsub)

    # build lookup table
    lookup_table = np.zeros((args.m, ksub, ksub), dtype='float32')
    for m in range(args.m):
        A = centroids[m]  
        lookup_table[m] = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)

    lookup_table_1d = lookup_table.flatten()  
    lookup_table_1d.tofile(args.lookup_path_save)

    save_codes_to_csv(codes, args.dataset_path_save)
    save_codes_to_csv(query_codes, args.queryset_path_save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate duckdb database for SQLVec')
    parser.add_argument('--dim', type=int, default=960, help='dimension of vectors')
    parser.add_argument('--m', type=int, default=64, help='number of subspaces')
    parser.add_argument('--n', type=int, default=8, help='number of bits per subspace')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--queryset_path', type=str, required=True)
    parser.add_argument('--dataset_path_save', type=str, required=True)
    parser.add_argument('--queryset_path_save', type=str, required=True)
    parser.add_argument('--lookup_path_save', type=str, required=True)
    args = parser.parse_args()

    main(args)
