
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import pickle
import time
import duckdb
import os
import struct
import argparse


N_DIM = 128
DATA_SIZE = 4


def read_fbin_at_offset(filename, offset):
    with open(filename, "rb") as f:
        f.seek(offset)
        header = f.read(8)
        if len(header) < 8:
            raise ValueError("EOF reached unexpectedly")
        npts, dim = struct.unpack("II", header)
        print(f"At offset {offset}: npts={npts}, dim={dim}")
        total = npts * dim
        if npts != 241:
            data = np.fromfile(f, dtype=np.float32, count=total)
        else:
            data = np.fromfile(f, dtype=np.uint32, count=total)
        if data.size != total:
            raise ValueError("Unexpected EOF when reading data")
        data = data.reshape(npts, dim)
    return npts, dim, data


def pq_quantize_vectors(vectors, pq_centroids, chunk_dim_map, chunk_boundaries, num_pq_chunks=64, num_centers=256):
    num_vectors, dim = vectors.shape
    
    # Validate shapes of inputs
    assert pq_centroids.shape == (num_centers, dim), "Centroids shape should be (%d, %d)" % (num_centers, dim)
    assert chunk_dim_map.shape == (dim,), f"chunk_dim_map shape should be ({dim},)"
    assert chunk_boundaries[-1] == dim, "The last boundary must equal the vector dimension"

    centered_vectors = vectors - chunk_dim_map

    # 2. Initialize PQ codes (one byte per subspace)
    pq_codes = np.zeros((num_vectors, num_pq_chunks), dtype=np.uint8)

    # 3. Iterate over each subspace to perform quantization
    for i in range(num_pq_chunks):  # iterate over subspaces
        # Get the dimension range for the current subspace
        start_dim = chunk_boundaries[i]
        end_dim = chunk_boundaries[i + 1]
        chunk_size = end_dim - start_dim

        # Extract centroids for the current subspace: slice of pq_centroids
        subspace_centroids = pq_centroids[:, start_dim:end_dim]  # shape [num_centers, chunk_size]

        # Extract vectors for the current subspace
        subspace_vectors = centered_vectors[:, start_dim:end_dim]  # shape [num_vectors, chunk_size]

        # Compute nearest centroid for each vector in this subspace (vectorized)
        vec_norms = np.sum(subspace_vectors**2, axis=1, keepdims=True)  # [num_vectors, 1]
        centroid_norms = np.sum(subspace_centroids**2, axis=1, keepdims=True)  # [num_centers, 1]
        dot_products = np.dot(subspace_vectors, subspace_centroids.T)  # [num_vectors, num_centers]

        distances = vec_norms + centroid_norms.T - 2 * dot_products  # [num_vectors, num_centers]
        closest_indices = np.argmin(distances, axis=1)  # [num_vectors]

        pq_codes[:, i] = closest_indices

    return pq_codes

def bvecs_read(fname):
    a = np.fromfile(fname, dtype=np.int32, count=1)
    b = np.fromfile(fname, dtype=np.uint8)
    d = a[0]
    return b.reshape(-1, d + 4)[:, 4:].copy()

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

def main(args):
    con = duckdb.connect(args.db_path) 

    if args.bin_path is not None:
        with open(args.bin_path, "rb") as f:
            npts = np.fromfile(f, dtype=np.uint32, count=1)[0]
            dim = np.fromfile(f, dtype=np.uint32, count=1)[0]
            print(f"Header: npts = {npts}, dim = {dim}") 
            base_codes = np.fromfile(f, dtype=np.uint8).reshape((npts, dim))

        print("PQ codes shape:", base_codes.shape)

        offsets = [4096, 987144, 990992] # , 991260

        blocks = []
        for off in offsets:
            npts, dim, data = read_fbin_at_offset(args.pivots_path, off)
            blocks.append((npts, dim, data))

        pq_centroids     = blocks[0][2]
        chunk_dim_map    = blocks[1][2]
        chunk_boundaries = blocks[2][2]

        print(f"PQ Centroids shape:     {pq_centroids.shape}")
        print(f"Chunk Dim Map shape:    {chunk_dim_map.shape}")
        print(f"Chunk Boundaries shape: {chunk_boundaries.shape}")

        query = fvecs_read(args.queryset_path, record_dtype=np.float32)
        query_codes = pq_quantize_vectors(query, pq_centroids, chunk_dim_map.flatten(), chunk_boundaries.flatten(),num_pq_chunks=base_codes.shape[1])

        query_codes = pd.DataFrame({'data': [row.tolist() for row in query_codes]})
        query_codes['num'] = range(len(query_codes))
        query_codes = query_codes[['num', 'data']]

        base_codes = pd.DataFrame({'data': [row.tolist() for row in base_codes]})
        base_codes['id'] = range(len(base_codes))
        base_codes = base_codes[['id', 'data']]
    else:
        q = pd.read_csv(args.queryset_path)
        query_codes = con.from_df(q)
        b = pd.read_csv(args.dataset_path)
        base_codes = con.from_df(b)

    # --------------------------------------------------      
    # generate database

    G_point = pd.read_csv(args.index_path)
    G = con.from_df(G_point)
    con.execute('CREATE TABLE temp_table AS SELECT * FROM G') 
    con.execute('CREATE TABLE Graph_index_table AS select id, cast(neighbor as int[]) as nbs from temp_table')
    con.execute('DROP TABLE temp_table') 
    
    # --------------------------------------------------      
    con.execute('CREATE TABLE temp_table AS SELECT * FROM query_codes') 
    con.execute(f'CREATE TABLE query_table AS select num, cast(data as INT[{args.data_dim}]) as data from temp_table') 
    con.execute('DROP TABLE temp_table')

    con.execute('CREATE TABLE temp_table AS SELECT * FROM base_codes') 
    con.execute(f'CREATE TABLE data_table AS select id, cast(data as INT[{args.data_dim}]) as data from temp_table') 
    con.execute('DROP TABLE temp_table')

    # index
    con.execute('CREATE INDEX g_id_index ON Graph_index_table(id)')
    con.execute('CREATE INDEX d_id_index ON data_table(id)')
    con.execute('CREATE INDEX q_id_index ON query_table(num)')

    # check
    con.sql("SELECT * FROM data_table LIMIT 2").show()
    con.sql("SELECT * FROM query_table LIMIT 2").show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SQLVec_pq_database_generate')
    parser.add_argument('--db_path', type=str, default=':memory:')
    parser.add_argument('--bin_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--data_dim', type=int, default=240)
    parser.add_argument('--queryset_path', type=str)
    parser.add_argument('--pivots_path', type=str)
    parser.add_argument('--index_path', type=str, required=True)
    args = parser.parse_args()

    main(args)
