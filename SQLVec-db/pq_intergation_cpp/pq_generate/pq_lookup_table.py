
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
        if npts != 65:
            data = np.fromfile(f, dtype=np.float32, count=total)
        else:
            data = np.fromfile(f, dtype=np.uint32, count=total)
        if data.size != total:
            raise ValueError("Unexpected EOF when reading data")
        data = data.reshape(npts, dim)
    return npts, dim, data


def pq_quantize_vectors(vectors, pq_centroids, chunk_dim_map, chunk_boundaries, num_pq_chunks=64, num_centers=256):
    num_vectors, dim = vectors.shape
    
    # 验证数据形状
    assert pq_centroids.shape == (num_centers, dim), "质心形状应为 (%d, %d)" % (num_centers, dim)
    assert chunk_dim_map.shape == (dim,), f"全局质心形状应为 ({dim},)"
    assert chunk_boundaries[-1] == dim, "最后一个边界必须等于向量维度"
    
    centered_vectors = vectors - chunk_dim_map 

    # 2. 初始化PQ编码（每个子空间对应一个字节）
    pq_codes = np.zeros((num_vectors, num_pq_chunks), dtype=np.uint8)
    
    # 3. 遍历每个子空间进行量化
    for i in range(num_pq_chunks): # 64个子空间
        # 获取当前子空间的维度范围
        start_dim = chunk_boundaries[i]
        end_dim = chunk_boundaries[i + 1]
        chunk_size = end_dim - start_dim
        
        # 提取当前子空间的质心：pq_centroids的第i个子空间维度范围
        subspace_centroids = pq_centroids[:, start_dim:end_dim]  # 形状 [256, chunk_size]
        
        # 提取当前子空间的向量数据
        subspace_vectors = centered_vectors[:, start_dim:end_dim]  # 形状 [num_vectors, chunk_size]
        
        # 计算每个向量在当前子空间的最近质心（向量化计算）
        vec_norms = np.sum(subspace_vectors**2, axis=1, keepdims=True)  # [num_vectors, 1]
        centroid_norms = np.sum(subspace_centroids**2, axis=1, keepdims=True)  # [256, 1]
        dot_products = np.dot(subspace_vectors, subspace_centroids.T)  # [num_vectors, 256]
        
        distances = vec_norms + centroid_norms.T - 2 * dot_products  # [num_vectors, 256]
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
    with open(args.data_bin_path, "rb") as f:
        npts = np.fromfile(f, dtype=np.uint32, count=1)[0]
        dim = np.fromfile(f, dtype=np.uint32, count=1)[0]
        print(f"Header: npts = {npts}, dim = {dim}")  
        base_codes = np.fromfile(f, dtype=np.uint8).reshape((npts, dim))

    print("PQ codes shape:", base_codes.shape)

    offsets = [4096, 987144, 990992] 

    blocks = []
    for off in offsets:
        npts, dim, data = read_fbin_at_offset(args.pivots_path, off)
        blocks.append((npts, dim, data))

    # 命名解包
    pq_centroids     = blocks[0][2]
 
    ksub = 1 << args.n
    dsub = args.dim // args.m
    centroids = pq_centroids.reshape(ksub, args.m, dsub).transpose(1, 0, 2) # 
    
    # build 
    lookup_table = np.zeros((args.m, ksub, ksub), dtype='float32')
    for m in range(args.m):
        A = centroids[m]  
        lookup_table[m] = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)

    # 展平成 1D 数组并保存
    lookup_table_1d = lookup_table.flatten()  
    lookup_table_1d.tofile(args.lookup_path_save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate duckdb database for SQLVec')
    parser.add_argument('--dim', type=int, default=960, help='dimension of vectors')
    parser.add_argument('--m', type=int, default=64, help='number of subspaces')
    parser.add_argument('--n', type=int, default=8, help='number of bits per subspace')
    parser.add_argument('--data_bin_path', type=str, required=True)
    parser.add_argument('--pivots_path', type=str, required=True)
    parser.add_argument('--lookup_path_save', type=str, required=True)
    args = parser.parse_args()

    main(args)
