import struct
import pandas as pd
import numpy as np
import argparse
import os

def load_nsg(filename):
    final_graph = []
    with open(filename, "rb") as file:
        width = struct.unpack("I", file.read(4))[0]
        ep_ = struct.unpack("I", file.read(4))[0]
        while True:
            k_bytes = file.read(4)
            if not k_bytes:
                break
            k = struct.unpack("I", k_bytes)[0]
            tmp = struct.unpack(f"{k}I", file.read(k * 4))
            final_graph.append(list(tmp))

    return final_graph

def load_impl(filename):
    graph = []
    with open(filename, 'rb') as f:
        # Read metadata
        max_range_of_graph = 64 
        expected_file_size = struct.unpack('Q', f.read(8))[0]
        max_observed_degree = struct.unpack('I', f.read(4))[0]
        start = struct.unpack('I', f.read(4))[0]
        file_frozen_pts = struct.unpack('Q', f.read(8))[0]
        vamana_metadata_size = 8 + 4 + 4 + 8
        
        # Read graph data
        bytes_read = vamana_metadata_size
        cc = 0
        nodes_read = 0
        while bytes_read != expected_file_size:
            k = struct.unpack('I', f.read(4))[0]
            if k == 0:
                print("ERROR: Point found with no out-neighbours, point#", nodes_read)
            cc += k
            nodes_read += 1
            tmp = struct.unpack(f'{k}I', f.read(k * 4))
            graph.append(list(tmp))
            bytes_read += (k + 1) * 4
            if nodes_read % 10000000 == 0:
                print(".", end='', flush=True)
            if k > max_range_of_graph:
                max_range_of_graph = k
        
        print("done. Index has", nodes_read, "nodes and", cc, "out-edges, start is set to", start)
        
        return graph, nodes_read, start, file_frozen_pts
    

def main(args):
    if args.filename.endswith('.nsg'):
        graph = load_nsg(args.filename)
    elif args.filename.endswith('.data') or '.' not in args.filename:
        graph, nodes_read, start, file_frozen_pts = load_impl(args.filename)
        print(nodes_read,start)
    else:
        raise ValueError("Unsupported file format. Please use .nsg or .data files.")

    df = pd.DataFrame({'neighbor': graph})
    df['id'] = range(len(df))
    df = df[['id'] + ['neighbor']]
    df.to_csv(args.savename + '.csv', index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load index')
    parser.add_argument('--filename', type=str, help='Path to the index file')
    parser.add_argument('--savename', type=str, help='Path to the output CSV file')
    args = parser.parse_args()
    
    main(args)
    