# SQLVec

This is the supplement for the paper - SQLVec: SQL-Based Vector Similarity Search

## Pre-load Indexs
In our project, we utilize pre-built indexes for experiments, and convert them into a database-readable format (`.csv`) via the `load_index` function, which are then ultimately imported into the database. You can build indexes using algorithms such as [DiskANN](https://github.com/microsoft/DiskANN), [EFANNA,](https://github.com/ZJULearning/efanna) and [NSG](https://github.com/ZJULearning/nsg).

For a simple example, refer to NSG with the sample file [sift.nsg](http://downloads.zjulearning.org.cn/nsg/sift.nsg)



## Quick Start

```bash
python index2csv.py --filename ./data/index_file --savename ./output/index_csv
```

| parameter    | type   | required | description                                                  |
| ------------ | ------ | -------- | ------------------------------------------------------------ |
| `--filename` | string | True     | Path to the input index file                                 |
| `--savename` | string | True     | Prefix of the output CSV file (the final output file will be `savename.csv`, no need to add `.csv` suffix) |
