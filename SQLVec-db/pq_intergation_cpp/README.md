# SQLVec

This is the supplement for the paper - SQLVec: SQL-Based Vector Similarity Search

## Requirements
The CPP code requires g++, cmake, which can be installed via：
```bash
sudo apt-get install g++ cmake
```

And you can install **libduckdb** through [the offical website](libduckdb)

and the structure of the repository can be:

````
.
├── db              # Database storage directory
├── libduckdb  		# Dependent library directory
├── pq_generate  	# Data & database generation module
└── CMakeLists.txt             
````

__See each folder for more information.__



## Quick Start

```bash
mkdir build
cd build
cmake ..
make

./SQLvec
```

