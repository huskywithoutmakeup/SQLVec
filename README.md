# SQLVec

This is the supplement for the paper - SQLVec: SQL-Based Vector Similarity Search

## Structure of the Repository

````
.
├── Scripts            # Unified scripts for all query variants and optimization strategies across databases
├── dataload           # Loading of datasets and indexes
├── SQLvec-db  		   # SQLvec framework implemented on DuckDB, including specific query examples
├── SQLvec-pg  		   # SQLvec framework implemented on PostgreSQL, including specific query examples
└── others             # more details
````

__See each folder for more information.__

## Requirements
The experiments were tested using Python version 3.8，and all databases are connected through their corresponding Python connection libraries. 

### Postgres
The experiments need a postgres DBMS installation and python connector. We used psql version 17.2. You can install postgres on linux using:

````commandline
sudo apt-get install postgresql-17 psycopg2
````
For more details, please visit [PostgreSQL](https://www.postgresql.org/download/linux/ubuntu/).

The postgres installation should have a database with the following configuaration:

* name: 'postgres',
* user: 'postgres',
* password: 'pw',
* host: 'localhost'.

### DuckDB

We used duckdb version 1.1.3. You can install duckdb on linux using

````commandline
pip install duckdb==1.1.3
````

### Other

The experiments further need other DBMS, such as MariaDB.

