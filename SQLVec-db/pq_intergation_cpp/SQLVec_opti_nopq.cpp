#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cassert>
#include <duckdb.hpp>
#include <algorithm>
#include <unordered_map>
#include <chrono>

#define N_DIM 128
#define DATA_SIZE 4

std::vector<float> fvecs_read(const std::string &filename, bool c_contiguous = true, int record_count = -1, int line_offset = 0, const std::string &record_dtype = "int32") {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    if (record_count > 0) {
        record_count *= N_DIM + 1;
    }
    if (line_offset > 0) {
        line_offset *= (N_DIM + 1) * DATA_SIZE;
    }

    input.seekg(line_offset, std::ios::beg);
    std::vector<float> fv(record_count);
    input.read(reinterpret_cast<char*>(fv.data()), record_count * sizeof(float));
    input.close();

    if (fv.size() == 0) {
        return std::vector<float>(0);
    }

    int dim = static_cast<int>(fv[0]);
    assert(dim > 0);
    fv.resize(fv.size() / (dim + 1) * dim);

    if (c_contiguous) {
        std::vector<float> contiguous_fv(fv.begin(), fv.end());
        return contiguous_fv;
    }
    return fv;
}

int main(int argc, char** argv) {
    std::string database = "./db/sift.duckdb";  
    int enter_node = 123742;
    std::string groundtruth_path = "/home/anns/ANNS/dataset/sift1m/sift_groundtruth.ivecs";
    int L = 100;
    int K = 100;
    int T = 5;
    int W = 8;

    // Connect to DuckDB database
    duckdb::DuckDB db(database);
    duckdb::Connection con(db);

    
    con.Query("DROP TABLE IF EXISTS L_table");
    con.Query("DROP TABLE IF EXISTS V_table");
    con.Query("DROP TABLE IF EXISTS tp_table");

    con.Query("PRAGMA enable_optimizer");
    con.Query("SET threads = 8");
    con.Query("SET worker_threads = 8");
    con.Query("SET memory_limit = '10GB'");
    con.Query("SET max_memory = '10GB'");

    auto enter_data = con.Query("SELECT data FROM data_table WHERE id = 123742")->Fetch()->GetValue(0,0).ToString();

    auto start_time_total = std::chrono::high_resolution_clock::now();

    con.Query("CREATE TEMP TABLE L_table(id INTEGER, dist DOUBLE, batch INTEGER)");
    con.Query("INSERT INTO L_table SELECT 123742, array_distance(CAST(" + enter_data + " AS FLOAT[128]), data) as dist, num as batch from query_table");
    con.Query("CREATE TEMP TABLE V_table(id INTEGER, batch INTEGER)");


    int cnt = 9, it = 0;
    while (cnt != 0) {
        it++;

        con.Query("CREATE TEMP TABLE tp1 AS SELECT L_table.* FROM L_table LEFT JOIN V_table ON L_table.id = V_table.id AND L_table.batch = V_table.batch WHERE V_table.id IS NULL");
        cnt = con.Query("SELECT id FROM tp1")->RowCount();

        if (it >= T) {
            con.Query("CREATE TEMP TABLE tpv AS SELECT d.id, d.dist, d.batch FROM "
                    "(SELECT t1.*, ROW_NUMBER() OVER (PARTITION BY batch ORDER BY dist) AS class_rank FROM tp1 t1) d "
                    "WHERE d.class_rank <= " + std::to_string(W) + " ORDER BY batch, dist");
        } else {
            con.Query("CREATE TEMP TABLE tpv AS SELECT t1.* FROM tp1 t1 JOIN "
                    "(SELECT batch, MIN(dist) AS mind FROM tp1 GROUP BY batch) t2 "
                    "ON t1.batch = t2.batch AND t1.dist = t2.mind");
        }

        con.Query("CREATE TEMP TABLE tp2 AS SELECT DISTINCT batch, unnest(nbs) AS neighbor "
                "FROM tpv JOIN Graph_index_table USING(id)");

        con.Query("CREATE TEMP TABLE tp3 AS SELECT neighbor as id, array_distance(data_table.data,query_table.data) as dist,tp2.batch FROM data_table,query_table,tp2 WHERE tp2.neighbor = data_table.id AND query_table.num = tp2.batch");

        con.Query("INSERT INTO V_table SELECT id, batch FROM tpv");
        con.Query("DROP TABLE tp1");
        con.Query("DROP TABLE tp2");
        con.Query("DROP TABLE tpv");

        con.Query("CREATE TEMP TABLE tp_table AS SELECT d.id,d.dist,d.batch FROM (SELECT l1.*, row_number() over (PARTITION BY batch ORDER BY dist) AS class_rank FROM (SELECT * FROM L_table UNION SELECT * FROM tp3) l1) d WHERE d.class_rank <= " + std::to_string(L) + " order by batch,dist");
        con.Query("DROP TABLE tp3");
        con.Query("DROP TABLE L_table");
        con.Query("ALTER TABLE tp_table RENAME TO L_table");
    }

    auto end_time_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time_total - start_time_total;

    // // Collect results
    // auto res_all = con.Query("SELECT d.id, d.batch FROM (SELECT l1.*, row_number() over (PARTITION BY batch ORDER BY dist) AS class_rank FROM L_table l1) d WHERE d.class_rank <= " + std::to_string(K) + " ORDER BY batch, dist");
    // std::unordered_map<int, std::vector<int>> batch_dict;

    // for (size_t i = 0; i < res_all.size(); i++) {
    //     int id = res_all[i][0];
    //     int batch = res_all[i][1];
    //     batch_dict[batch].push_back(id);
    // }

    // std::vector<std::vector<int>> result;
    // for (const auto &entry : batch_dict) {
    //     result.push_back(entry.second);
    // }

    // // Read ground truth data
    // std::vector<float> ground_truth = fvecs_read(groundtruth_path);

    // // Calculate average recall
    // int intersection = 0;
    // for (size_t i = 0; i < result.size(); i++) {
    //     std::vector<int> &res = result[i];
    //     std::vector<int> &truth = ground_truth[i];
    //     std::vector<int> inter;
    //     std::set_intersection(res.begin(), res.end(), truth.begin(), truth.end(), std::back_inserter(inter));
    //     intersection += inter.size();
    // }
    // std::cout << "Average Recall: " << static_cast<double>(intersection) / (100 * 10000) << std::endl;

    std::cout << "Iteration: " << it << std::endl;
    std::cout << "Total Time Taken: " << elapsed_time.count() << " seconds" << std::endl;
    std::cout << "QPS: " << 10000 / elapsed_time.count() << std::endl;

    return 0;
}
