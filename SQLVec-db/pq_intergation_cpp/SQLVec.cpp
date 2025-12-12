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
    std::string database = "/home/anns/ANNS/duckdb/sift.duckdb";  
    int enter_node = 123742;
    std::string groundtruth_path = "/home/anns/ANNS/dataset/sift1m/sift_groundtruth.ivecs";
    int L = 100;
    int K = 100;

    // Connect to DuckDB database
    duckdb::DuckDB db(database);
    duckdb::Connection con(db);

    // Create table
    // index table
    con.Query("CREATE TABLE temp_table AS SELECT * FROM read_csv('/home/zzq229/ANNSonSQL/tables/graph_index_vamana_R70_L75_A12.csv')");
    con.Query("CREATE TABLE Graph_index_table AS select id, cast(neighbor as int[]) as neighbor from temp_table");
    con.Query("DROP TABLE temp_table"); 
    // data_table
    con.Query("CREATE TABLE temp_table AS SELECT * FROM read_csv('/home/zzq229/ANNSonSQL/tables/data_table.csv')");
    con.Query("CREATE TABLE data_table AS select id, cast(data as int[]) as data from temp_table");
    con.Query("DROP TABLE temp_table");
    // query_table
    con.Query("CREATE TABLE temp_table AS SELECT * FROM read_csv('/home/zzq229/ANNSonSQL/tables/query_table.csv')");
    con.Query("CREATE TABLE query_table AS select num, cast(data as int[]) as data from temp_table");
    con.Query("DROP TABLE temp_table");
    // Read ground truth data
    // std::vector<float> ground_truth = fvecs_read(groundtruth_path);

    // Create indices
    con.Query("CREATE INDEX g_id_index ON Graph_index_table(id)");
    con.Query("CREATE INDEX d_id_index ON data_table(id)");

    // Enable optimizer
    con.Query("PRAGMA enable_optimizer");

    // auto enter_node = con.Query("SELECT data FROM data_table WHERE id = 123742")->FetchRaw();
    // std::cout << enter_node << std::endl;

    // Start query processing
    auto start_time_total = std::chrono::high_resolution_clock::now();

    con.Query("CREATE TEMP TABLE L_table(id INTEGER, dist DOUBLE, batch INTEGER)");

    // std::unique_ptr<duckdb::PreparedStatement> p1 = con.Prepare("INSERT INTO L_table SELECT 123742, list_distance([20, 8, 19, 10, 13, 27, 23, 9, 58, 45, 34, 18, 17, 27, 33, 58, 27, 51, 85, 28, 18, 38, 34, 67, 18, 7, 32, 38, 63, 38, 7, 3, 52, 21, 27, 22, 39, 29, 37, 22, 86, 51, 62, 22, 57, 60, 54, 62, 52, 34, 37, 21, 60, 64, 102, 101, 60, 56, 66, 20, 36, 55, 33, 40, 44, 24, 40, 20, 40, 35, 35, 23, 93, 50, 62, 31, 54, 53, 81, 44, 80, 67, 100, 22, 78, 56, 58, 50, 43, 14, 42, 17, 40, 53, 59, 54, 12, 5, 28, 40, 63, 12, 13, 13, 42, 21, 21, 24, 36, 39, 47, 33, 66, 23, 21, 18, 25, 31, 51, 73, 25, 14, 25, 20, 33, 26, 18, 15], data) as dist, num as batch from query_table");
    // p1->Execute();
    con.Query("INSERT INTO L_table SELECT 123742, list_distance([20,8,19,10,13,27,23,9,58,45,34,18,17,27,33,58,27,51,85,28,18,38,34,67,18,7,32,38,63,38,7,3,52,21,27,22,39,29,37,22,86,51,62,22,57,60,54,62,52,34,37,21,60,64,102,101,60,56,66,20,36,55,33,40,44,24,40,20,40,35,35,23,93,50,62,31,54,53,81,44,80,67,100,22,78,56,58,50,43,14,42,17,40,53,59,54,12,5,28,40,63,12,13,13,42,21,21,24,36,39,47,33,66,23,21,18,25,31,51,73,25,14,25,20,33,26,18,15], data) as dist, num as batch from query_table");

    con.Query("CREATE TEMP TABLE V_table(id INTEGER, dist DOUBLE, batch INTEGER)");

    int cnt = 9;
    int it = 0;
    while (cnt != 0) {
        it++;
        // Step 1
        con.Query("CREATE TEMP TABLE tp1 AS SELECT * FROM L_table EXCEPT SELECT * FROM V_table");
        cnt = con.Query("SELECT * FROM tp1")->RowCount();
        // std::cout << cnt << std::endl;

        // Step 1+2
        con.Query("CREATE TEMP TABLE tpv AS SELECT DISTINCT t1.* FROM tp1 t1 JOIN (SELECT batch, min(dist) AS mind FROM tp1 GROUP BY batch) t2 ON t1.batch = t2.batch AND t1.dist = t2.mind");

        // Step 3
        con.Query("CREATE TEMP TABLE tp2 AS SELECT DISTINCT batch, unnest(neighbor) as neighbor FROM tpv JOIN Graph_index_table USING(id)");

        con.Query("INSERT INTO L_table SELECT neighbor as id, list_distance(data_table.data, query_table.data) as dist, tp2.batch FROM data_table, query_table, tp2 WHERE tp2.neighbor = data_table.id AND query_table.num = tp2.batch");

        con.Query("INSERT INTO V_table SELECT * FROM tpv");
        con.Query("DROP TABLE tp1");
        con.Query("DROP TABLE tp2");
        con.Query("DROP TABLE tpv");

        con.Query("CREATE TABLE tp_table AS SELECT DISTINCT d.id, d.dist, d.batch FROM (SELECT l1.*, row_number() over (PARTITION BY batch ORDER BY dist) AS class_rank FROM (SELECT DISTINCT * FROM L_table) l1) d WHERE d.class_rank <= 100 ORDER BY batch, dist");
        con.Query("DROP TABLE L_table");
        con.Query("ALTER TABLE tp_table RENAME TO L_table");
    }

    auto end_time_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time_total - start_time_total;

    // Collect results
    // auto res_all = con.Query("SELECT d.id, d.batch FROM (SELECT l1.*, row_number() over (PARTITION BY batch ORDER BY dist) AS class_rank FROM L_table l1) d WHERE d.class_rank <= 100 ORDER BY batch, dist")
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

    // // Calculate average recall
    // int intersection = 0;
    // for (size_t i = 0; i < result.size(); i++) {
    //     std::vector<int> &res = result[i];
    //     std::vector<int> &truth = ground_truth[i];
    //     std::vector<int> inter;
    //     std::set_intersection(res.begin(), res.end(), truth.begin(), truth.end(), std::back_inserter(inter));
    //     intersection += inter.size();
    // }

    std::cout << "Iteration: " << it << std::endl;
    // std::cout << "Average Recall: " << static_cast<double>(intersection) / (100 * 10000) << std::endl;
    std::cout << "Total Time Taken: " << elapsed_time.count() << " seconds" << std::endl;
    std::cout << "Average Time Taken: " << elapsed_time.count() / 10000 << " seconds" << std::endl;

    return 0;
}
