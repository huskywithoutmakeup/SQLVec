#include "duckdb.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <chrono>
#include <sys/stat.h>

#define N_DIM 128
#define DATA_SIZE 4
#define TOP_K 100

typedef struct {
    float *lookup_table;
} pq_distance_bind_data;

uint32_t* load_truthset(const char* filename, int* out_npts, int* out_dim) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // 读取 npts 和 dim（K）
    int32_t npts_i32, dim_i32;
    if (fread(&npts_i32, sizeof(int32_t), 1, f) != 1 ||
        fread(&dim_i32, sizeof(int32_t), 1, f) != 1) {
        fprintf(stderr, "Failed to read npts and dim from file\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    int npts = npts_i32;
    int dim = dim_i32;

    // 获取文件大小
    fseek(f, 0, SEEK_END);
    size_t actual_size = ftell(f);
    fseek(f, 8, SEEK_SET);  // 重置到 ids 起始位置

    size_t expected_with_dists = 8 + 8 * npts * dim;
    size_t expected_just_ids   = 8 + 4 * npts * dim;

    int truthset_type = -1;
    if (actual_size == expected_with_dists) {
        truthset_type = 1;
    } else if (actual_size == expected_just_ids) {
        truthset_type = 2;
    } else {
        fprintf(stderr, "File size mismatch. Actual: %zu, Expected: %zu or %zu\n",
                actual_size, expected_with_dists, expected_just_ids);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    // 读取 ID
    uint32_t* ids = (uint32_t*)malloc(npts * dim * sizeof(uint32_t));
    if (!ids) {
        fprintf(stderr, "Failed to allocate memory for ids\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    if (fread(ids, sizeof(uint32_t), npts * dim, f) != (size_t)(npts * dim)) {
        fprintf(stderr, "Failed to read ids from file\n");
        free(ids);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    // 如果存在距离值，也读取它们（可选）
    if (truthset_type == 1) {
        float* dists = (float*)malloc(npts * dim * sizeof(float));
        if (!dists) {
            fprintf(stderr, "Failed to allocate memory for distances\n");
            free(ids);
            fclose(f);
            exit(EXIT_FAILURE);
        }
        if (fread(dists, sizeof(float), npts * dim, f) != (size_t)(npts * dim)) {
            fprintf(stderr, "Failed to read distances from file\n");
            free(ids);
            free(dists);
            fclose(f);
            exit(EXIT_FAILURE);
        }
        // 如果你不需要 distances，可以立即释放
        free(dists);
    }

    fclose(f);
    *out_npts = npts;
    *out_dim = dim;
    return ids;
}

void destroy_lookup_bind_data(void *data) {
    free(data);  // 配合 malloc 的
}

// Function implementation
static void pq_distance_exec(duckdb_function_info info,  duckdb_data_chunk input,duckdb_vector output) {
    pq_distance_bind_data *bind_data = (pq_distance_bind_data *)duckdb_scalar_function_get_extra_info(info);
    float *lookup_table = bind_data->lookup_table;
    int M = 240;
    idx_t row_count = duckdb_data_chunk_get_size(input);
    
    duckdb_vector left_vec = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector right_vec = duckdb_data_chunk_get_vector(input, 1);
    float* result_data = static_cast<float*>(duckdb_vector_get_data(output));
    
    duckdb_vector left_child = duckdb_list_vector_get_child(left_vec);
    duckdb_vector right_child = duckdb_list_vector_get_child(right_vec);

    int32_t *lptr = (int32_t *)duckdb_vector_get_data(left_child);
    int32_t *rptr = (int32_t *)duckdb_vector_get_data(right_child);

    for (idx_t row = 0; row < row_count; row++) {
        float total = 0;

        for (int i = 0; i < M; i++) {
            int pos = 256 * 256 * i + 256 * lptr[i+M*row] + rptr[i+M*row];
            total += lookup_table[pos];
        }
        result_data[row] = total;
    }
}

void register_pq_function(duckdb_connection conn, float *lookup_table) {
    auto function = duckdb_create_scalar_function();
    duckdb_scalar_function_set_name(function, "pq_distance_compute");
    
    duckdb_logical_type element_type = duckdb_create_logical_type(DUCKDB_TYPE_INTEGER); // duckdb_logical_type duckdb_create_logical_type(duckdb_type type)
    duckdb_logical_type array_type = duckdb_create_array_type(element_type,240); // duckdb_logical_type duckdb_create_array_type(duckdb_logical_type type, idx_t array_size) 
    
    duckdb_scalar_function_add_parameter(function, array_type);
    duckdb_scalar_function_add_parameter(function, array_type);
    duckdb_destroy_logical_type(&element_type);
    
    // set the return type to bigint
    duckdb_logical_type type = duckdb_create_logical_type(DUCKDB_TYPE_FLOAT);
    duckdb_scalar_function_set_return_type(function, type);
    duckdb_destroy_logical_type(&type);
    
    duckdb_scalar_function_set_function(function, pq_distance_exec);
    
    pq_distance_bind_data *data = (pq_distance_bind_data *)malloc(sizeof(pq_distance_bind_data));
    data->lookup_table = lookup_table;  
    
    duckdb_scalar_function_set_extra_info(function, data, destroy_lookup_bind_data);

    duckdb_register_scalar_function(conn, function);
    duckdb_destroy_scalar_function(&function);
}


float* fvecs_read(const char* filename, int* num_vectors) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }

    int dim;
    fread(&dim, sizeof(int), 1, file);
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    *num_vectors = file_size / ((N_DIM + 1) * sizeof(int));
    fseek(file, 0, SEEK_SET);

    float* data = (float*)malloc((*num_vectors) * N_DIM * sizeof(float));
    for (int i = 0; i < *num_vectors; i++) {
        int tmp_dim;
        fread(&tmp_dim, sizeof(int), 1, file);
        fread(data + i * N_DIM, sizeof(float), N_DIM, file);
    }

    fclose(file);
    return data;
}

int main() {
    const char* database_path = "/home/anns/ANNS/cpp/db/gistpq1.duckdb";
    const int enter_node = 356422;
    const char* groundtruth_path = "/home/anns/ANNS/duckdb/gist_base_gt100";
    const int L = 300;
    const int K = 100;
    const int T = 5;
    const int W = 16;

    duckdb_database db;
    duckdb_connection con;
    duckdb_state state;

    // 初始化数据库
    state = duckdb_open(database_path, &db);
    if (state != DuckDBSuccess) {
        fprintf(stderr, "Failed to open database.\n");
        return 1;
    }

    // 创建连接
    state = duckdb_connect(db, &con);
    if (state != DuckDBSuccess) {
        fprintf(stderr, "Failed to connect to database.\n");
        duckdb_close(&db);
        return 1;
    }

    // 初始化 lookup table
    const size_t M = 240;
    const size_t Ksub = 256;
    const size_t TABLE_SIZE = M * Ksub * Ksub;
    
    float *lookup_table = new float[TABLE_SIZE];

    std::ifstream infile("/home/anns/ANNS/cpp/db/pq_lookup_table.bin", std::ios::binary);
    if (!infile) {
        std::cerr << "Failed to open pq_lookup_table.bin" << std::endl;
        return 1;
    }

    infile.read(reinterpret_cast<char *>(lookup_table), TABLE_SIZE * sizeof(float));
    infile.close();

    // 注册函数
    register_pq_function(con, lookup_table);
    
    // 设置参数
    const char* settings[] = {
        "PRAGMA enable_optimizer",
        "SET threads = 8",
        "SET worker_threads = 8",
        "SET memory_limit = '10GB'",
        "SET max_memory = '10GB'"
    };
    for (int i = 0; i < 5; i++) {
        state = duckdb_query(con, settings[i], NULL);
        if (state != DuckDBSuccess) {
            fprintf(stderr, "Failed to execute setting: %s\n", settings[i]);
            duckdb_disconnect(&con);
            duckdb_close(&db);
            return 1;
        }
    }

    // 删除旧表
    const char* drop_tables[] = {
        "DROP TABLE IF EXISTS L_table",
        "DROP TABLE IF EXISTS V_table",
        "DROP TABLE IF EXISTS tp_table"
    };
    for (int i = 0; i < 3; i++) {
        state = duckdb_query(con, drop_tables[i], NULL);
        if (state != DuckDBSuccess) {
            fprintf(stderr, "Failed to drop table: %s\n", drop_tables[i]);
            duckdb_disconnect(&con);
            duckdb_close(&db);
            return 1;
        }
    }

    // 获取 enter_data
    duckdb_result result;
    char query[4096];
    snprintf(query, sizeof(query), "SELECT CAST(data AS VARCHAR) FROM data_table WHERE id = %d", enter_node);
    state = duckdb_query(con, query, &result);
    if (state != DuckDBSuccess) {
        fprintf(stderr, "Failed to execute query: %s\n", query);
        duckdb_disconnect(&con);
        duckdb_close(&db);
        return 1;
    }
    const char* enter_data = duckdb_value_varchar(&result, 0, 0);
    duckdb_destroy_result(&result);

    // 创建临时表 L_table 和 V_table
    state = duckdb_query(con, "CREATE TEMP TABLE L_table(id INTEGER, dist DOUBLE, batch INTEGER)", NULL);
    if (state != DuckDBSuccess) {
        fprintf(stderr, "Failed to create L_table.\n");
        duckdb_disconnect(&con);
        duckdb_close(&db);
        return 1;
    }

    char insert_query[2048];
    snprintf(insert_query, sizeof(insert_query),
            "INSERT INTO L_table SELECT %d, pq_distance_compute(CAST('%s' AS INT[240]), data) as dist, num as batch from query_table",
            enter_node, enter_data);

    state = duckdb_query(con, insert_query, &result);
    if (state != DuckDBSuccess) {
        const char *err = duckdb_result_error(&result);
        if (err) {
            fprintf(stderr, "Failed to insert into L_table: %s\n", err);
        } else {
            fprintf(stderr, "Failed to insert into L_table with unknown error.\n");
        }
        duckdb_destroy_result(&result);
        duckdb_disconnect(&con);
        duckdb_close(&db);
        return 1;
    }

    // duckdb_query(con, "SELECT id,dist,batch FROM L_table", &result);
    // idx_t rc = duckdb_row_count(&result);
    // printf("row_count: %d\n", rc);

    // for (idx_t i = 0; i < 10; i++) {
    //     int id = duckdb_value_int32(&result, 0, i);
    //     double dist = duckdb_value_double(&result, 1, i);
    //     int batch = duckdb_value_int32(&result, 2, i);
    //     printf("id: %d\n", id);
    //     printf("dist: %f\n", dist);
    //     printf("batch: %d\n", batch);
    // }

    // 这段有问题

    state = duckdb_query(con, "CREATE TEMP TABLE V_table(id INTEGER, batch INTEGER)", NULL);
    if (state != DuckDBSuccess) {
        fprintf(stderr, "Failed to create V_table.\n");
        duckdb_disconnect(&con);
        duckdb_close(&db);
        return 1;
    }

    // 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();

    int cnt = 9, it = 0;
    while (cnt != 0) {
        it++;

        state = duckdb_query(con,
            "CREATE TEMP TABLE tp1 AS SELECT L_table.* FROM L_table LEFT JOIN V_table ON L_table.id = V_table.id AND L_table.batch = V_table.batch WHERE V_table.id IS NULL",
            NULL);
        if (state != DuckDBSuccess) {
            fprintf(stderr, "Failed to create tp1.\n");
            duckdb_disconnect(&con);
            duckdb_close(&db);
            return 1;
        }

        state = duckdb_query(con, "SELECT COUNT(*) FROM tp1", &result);
        if (state != DuckDBSuccess) {
            fprintf(stderr, "Failed to count tp1.\n");
            duckdb_disconnect(&con);
            duckdb_close(&db);
            return 1;
        }
        cnt = duckdb_value_int32(&result, 0, 0);
        // printf("cnt: %d\n", cnt);
        duckdb_destroy_result(&result);

        if (it >= T) {
            char tpv_query[1024];
            // 应该是这一步的
            snprintf(tpv_query, sizeof(tpv_query),
                     "CREATE TEMP TABLE tpv AS SELECT d.id, d.dist, d.batch FROM "
                     "(SELECT t1.*, RANK() OVER (PARTITION BY batch ORDER BY dist) AS class_rank FROM tp1 t1) d "
                     "WHERE d.class_rank <= %d ORDER BY batch, dist", W);
            state = duckdb_query(con, tpv_query, NULL);
        } else {
            state = duckdb_query(con,
                "CREATE TEMP TABLE tpv AS SELECT t1.* FROM tp1 t1 JOIN "
                "(SELECT batch, MIN(dist) AS mind FROM tp1 GROUP BY batch) t2 "
                "ON t1.batch = t2.batch AND t1.dist = t2.mind",
                NULL);
        }
        if (state != DuckDBSuccess) {
            fprintf(stderr, "Failed to create tpv.\n");
            duckdb_disconnect(&con);
            duckdb_close(&db);
            return 1;
        }

        // duckdb_query(con, "SELECT count(*) FROM tpv", &result);
        // idx_t rc = duckdb_value_int32(&result, 0, 0);
        // printf("tpv_count: %d\n", rc);

        state = duckdb_query(con,
            "CREATE TEMP TABLE tp2 AS SELECT DISTINCT batch, unnest(nbs) AS neighbor "
            "FROM tpv JOIN Graph_index_table USING(id)",
            NULL);
        if (state != DuckDBSuccess) {
            fprintf(stderr, "Failed to create tp2.\n");
            duckdb_disconnect(&con);
            duckdb_close(&db);
            return 1;
        }

        state = duckdb_query(con,
            "CREATE TEMP TABLE tp3 AS SELECT neighbor as id, pq_distance_compute(data_table.data,query_table.data) as dist,tp2.batch "
            "FROM data_table,query_table,tp2 WHERE tp2.neighbor = data_table.id AND query_table.num = tp2.batch",
            NULL);
        if (state != DuckDBSuccess) {
            fprintf(stderr, "Failed to create tp3.\n");
            duckdb_disconnect(&con);
            duckdb_close(&db);
            return 1;
        }

        state = duckdb_query(con, "INSERT INTO V_table SELECT id, batch FROM tpv", NULL);
        if (state != DuckDBSuccess) {
            fprintf(stderr, "Failed to insert into V_table.\n");
            duckdb_disconnect(&con);
            duckdb_close(&db);
            return 1;
        }

        state = duckdb_query(con, "DROP TABLE tp1", NULL);
        state = duckdb_query(con, "DROP TABLE tp2", NULL);
        state = duckdb_query(con, "DROP TABLE tpv", NULL);

        char tp_table_query[1024];
        snprintf(tp_table_query, sizeof(tp_table_query),
                 "CREATE TEMP TABLE tp_table AS SELECT d.id,d.dist,d.batch FROM "
                 "(SELECT l1.*, rank() over (PARTITION BY batch ORDER BY dist) AS class_rank FROM "
                 "(SELECT * FROM L_table UNION SELECT * FROM tp3) l1) d "
                 "WHERE d.class_rank <= %d order by batch,dist", L);
        state = duckdb_query(con, tp_table_query, NULL);
        if (state != DuckDBSuccess) {
            fprintf(stderr, "Failed to create tp_table.\n");
            duckdb_disconnect(&con);
            duckdb_close(&db);
            return 1;
        }

        state = duckdb_query(con, "DROP TABLE tp3", NULL);
        state = duckdb_query(con, "DROP TABLE L_table", NULL);
        state = duckdb_query(con, "ALTER TABLE tp_table RENAME TO L_table", NULL);
    }

    // 结束计时
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    printf("Iteration: %d\n", it);
    printf("Total Time Taken: %f seconds\n", elapsed_time.count());
    printf("QPS: %f\n", 1000 / elapsed_time.count());

    int query_size =1000, dim=100;
    uint32_t *ground_truth = load_truthset("/home/anns/ANNS/dataset/gist1m/gist_base_gt100", &query_size, &dim);

    // Evaluate Recall
    duckdb_query(con,
        "SELECT d.id, d.batch FROM ("
        "SELECT l1.*, row_number() over (PARTITION BY batch ORDER BY dist) AS class_rank "
        "FROM L_table l1) d WHERE d.class_rank <= 300 ORDER BY batch, dist", &result);

    int intersection = 0;
    idx_t row_count = duckdb_row_count(&result);
    
    // 好像还有点问题
    for (idx_t i = 0; i < row_count; i++) {
        int id = duckdb_value_int32(&result, 0, i);
        int batch = duckdb_value_int32(&result, 1, i);
        // 比较当前结果是否存在于 ground_truth[batch]
        for (int j = 0; j < dim; j++) {
            if (ground_truth[batch * dim + j] == id) {
                intersection++;
                break;
            }
        }
    }

    printf("Recall: %.4f", (double)intersection / (dim * query_size));

    // 清理资源
    duckdb_disconnect(&con);
    duckdb_close(&db);

    return 0;
}    