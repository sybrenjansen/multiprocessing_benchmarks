import psutil

# General parameters
num_trials = 5

# Count the number of physical CPUs
num_cpus = psutil.cpu_count(logical=False)
n_jobs_list = [n_jobs for n_jobs in [1, 2, 4, 8, 16, 20, 32, 64, 128, 256]
               if n_jobs <= num_cpus]

# Benchmark parameters
benchmark_1_params = {
    'random_seed': 42,
    'num_filters': num_cpus * 10,
    'image_size': (5_000, 5_000)
}

benchmark_2_params = {
    'random_seed': 42,
    'n_docs': 2_000,
    'n_words_per_doc': 10_000,
    'n_words_per_doc_odd_even_delta': 5_000,
    'n_bytes_per_word': 20
}

benchmark_3_params = {
    'random_seed': 42,
    'n_runs': 50,
    'n_evals': 10
}
