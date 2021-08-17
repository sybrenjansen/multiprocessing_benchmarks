import os
import pickle
import time
from collections import defaultdict
from itertools import islice
from multiprocessing import Process
from typing import Callable, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from labellines import labelLines

from params import (benchmark_1_params, benchmark_2_params, n_jobs_list,
                    num_trials, benchmark_3_params)

# Contains: list of (n_jobs, init_durations, task_durations, exit_durations)
# tuples
BechmarkResults = List[Tuple[int, List[float], List[float], List[float]]]

# Contains library name, x, y_total_time, y_total_time_err
ProcessedBenchmarkResults = Tuple[str, List[int], List[float], List[float]]

# Apply seaborn theme
sns.set_theme()
sns.color_palette("hls", 6)


def create_image_filters() -> List[np.ndarray]:
    """
    Creates image filters for benchmark #1

    :return: List of image filters
    """
    np.random.seed(benchmark_1_params['random_seed'])
    return [np.random.normal(size=(4, 4))
            for _ in range(benchmark_1_params['num_filters'])]


def create_image() -> np.ndarray:
    """
    Creates an image for benchmark #1

    :return: Image
    """
    return np.zeros(benchmark_1_params['image_size'])


def create_documents() -> List[List[bytes]]:
    """
    Creates a list of documents

    :return: List of documents
    """
    print("Creating benchmark #2 documents data ...")
    np.random.seed(benchmark_2_params['random_seed'])
    data = iter(np.random.bytes(benchmark_2_params['n_docs'] *
                                benchmark_2_params['n_words_per_doc'] *
                                benchmark_2_params['n_bytes_per_word']))

    # For even indices we add delta words, for odd indices we subtract delta.
    # This is done to make the workload unbalanced. Libraries that can handle
    # unbalanced tasks are better suited.
    delta = benchmark_2_params['n_words_per_doc_odd_even_delta']
    return [[bytes(islice(data, benchmark_2_params['n_bytes_per_word']))
             for _ in range(benchmark_2_params['n_words_per_doc'] +
                            (delta if idx % 2 == 0 else -delta))]
            for idx in range(benchmark_2_params['n_docs'])]


def create_and_save_tf_model(filename: str) -> None:
    """
    Train and save the model. This has to be done in a separate process because
    otherwise Python multiprocessing will hang when you try do run the code
    below.

    :param filename: Filename to save the model to
    """
    p = Process(target=_create_and_save_tf_model, args=(filename,))
    p.start()
    p.join()


def _create_and_save_tf_model(filename: str) -> None:
    """
    Creates, trains, and saves a simple tensorflow model to disk

    :param filename: Filename to save the model to
    """
    print("Creating tensorflow model for benchmark #3 ...")
    tf.random.set_seed(benchmark_3_params['random_seed'])
    mnist = tf.keras.datasets.mnist.load_data()
    x_train, y_train = mnist[0]
    x_train = x_train / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    # Train and save the model to disk
    model.fit(x_train, y_train, epochs=1)
    model.save(filename)


def run_trials(benchmark_function: Callable, benchmark_name: str,
               library_name: str, init_function: Optional[Callable] = None,
               exit_function: Optional[Callable] = None,
               serial_processing: bool = False) -> None:
    """
    Runs the benchmark function num_trials times and determines the run times,
    including mean and standard deviation.

    :param benchmark_function: Function to benchmark
    :param benchmark_name: Name of the benchmark, for printing purposes
    :param library_name: Name of the library used to create these results
    :param init_function: Initialization function, or None
    :param exit_function: Exit function, or None
    :param serial_processing: When True, will not use n_jobs > 1
    """
    durations = []
    for n_jobs in [1] if serial_processing else n_jobs_list:
        print(f'Running with {n_jobs} job{"s" if n_jobs != 1 else ""}')
        init_durations = []
        task_durations = []
        exit_durations = []
        for _ in range(num_trials):
            # Initialize
            if init_function is not None:
                start_time = time.time()
                init_function(n_jobs)
                duration = time.time() - start_time
                init_durations.append(duration)
                print(f"Initialization took {duration} seconds.")

            # Benchmark
            start_time = time.time()
            benchmark_function(n_jobs)
            duration = time.time() - start_time
            task_durations.append(duration)
            print(f"{benchmark_name} ({library_name}) took {duration} "
                  f"seconds.")

            # Shutdown
            if exit_function is not None:
                start_time = time.time()
                exit_function()
                duration = time.time() - start_time
                exit_durations.append(duration)
                print(f"Shutting down took {duration} seconds.")

        durations.append((n_jobs, init_durations, task_durations,
                          exit_durations))

    show_and_save_results_summary(benchmark_name, library_name, durations)


def show_and_save_results_summary(benchmark_name: str, library_name: str,
                                  all_durations: BechmarkResults) -> None:
    """
    Process, print and save the benchmark results summary

    :param benchmark_name: Benchmark name
    :param library_name: Library name
    :param all_durations: List of (n_jobs, init_durations, task_durations,
        exit_durations) tuples
    """
    # Clear file if it exists
    try:
        os.unlink(f"{benchmark_name}_{library_name}.p3")
    except FileNotFoundError:
        pass

    # Loop through different n_jobs settings
    all_results = []
    for n_jobs, init_durations, task_durations, exit_durations in all_durations:

        # Calculate total durations
        total_durations = np.sum([durations for durations in
                                  [init_durations, task_durations,
                                   exit_durations] if durations], axis=0)

        # Format
        results = []
        plot_duration_stats = {}
        for name, durations in [("runtime", task_durations),
                                ("init", init_durations),
                                ("exit", exit_durations),
                                ("total", total_durations)]:
            if len(durations):
                mean = np.mean(durations)
                std = np.std(durations)
                results.append(f"{name} mean: {mean}, std: {std}")
                if name == "total":
                    plot_duration_stats[name] = mean, std

        all_results.append(f"  - {n_jobs} jobs, {', '.join(results)}")

        # Save results to disk for plotting
        with open(f"{benchmark_name}_{library_name}.p3", "ab+") as f:
            pickle.dump((benchmark_name, library_name, n_jobs,
                         plot_duration_stats), f)

    print("Results:")
    results_str = "\n".join(all_results)
    print(f"- {benchmark_name}:\n{results_str}")


def visualize_results_for_benchmark(files: List[str],
                                    align_labels: bool = False) -> None:
    """
    Aggregates results per library for a single benchmark and saves the figure
    to file.

    :param files: List of files that belong to a specific benchmark
    :param align_labels: Whether to align the labels
    """
    # Aggregate results per library
    results_per_library = []
    benchmark_name = None
    for filename in files:
        benchmark_name, data = parse_plotting_data_file(filename)
        results_per_library.append(data)

    # Experiment with order
    create_figure(benchmark_name, results_per_library, align_labels)


def visualize_total_benchmark_times(files: List[List[str]],
                                    align_labels: bool = False) -> None:
    """
    Aggregates results per library for all benchmarks and saves the figure to
    file. It normalizes the total times of each benchmark to the range 0-1.

    :param files: List containing list of files that belong to a specific
    benchmark
    :param align_labels: Whether to align the labels
    """
    data = defaultdict(lambda: np.zeros(len(n_jobs_list)))
    for benchmark in files:
        max_value = 0
        benchmark_results = {}
        for filename in benchmark:
            _, (library_name, _, y_total_time,
                _) = parse_plotting_data_file(filename)
            max_value = max(max_value, *y_total_time)
            benchmark_results[library_name] = y_total_time
        for library_name, y_total_time in benchmark_results.items():
            data[library_name] += np.array(y_total_time) / max_value

    create_figure("All benchmarks",
                  [(library_name, n_jobs_list, list(y_total_time / len(files)),
                    []) for library_name, y_total_time in data.items()],
                  align_labels, "Normalized average time")


def parse_plotting_data_file(filename: str) -> Tuple[str,
                                                     ProcessedBenchmarkResults]:
    """
    Parses a data file for plotting

    :param filename: Filename to parse
    :return: Benchmark name + tuple consisting of library name, x, y_total_time
    """
    with open(filename, 'rb') as f:
        library_name = None
        x = []
        y_total_time = []
        y_total_time_err = []
        while True:
            try:
                (benchmark_name, library_name, n_jobs,
                 results) = pickle.load(f)
                x.append(n_jobs)
                y_total_time.append(results['total'][0])
                y_total_time_err.append(results['total'][1])
            except EOFError:
                break

        print(library_name, x, y_total_time)

        # Add additional data for serial processing
        if len(x) == 1:
            x = n_jobs_list
            y_total_time *= len(n_jobs_list)
            y_total_time_err *= len(n_jobs_list)

        return benchmark_name, (library_name, x, y_total_time, y_total_time_err)


def create_figure(benchmark_name: str,
                  results_per_library: List[ProcessedBenchmarkResults],
                  align_labels: bool, y_label: Optional[str] = None) -> None:
    """
    Creates a matplotlib figure and saves it

    :param benchmark_name: Benchmark name
    :param results_per_library: List of processed benchmark results
    :param align_labels: Whether to align the labels
    :param y_label: Y-axis label. If None, will use "Average time (seconds)"
    """
    def _create_figure(order):
        # Create figure for runtimes
        _fig = plt.figure(figsize=(12, 8))
        x = None
        for library_name, x, y_total_time, *_ in results_per_library:
            # Set style
            if library_name != "MPIRE":
                linestyle = "dashed"
                linewidth = 1.
            else:
                linestyle = "solid"
                linewidth = 2.5
            _lines = plt.plot(x, y_total_time, label=library_name,
                              linestyle=linestyle, linewidth=linewidth)

        plt.title(benchmark_name)
        plt.xlabel("Number of workers")
        plt.ylabel(y_label or "Average time (seconds)")
        plt.xticks(x)
        plt.tight_layout()
        _lines = {line._label: line for line in plt.gca().get_lines()}
        _lines = [_lines[label] for label in order]
        labelLines(_lines, align=align_labels, zorder=2.5)
        plt.show()
        return _fig

    # Interactive mode
    plt.ion()

    # Obtain a good order of label locations from the user
    original_order = [x[0] for x in results_per_library]
    fig = _create_figure(original_order)
    happy = False
    while not happy:
        print(", ".join(f"{idx}. {name}"
                        for idx, name in enumerate(original_order)))
        while True:
            try:
                new_order = input("Set order of plot labels (q=quit): ")
                if new_order == 'q':
                    break
                new_order = [int(idx) for idx in new_order.split()]
                if (len(set(new_order)) != len(original_order) or
                        not all(0 <= idx < len(original_order)
                                for idx in new_order)):
                    raise ValueError
                break
            except ValueError:
                print(f"Please provide a list of {len(original_order)} "
                      f"unique numbers between 0-{len(original_order)}")

        # Redraw and ask for hapiness
        if new_order != 'q':
            plt.close(fig)
            new_order = [original_order[idx] for idx in new_order]
            fig = _create_figure(new_order)
        happy = input("Happy? (y/n): ") == "y"

    # Save
    plt.savefig(f"{benchmark_name}_time.png")
    plt.close(fig)
