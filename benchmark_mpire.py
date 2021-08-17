import os
from collections import defaultdict
from functools import partial

# Disable tensorflow verbose output. Needs to be done before importing tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import scipy.signal
import tensorflow as tf
from mpire import WorkerPool
from mpire.utils import make_single_arguments

from params import benchmark_3_params
from util import (create_and_save_tf_model, create_documents, create_image,
                  create_image_filters, run_trials)


######################################
# Benchmark 1: numerical computation #
######################################


def process_image(image, random_filter):
    # Do some image processing.
    return scipy.signal.convolve2d(image, random_filter)[::5, ::5]


def benchmark_1(filters, n_jobs):
    image = create_image()
    with WorkerPool(n_jobs, shared_objects=image) as pool:
        for _ in pool.imap(process_image, filters):
            pass


#####################################
# Benchmark 2: stateful computation #
#####################################


class StreamingPrefixCount:
    def __init__(self):
        self.prefix_count = defaultdict(int)

    def add_document(self, document):
        for word in document:
            for i in range(1, len(word)):
                prefix = word[:i]
                self.prefix_count[prefix] += 1

    def get_popular(self):
        return {prefix for prefix, cnt in self.prefix_count.items() if cnt > 3}


def benchmark_2(documents, n_jobs):
    # I'm aware that doing a prefix count this way is not guaranteed to return
    # the correct prefixes over all documents. This is merely to illustrate
    # stateful computation

    # MPIRE makes use of copy-on-write, so when the streaming_actor gets updated
    # within a worker it will get copied and each worker will have its own copy.
    # If your system can't use fork, you can make use of worker state instead,
    # which is as fast
    streaming_actor = StreamingPrefixCount()
    with WorkerPool(n_jobs, start_method="fork") as pool:
        pool.map_unordered(streaming_actor.add_document,
                           make_single_arguments(documents),
                           iterable_len=len(documents),
                           worker_exit=streaming_actor.get_popular)

        # Aggregate all of the results.
        popular_prefixes = set()
        for prefixes in pool.get_exit_results():
            popular_prefixes |= prefixes


#########################################
# Benchmark 3: expensive initialization #
#########################################


class Model:
    def __init__(self, filename):
        # Load the model and some data.
        self.model = tf.keras.models.load_model(filename)
        mnist = tf.keras.datasets.mnist.load_data()
        self.x_test = mnist[1][0] / 255.0

    def evaluate_next_batch(self, idx):
        # Note that we reuse the same data over and over, but in a real
        # application, the data would be different each time. To simulate the
        # latter we add idx (this avoids libraries to use caching mechanisms)
        return self.model.predict(self.x_test) + idx


def worker_init_3(filename, worker_state):
    worker_state["model"] = Model(filename)


def evaluate_next_batch(worker_state, idx):
    return worker_state["model"].evaluate_next_batch(idx)


def benchmark_3(filename, n_jobs):
    # Pin the actor to a specific core to prevent contention between the
    # different actors since TensorFlow uses multiple threads.
    # For this benchmark we can't make use of the forking benefits as used in
    # benchmark #1, as TensorFlow is buggy when doing that. Fortunately, using
    # this approach of utilizing worker state is equally fast
    with WorkerPool(n_jobs, use_worker_state=True, keep_alive=True,
                    cpu_ids=list(range(n_jobs))) as pool:
        # We run it multiple times to better see the effect of initialization
        worker_init = partial(worker_init_3, filename)
        for _ in range(benchmark_3_params["n_runs"]):
            for _ in pool.imap_unordered(evaluate_next_batch,
                                         range(benchmark_3_params["n_evals"]),
                                         worker_init=worker_init):
                pass


##################
# Run benchmarks #
##################


def main():
    print("=====")
    print("MPIRE")
    print("=====")
    library_name = "MPIRE"

    print("Setting up benchmark 1 ...")
    filters = create_image_filters()

    print("Benchmark #1 started")
    run_trials(partial(benchmark_1, filters), "Numerical computation",
               library_name)

    print("Setting up benchmark 2 ...")
    documents = create_documents()

    print("Benchmark #2 started")
    run_trials(partial(benchmark_2, documents), "Stateful computation",
               library_name)

    print("Setting up benchmark 3 ...")
    filename = "/tmp/model"
    create_and_save_tf_model(filename)

    print("Benchmark #3 started")
    run_trials(partial(benchmark_3, filename), "Expensive initialization",
               library_name)


if __name__ == "__main__":
    main()
