import os
from collections import defaultdict
from functools import partial
from typing import Optional

# Disable tensorflow verbose output. Needs to be done before importing tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import scipy.signal
import tensorflow as tf
from dask.distributed import Client, get_worker

from params import benchmark_3_params
from util import (create_and_save_tf_model, create_documents, create_image,
                  create_image_filters, run_trials)

DASK_CLIENT = None  # type: Optional[Client]


def init_dask(n_jobs: int) -> None:
    # Usually, you have to avoid using globals, but this was the most convenient
    global DASK_CLIENT
    DASK_CLIENT = Client(n_workers=n_jobs, threads_per_worker=1)


def shutdown_dask() -> None:
    global DASK_CLIENT
    DASK_CLIENT.close()


######################################
# Benchmark 1: numerical computation #
######################################


def process_image(image, random_filter):
    # Do some image processing.
    return scipy.signal.convolve2d(image, random_filter)[::5, ::5]


def benchmark_1(filters, _):
    image = create_image()
    image_b = DASK_CLIENT.scatter(image, broadcast=True)
    for _ in DASK_CLIENT.gather([DASK_CLIENT.submit(process_image, image_b,
                                                    filter_)
                                 for filter_ in filters]):
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


def accumulate_prefixes(document):
    worker = get_worker()
    if not hasattr(worker, "streaming_actor"):
        worker.streaming_actor = StreamingPrefixCount()
    if document is None:
        return worker.streaming_actor.get_popular()
    else:
        worker.streaming_actor.add_document(document)


def benchmark_2(documents, n_jobs):
    # I'm aware that doing a prefix count this way is not guaranteed to return
    # the correct prefixes over all documents. This is merely to illustrate
    # stateful computation

    # I would implement this using Dask actors, but that implementation
    # (provided below) failed to run for some unknown reason.
    DASK_CLIENT.gather([DASK_CLIENT.submit(accumulate_prefixes, document)
                        for document in documents])

    # Add sentinel to each individual worker
    workers = DASK_CLIENT.scheduler_info()['workers'].keys()
    futures = [DASK_CLIENT.submit(accumulate_prefixes, None, workers=[worker])
               for worker in workers]

    # Aggregate
    popular_prefixes = set()
    for future in futures:
        popular_prefixes |= future.result()


def benchmark_2_actors(documents, n_jobs):
    # I'm aware that doing a prefix count this way is not guaranteed to return
    # the correct prefixes over all documents. This is merely to illustrate
    # stateful computation
    streaming_actors = [DASK_CLIENT.submit(StreamingPrefixCount,
                                           actor=True).result()
                        for _ in range(n_jobs)]

    # Client.gather() doesn't work for ActorFuture objects
    futures = [streaming_actors[idx % n_jobs].add_document(document)
               for idx, document in enumerate(documents)]
    [future.result() for future in futures]

    # Aggregate
    popular_prefixes = set()
    for actor in streaming_actors:
        popular_prefixes |= actor.get_popular().result()


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


def evaluate_next_batch(filename, idx):
    worker = get_worker()
    if not hasattr(worker, 'model_actor'):
        worker.model_actor = Model(filename)
    return worker.model_actor.evaluate_next_batch(idx)


def benchmark_3(filename, n_jobs):
    # We run it multiple times to better see the effect of initialization.
    for _ in range(benchmark_3_params["n_runs"]):
        for _ in DASK_CLIENT.gather([
            DASK_CLIENT.submit(evaluate_next_batch, filename, idx)
            for idx in range(benchmark_3_params["n_evals"])
        ]):
            pass


def benchmark_3_actors(filename, n_jobs):
    actors = [DASK_CLIENT.submit(Model, filename, actor=True).result()
              for _ in range(n_jobs)]

    # We run it multiple times to better see the effect of initialization.
    # Client.gather() doesn't work for ActorFuture objects
    for _ in range(benchmark_3_params["n_runs"]):
        futures = [actors[idx % n_jobs].evaluate_next_batch()
                   for idx in range(benchmark_3_params["n_evals"])]
        for future in futures:
            future.result()


##################
# Run benchmarks #
##################


def main():
    print("====")
    print("Dask")
    print("====")
    library_name = "Dask"

    print("Setting up benchmark 1 ...")
    filters = create_image_filters()

    print("Benchmark #1 started")
    run_trials(partial(benchmark_1, filters), "Numerical computation",
               library_name, init_function=init_dask,
               exit_function=shutdown_dask)

    print("Setting up benchmark 2 ...")
    documents = create_documents()

    print("Benchmark #2 started")
    run_trials(partial(benchmark_2, documents), "Stateful computation",
               library_name, init_function=init_dask,
               exit_function=shutdown_dask)

    print("Setting up benchmark 3 ...")
    filename = "/tmp/model"
    create_and_save_tf_model(filename)

    print("Benchmark #3 started")
    run_trials(partial(benchmark_3, filename), "Expensive initialization",
               library_name, init_function=init_dask,
               exit_function=shutdown_dask)


if __name__ == "__main__":
    main()
