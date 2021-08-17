import os
import sys
from collections import defaultdict
from functools import partial

# Disable tensorflow verbose output. Needs to be done before importing tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import psutil
import ray
import scipy.signal
import tensorflow as tf
from ray.util import ActorPool

from params import benchmark_3_params
from util import (create_and_save_tf_model, create_documents, create_image,
                  create_image_filters, run_trials)


def init_ray(n_jobs):
    ray.init(num_cpus=n_jobs)


def shutdown_ray():
    ray.shutdown()


######################################
# Benchmark 1: numerical computation #
######################################


@ray.remote
def process_image(image, random_filter):
    # Do some image processing.
    return scipy.signal.convolve2d(image, random_filter)[::5, ::5]


def benchmark_1(filters, n_jobs):
    image = create_image()
    image_id = ray.put(image)
    for _ in ray.get([process_image.remote(image_id, filter_)
                      for filter_ in filters]):
        pass


#####################################
# Benchmark 2: stateful computation #
#####################################


@ray.remote
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
    streaming_actors = [StreamingPrefixCount.remote() for _ in range(n_jobs)]
    pool = ActorPool(streaming_actors)
    list(pool.map_unordered(lambda actor, doc: actor.add_document.remote(doc),
                            documents))

    # Aggregate all of the results.
    popular_prefixes = set()
    for prefixes in ray.get([actor.get_popular.remote()
                             for actor in streaming_actors]):
        popular_prefixes |= prefixes


#########################################
# Benchmark 3: expensive initialization #
#########################################


@ray.remote
class Model:
    def __init__(self, filename, i):
        # Pin the actor to a specific core if we are on Linux to prevent
        # contention between the different actors since TensorFlow uses
        # multiple threads. This helps for Ray, but is not needed for the others
        if sys.platform == "linux":
            psutil.Process().cpu_affinity([i])

        # Load the model and some data.
        self.model = tf.keras.models.load_model(filename)
        mnist = tf.keras.datasets.mnist.load_data()
        self.x_test = mnist[1][0] / 255.0

    def evaluate_next_batch(self, idx):
        # Note that we reuse the same data over and over, but in a real
        # application, the data would be different each time. To simulate the
        # latter we add idx (this avoids libraries to use caching mechanisms)
        return self.model.predict(self.x_test) + idx


def benchmark_3(filename, n_jobs):
    actors = [Model.remote(filename, i) for i in range(n_jobs)]
    pool = ActorPool(actors)

    # We run it multiple times to better see the effect of initialization
    for _ in range(benchmark_3_params["n_runs"]):
        for _ in pool.map(
                lambda actor, idx: actor.evaluate_next_batch.remote(idx),
                range(benchmark_3_params["n_evals"])
        ):
            pass


##################
# Run benchmarks #
##################


def main():
    print("===")
    print("Ray")
    print("===")
    library_name = "Ray"

    print("Setting up benchmark 1 ...")
    filters = create_image_filters()

    print("Benchmark #1 started")
    run_trials(partial(benchmark_1, filters), "Numerical computation",
               library_name, init_function=init_ray, exit_function=shutdown_ray)

    print("Setting up benchmark 2 ...")
    documents = create_documents()

    print("Benchmark #2 started")
    run_trials(partial(benchmark_2, documents), "Stateful computation",
               library_name, init_function=init_ray, exit_function=shutdown_ray)

    print("Setting up benchmark 3 ...")
    filename = "/tmp/model"
    create_and_save_tf_model(filename)

    print("Benchmark #3 started")
    run_trials(partial(benchmark_3, filename), "Expensive initialization",
               library_name, init_function=init_ray, exit_function=shutdown_ray)


if __name__ == "__main__":
    main()
