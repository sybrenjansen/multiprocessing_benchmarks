import os
from collections import defaultdict
from functools import partial
from multiprocessing.managers import BaseManager

# Disable tensorflow verbose output. Needs to be done before importing tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import scipy.signal
import tensorflow as tf
from joblib import Parallel, delayed

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
    for _ in Parallel(n_jobs=n_jobs)(delayed(process_image)(image, filter_)
                                     for filter_ in filters):
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


def accumulate_prefixes(streaming_actor, document):
    streaming_actor.add_document(document)


def benchmark_2(documents, n_jobs):
    # I'm aware that doing a prefix count this way is not guaranteed to return
    # the correct prefixes over all documents. This is merely to illustrate
    # stateful computation

    # Create a manger object for shared state
    BaseManager.register("StreamingPrefixCount", StreamingPrefixCount)
    manager = BaseManager()
    manager.start()
    streaming_actors = [manager.StreamingPrefixCount() for _ in range(n_jobs)]

    Parallel(n_jobs=n_jobs)(
        delayed(accumulate_prefixes)(streaming_actors[idx % n_jobs], document)
        for idx, document in enumerate(documents)
    )

    # Get all of the popular prefixes
    popular_prefixes = set()
    for actor in streaming_actors:
        popular_prefixes |= actor.get_popular()

    manager.shutdown()


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


def benchmark_3(filename, n_jobs):
    actors = [Model(filename) for _ in range(n_jobs)]
    with Parallel(n_jobs=n_jobs, require="sharedmem") as p:
        for _ in range(benchmark_3_params["n_runs"]):
            for _ in p(delayed(actors[idx % n_jobs].evaluate_next_batch)(idx)
                       for idx in range(benchmark_3_params["n_evals"])):
                pass


##################
# Run benchmarks #
##################


def main():
    print("======")
    print("Joblib")
    print("======")
    library_name = "Joblib"

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
