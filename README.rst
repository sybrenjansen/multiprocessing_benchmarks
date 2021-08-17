Multiprocessing benchmarks
==========================

This repository contains several benchmarks for Python multiprocessing
libraries and serves as complementary material for the `blog post on MPIRE`_.
The benchmarks are inspired by this `GitHub Gist`_.

.. _blog post on MPIRE: TODO
.. _GitHub Gist: https://gist.github.com/robertnishihara/2b81595abd4f50a049767a040ce435ab

The tested libraries include:

- Serial processing (not a library)
- multiprocessing.Pool
- concurrent.futures.ProcessPoolExecutor
- Joblib
- Dask
- Ray
- MPIRE

How to run
----------

Make sure there's no interference from other processes on your machine before
you run these benchmarks.

All benchmarks are parameterized in ``params.py``. Change these numbers to your
liking.

1. Install the requirements from the ``requirements.txt`` file.
2. Run each ``benchmark_<library>.p3`` script. This runs each benchmark for a
   a single library and stores a summary of the results to disk.
3. Use the ``utils.visualize_results_for_benchmark`` function to create a
   visualization of the results for a single benchmark. This function is
   interactive and lets you position the labels to your liking. Follow the
   instructions in the terminal.

Benchmarks
----------

1. Numerical computation: processes an image using different image filters. The
   image remains the same for each filter. Therefore, libraries that can somehow
   send the image to each process once have a clear advantage.
2. Stateful computation: each worker keeps track of its own state and should
   update it whenever new tasks come in. The task is about processing text
   documents and keeping track of word prefix counts, up to a size of 3
   characters. Whenever a certain prefix occurs more than 3 times, that prefix
   should be returned once all documents have been processed. Libraries that can
   store local data for each worker and return data when all the work is done
   are clearly the most suitable for this task.
3. Expensive initialization: a neural network model is used to predict labels on
   some image dataset. Loading this model only takes a few seconds, but if it
   has to be done for each task it quickly adds up. Although this benchmark
   seems similar to the previous one, this benchmark doesn't require keeping
   track of changes in the worker state.
