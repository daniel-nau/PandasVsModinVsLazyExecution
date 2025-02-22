import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import modin.pandas as mpd  # Import Modin for comparison
import ray

# Initialize Ray for Modin
ray.init(ignore_reinit_error=True, num_cpus=8)

# 1. **Lazy Evaluation Class** implementation with caching
class LazyFrame:
    def __init__(self, data):
        """
        A class to simulate lazy evaluation for pandas DataFrames, with caching.
        """
        self.data = data
        self.operations = []  # Store queued operations
        self.cache = None  # Cache to store the result of compute

    def filter(self, condition):
        """Add a filter operation to the lazy queue."""
        # Store condition as a lambda to be applied later
        self.operations.append(lambda df: df[condition(df)])
        self.clear_cache()  # Invalidate cache when a new operation is added
        return self

    def select(self, *columns):
        """Add a select operation to pick specific columns."""
        self.operations.append(lambda df: df[list(columns)])
        self.clear_cache()  # Invalidate cache when a new operation is added
        return self

    def groupby(self, *args):
        """Add groupby operation to the lazy queue."""
        self.operations.append(lambda df: df.groupby(*args))
        self.clear_cache()  # Invalidate cache when a new operation is added
        return self

    def mean(self):
        """Add mean operation to the lazy queue."""
        self.operations.append(lambda df: df.mean())
        self.clear_cache()  # Invalidate cache when a new operation is added
        return self

    def compute(self):
        """Execute all queued operations, using cached results if available."""
        if self.cache is not None:
            return self.cache  # Return cached result if available

        result = self.data
        for op in self.operations:
            result = op(result)

        self.cache = result  # Cache the result for future use
        return result

    def clear_cache(self):
        """Clear cached result if operations change."""
        self.cache = None

# 2. **Benchmarking Class** for standard and lazy operations
class Benchmark:
    def __init__(self, df, lazy_df, modin_df):
        """
        Benchmark class to time standard and lazy operations on DataFrames.

        Attributes:
            df (pd.DataFrame): Standard pandas DataFrame.
            lazy_df (LazyFrame): LazyFrame object for lazy evaluation.
            modin_df (modin.pandas.DataFrame): Modin DataFrame for comparison.
        """
        self.df = df
        self.lazy_df = lazy_df
        self.modin_df = modin_df

    def time_operations(self, func):
        """Helper function to time the execution of a function."""
        start = timeit.default_timer()
        func()
        return timeit.default_timer() - start

    def benchmark_standard(self, num_runs=1):
        """Benchmark standard pandas operations."""
        def operation():
            self.df[self.df['a'] > 10].groupby('b').mean()

        self._warmup(operation)
        return [self.time_operations(operation) for _ in range(num_runs)]

    def benchmark_lazy(self, num_runs=1):
        """Benchmark LazyFrame operations (with caching)."""
        def operation():
            # Ensure the mean is called after groupby
            self.lazy_df.filter(lambda df: df['a'] > 10).groupby('b').mean().compute()

        self._warmup(operation)
        return [self.time_operations(operation) for _ in range(num_runs)]

    def benchmark_modin(self, num_runs=1):
        """Benchmark Modin operations."""
        def operation():
            self.modin_df[self.modin_df['a'] > 10].groupby('b').mean()

        self._warmup(operation)
        return [self.time_operations(operation) for _ in range(num_runs)]

    def _warmup(self, operation):
        """Perform a warm-up run to eliminate first run overhead."""
        operation()  # Run once to "warm up" the system

# 3. **Result Verification Function with Debugging**
def verify_results(df, lazy_df, modin_df):
    """Verify that the results of standard, lazy, and Modin evaluations match."""
    # Standard operation: Apply filter, groupby, and aggregation
    result_standard = df[df['a'] > 10].groupby('b').mean()

    # Lazy operation: Apply filter, groupby, and aggregation
    result_lazy = lazy_df.filter(lambda df: df['a'] > 10).groupby('b').compute().mean()

    # Modin operation: Apply filter, groupby, and aggregation
    result_modin = modin_df[modin_df['a'] > 10].groupby('b').mean()

    # Reset the index and align columns before comparison
    result_standard = result_standard.reset_index(drop=True)
    result_modin = pd.DataFrame(result_modin)  # Convert Modin to pandas
    result_lazy = result_lazy.reset_index(drop=True)

    # Align column names for comparison (if necessary)
    result_standard, result_modin = result_standard.align(result_modin, join='inner', axis=1)
    result_standard, result_lazy = result_standard.align(result_lazy, join='inner', axis=1)

    # Use pandas' testing tools to handle floating-point tolerance and differences in index order
    try:
        pd.testing.assert_frame_equal(result_standard, result_lazy, atol=1e-6, rtol=1e-5)
        print("Standard and Lazy results match.")
    except AssertionError as e:
        print("\nDifferences between standard and lazy evaluation:")
        print(e)

    try:
        pd.testing.assert_frame_equal(result_standard, result_modin, atol=1e-6, rtol=1e-5)
        print("Standard and Modin results match.")
    except AssertionError as e:
        print("\nDifferences between standard and Modin evaluation:")
        print(e)

    # Optionally, if you want to print where the differences are
    if not result_standard.equals(result_modin):
        print("\nRows that differ between standard and Modin:")
        diff = result_standard.compare(result_modin)
        print(diff)

    # Check if results are equal with a tolerance for floating-point errors
    assert result_standard.equals(result_lazy), "Results mismatch in LazyFrame!"
    assert result_standard.equals(result_modin), "Results mismatch in Modin!"
    print("Results are consistent between standard, lazy, and Modin evaluations.")

# 4. **Data Generation for Testing**
def generate_large_data(size=1000000):
    """Generate a large sample DataFrame with random data."""
    return pd.DataFrame({
        'a': np.random.randint(0, 100, size=size),
        'b': np.random.randint(100, 200, size=size),
        'c': np.random.random(size=size)
    })

# 5. **Plot Results Function**
def plot_results(standard_times, lazy_times, modin_times, title="Execution Time Comparison"):
    """Plot the execution times for standard, lazy, and Modin evaluation."""
    plt.figure(figsize=(10, 6))
    plt.plot(standard_times, label="Standard pandas", marker='o', linestyle='--')
    plt.plot(lazy_times, label="Lazy evaluation", marker='x', linestyle='-.')
    plt.plot(modin_times, label="Modin", marker='s', linestyle=':')
    plt.xlabel('Run #')
    plt.ylabel('Execution Time (seconds)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_comparison(standard_results, lazy_results, modin_results):
    """Visualize the comparisons between standard, lazy, and Modin operations."""
    plt.figure(figsize=(12, 8))

    # Results comparison in subplots
    plt.subplot(3, 1, 1)
    plt.plot(standard_results, label="Standard pandas", color='blue', marker='o', linestyle='--')
    plt.title("Standard Pandas Execution Time")
    plt.xlabel("Run #")
    plt.ylabel("Time (s)")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(lazy_results, label="Lazy Evaluation", color='orange', marker='x', linestyle='-.')
    plt.title("Lazy Evaluation Execution Time")
    plt.xlabel("Run #")
    plt.ylabel("Time (s)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(modin_results, label="Modin", color='green', marker='s', linestyle=':')
    plt.title("Modin Execution Time")
    plt.xlabel("Run #")
    plt.ylabel("Time (s)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 6. **Main Code for Running Benchmarks**
def main():
    # Generate sample data (you can adjust the size for testing)
    df = generate_large_data(size=2500000)
    lazy_df = LazyFrame(df)
    modin_df = mpd.DataFrame(df)

    # Initialize Benchmarking class
    benchmark = Benchmark(df, lazy_df, modin_df)

    # Perform initial run without caching
    num_runs = 50

    # Benchmark the standard pandas operation
    standard_times = benchmark.benchmark_standard(num_runs=num_runs)
    avg_standard = np.mean(standard_times)

    # Benchmark the lazy operation
    lazy_times = benchmark.benchmark_lazy(num_runs=num_runs)
    avg_lazy = np.mean(lazy_times)

    # Benchmark the Modin operation
    modin_times = benchmark.benchmark_modin(num_runs=num_runs)
    avg_modin = np.mean(modin_times)

    # Print average times
    print(f"Average time (Standard Pandas): {avg_standard:.4f} seconds")
    print(f"Average time (Lazy Evaluation): {avg_lazy:.4f} seconds")
    print(f"Average time (Modin): {avg_modin:.4f} seconds")

    # Verify if results are the same
    verify_results(df, lazy_df, modin_df)

    # Plot results comparison
    plot_results(standard_times, lazy_times, modin_times)

if __name__ == "__main__":
    main()
