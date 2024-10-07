import timeit
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Set the matplotlib backend to TkAgg for GUI plotting
matplotlib.use('TkAgg')  # For GUI plotting

# 1. **Lazy Evaluation Class** implementation
class LazyFrame:
    def __init__(self, data):
        """
        A class to simulate lazy evaluation for pandas DataFrames.

        Attributes:
            data (pd.DataFrame): The source DataFrame.
            operations (list): A list of operations to be applied lazily.
            cache (pd.DataFrame or None): Cached result after computation.
        """
        self.data = data
        self.operations = []  # Store queued operations
        self.cache = None  # Cache to store the result of compute

    def filter(self, condition):
        """Add a filter operation to the lazy queue."""
        self.operations.append(lambda df: df[condition])
        return self

    def select(self, *columns):
        """Add a select operation to pick specific columns."""
        self.operations.append(lambda df: df[columns])
        return self

    def groupby(self, *args):
        """Add groupby operation to the lazy queue."""
        self.operations.append(lambda df: df.groupby(*args))
        return self

    def compute(self):
        """Execute all queued operations, potentially using cached results."""
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
    def __init__(self, df, lazy_df):
        """
        Benchmark class to time standard and lazy operations on DataFrames.

        Attributes:
            df (pd.DataFrame): Standard pandas DataFrame.
            lazy_df (LazyFrame): LazyFrame object for lazy evaluation.
        """
        self.df = df
        self.lazy_df = lazy_df

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
        """Benchmark LazyFrame operations."""
        def operation():
            self.lazy_df.filter(self.lazy_df.data['a'] > 10).groupby('b').compute()

        self._warmup(operation)
        return [self.time_operations(operation) for _ in range(num_runs)]

    def _warmup(self, operation):
        """Perform a warm-up run to eliminate first run overhead."""
        operation()  # Run once to "warm up" the system

# 3. **Result Verification Function with Debugging**
def verify_results(df, lazy_df):
    """Verify that the results of standard and lazy evaluations match."""
    # Standard operation: Apply filter, groupby and aggregation
    result_standard = df[df['a'] > 10].groupby('b').mean()

    # Lazy operation: Apply filter, groupby, and aggregation
    result_lazy = lazy_df.filter(lazy_df.data['a'] > 10).groupby('b').compute().mean()

    # Print both results for comparison
    print("\nStandard result:")
    print(result_standard.head())  # Print a small portion of the standard result
    
    print("\nLazy result:")
    print(result_lazy.head())  # Print a small portion of the lazy result

    # Optionally, print the difference between the results
    if not result_standard.equals(result_lazy):
        print("\nDifferences between standard and lazy evaluation:")
        print(result_standard.compare(result_lazy))
    
    assert result_standard.equals(result_lazy), "Results mismatch!"
    print("Results are consistent between standard and lazy evaluation.")

# 4. **Data Generation for Testing**
def generate_large_data(size=1000000):
    """Generate a large sample DataFrame with random data."""
    return pd.DataFrame({
        'a': np.random.randint(0, 100, size=size),
        'b': np.random.randint(100, 200, size=size),
        'c': np.random.random(size=size)
    })

# 5. **Plot Results Function**
def plot_results(standard_times, lazy_times):
    """Plot the execution times for standard vs lazy evaluation."""
    plt.figure(figsize=(10, 6))
    plt.plot(standard_times, label="Standard pandas", marker='o', linestyle='--')
    plt.plot(lazy_times, label="Lazy evaluation", marker='x', linestyle='-.')
    plt.xlabel('Run #')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison: Standard vs Lazy Evaluation')
    plt.legend()
    plt.grid(True)
    plt.show()

# 6. **Main Code for Running Benchmarks**
def main():
    # Generate sample data (you can adjust the size for testing)
    df = generate_large_data(size=1000000)
    lazy_df = LazyFrame(df)

    # Initialize Benchmarking class
    benchmark = Benchmark(df, lazy_df)

    # Perform initial run without caching
    num_runs = 10

    # Benchmark the standard pandas operation (first run)
    print("Initial runs (without caching):")

    standard_times_initial = benchmark.benchmark_standard(num_runs=num_runs)
    avg_standard_initial = np.mean(standard_times_initial)
    print(f"Standard pandas average execution time (initial): {avg_standard_initial:.4f} seconds")

    # Benchmark the lazy operation (first run, no caching)
    lazy_times_initial = benchmark.benchmark_lazy(num_runs=num_runs)
    avg_lazy_initial = np.mean(lazy_times_initial)
    print(f"Lazy evaluation average execution time (initial): {avg_lazy_initial:.4f} seconds")

    # Perform subsequent runs with caching
    print("\nSubsequent runs with caching (after first run):")

    standard_times_subsequent = benchmark.benchmark_standard(num_runs=num_runs)
    avg_standard_subsequent = np.mean(standard_times_subsequent)
    print(f"Standard pandas average execution time (subsequent): {avg_standard_subsequent:.4f} seconds")

    lazy_times_subsequent = benchmark.benchmark_lazy(num_runs=num_runs)
    avg_lazy_subsequent = np.mean(lazy_times_subsequent)
    print(f"Lazy evaluation average execution time (subsequent with caching): {avg_lazy_subsequent:.4f} seconds")

    # Verify results to ensure correctness
    verify_results(df, lazy_df)

    # Plot the results for visualization
    plot_results(standard_times_subsequent, lazy_times_subsequent)

if __name__ == "__main__":
    main()
