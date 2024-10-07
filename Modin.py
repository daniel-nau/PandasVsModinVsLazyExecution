import time
import pandas as pd
import numpy as np

# 1. **Lazy Evaluation Class** implementation
class LazyFrame:
    def __init__(self, data):
        self.data = data  # pandas DataFrame
        self.operations = []  # list to store the operations
        self.cache = None  # to store cached results

    def filter(self, condition):
        """Add filter operation to the lazy queue."""
        self.operations.append(lambda df: df[condition])
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

# 2. **Benchmarking Functions**

def time_operations(func, *args):
    """Helper function to time the operations."""
    start = time.perf_counter()
    func(*args)
    return time.perf_counter() - start

def benchmark_standard(df, num_runs=1):
    """Benchmark standard DataFrame operations for a given number of runs."""
    def operation():
        df[df['a'] > 10].groupby('b').mean()
    
    # Warm-up: run once to allow for caching or other optimizations
    operation()
    
    # Time multiple runs
    times = [time_operations(operation) for _ in range(num_runs)]
    return times

def benchmark_lazy(lazy_df, num_runs=1):
    """Benchmark Lazy DataFrame operations for a given number of runs."""
    def operation():
        lazy_df.filter(lazy_df.data['a'] > 10).groupby('b').compute()
    
    # Warm-up: run once to allow for caching or other optimizations
    operation()
    
    # Time multiple runs
    times = [time_operations(operation) for _ in range(num_runs)]
    return times

# 3. **Create sample data**
df = pd.DataFrame({
    'a': range(100000),  # 100,000 rows
    'b': range(100000, 200000)
})

# Initialize the lazy dataframe
lazy_df = LazyFrame(df)

# 4. **Run Benchmarks with Multiple Runs**
num_runs = 10

# 4.1 Initial run without caching
print("Initial runs (without caching):")

# Benchmark the standard pandas operation (first run)
standard_times_initial = benchmark_standard(df, num_runs=num_runs)
avg_standard_initial = np.mean(standard_times_initial)
median_standard_initial = np.median(standard_times_initial)
print(f"Standard pandas average execution time (initial): {avg_standard_initial:.4f} seconds")
print(f"Standard pandas median execution time (initial): {median_standard_initial:.4f} seconds")

# Benchmark the lazy operation (first run, no caching)
lazy_times_initial = benchmark_lazy(lazy_df, num_runs=num_runs)
avg_lazy_initial = np.mean(lazy_times_initial)
median_lazy_initial = np.median(lazy_times_initial)
print(f"Lazy evaluation average execution time (initial): {avg_lazy_initial:.4f} seconds")
print(f"Lazy evaluation median execution time (initial): {median_lazy_initial:.4f} seconds")

# 4.2 Subsequent runs with caching (for lazy evaluation)
print("\nSubsequent runs with caching (after first run):")

# Benchmark the standard pandas operation again (no caching in pandas)
standard_times_subsequent = benchmark_standard(df, num_runs=num_runs)
avg_standard_subsequent = np.mean(standard_times_subsequent)
median_standard_subsequent = np.median(standard_times_subsequent)
print(f"Standard pandas average execution time (subsequent): {avg_standard_subsequent:.4f} seconds")
print(f"Standard pandas median execution time (subsequent): {median_standard_subsequent:.4f} seconds")

# Benchmark the lazy operation again (now caching should speed things up for lazy)
lazy_times_subsequent = benchmark_lazy(lazy_df, num_runs=num_runs)
avg_lazy_subsequent = np.mean(lazy_times_subsequent)
median_lazy_subsequent = np.median(lazy_times_subsequent)
print(f"Lazy evaluation average execution time (subsequent with caching): {avg_lazy_subsequent:.4f} seconds")
print(f"Lazy evaluation median execution time (subsequent with caching): {median_lazy_subsequent:.4f} seconds")
