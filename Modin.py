import time
import pandas as pd

# 1. **Lazy Evaluation Class** implementation
class LazyFrame:
    def __init__(self, data):
        self.data = data  # pandas DataFrame
        self.operations = []  # list to store the operations

    def filter(self, condition):
        # Add filtering operation to the queue
        self.operations.append(lambda df: df[condition])
        return self

    def groupby(self, *args):
        # Add groupby operation to the queue
        self.operations.append(lambda df: df.groupby(*args))
        return self

    def compute(self):
        # Execute all queued operations
        result = self.data
        for op in self.operations:
            result = op(result)
        return result


# 2. **Benchmarking Functions**

def benchmark_standard(df):
    """Benchmark standard DataFrame operations"""
    start = time.time()
    result = df[df['a'] > 10].groupby('b').mean()
    return time.time() - start

def benchmark_lazy(lazy_df):
    """Benchmark Lazy DataFrame operations"""
    start = time.time()
    result = lazy_df.filter(lazy_df.data['a'] > 10).groupby('b').compute()
    return time.time() - start


# 3. **Create sample data**
df = pd.DataFrame({
    'a': range(100000),  # 100,000 rows
    'b': range(100000, 200000)
})

# Initialize the lazy dataframe
lazy_df = LazyFrame(df)

# 4. **Run Benchmarks**

# Benchmark the standard pandas operation
standard_time = benchmark_standard(df)
print(f"Standard pandas execution time: {standard_time:.4f} seconds")

# Benchmark the lazy operation
lazy_time = benchmark_lazy(lazy_df)
print(f"Lazy evaluation execution time: {lazy_time:.4f} seconds")
