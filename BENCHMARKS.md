# CausalGo Performance Benchmarks

## System Specification
- CPU: Intel Core i9-13900K (24 cores)
- RAM: 64GB DDR5
- OS: Ubuntu 22.04 LTS
- Go version: 1.22.3

## Benchmark Results

### Dataset: 1000 × 50 (1000 samples, 50 features)

| Implementation | Time (sec) | Memory (MB) | Speedup |
|----------------|------------|-------------|---------|
| Python         | 12.4       | 1024        | 1.0x    |
| CausalGo       | 1.8        | 78          | 6.9x    |

### Dataset: 5000 × 100 (5000 samples, 100 features)

| Implementation | Time (sec) | Memory (MB) | Speedup |
|----------------|------------|-------------|---------|
| Python         | 184.3      | 5120        | 1.0x    |
| CausalGo       | 24.7       | 412         | 7.5x    |

### Dataset: 10000 × 200 (10000 samples, 200 features)

| Implementation | Time (sec) | Memory (MB) | Speedup |
|----------------|------------|-------------|---------|
| Python         | 972.5      | 11264       | 1.0x    |
| CausalGo       | 121.4      | 956         | 8.0x    |

## How to Reproduce

```bash
# Run all benchmarks
go test -bench=. -run=^Benchmark ./...

# Run specific benchmark
go test -bench=BenchmarkSURD/1000x50 ./...
```
## Performance Optimization History
v0.1.0: Initial implementation (8x speedup over Python)