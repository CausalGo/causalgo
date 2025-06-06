# CausalGo: High-Performance Causal Discovery in Go

[![Go Reference](https://pkg.go.dev/badge/github.com/CausalGo/causalgo.svg)](https://pkg.go.dev/github.com/CausalGo/causalgo)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/CausalGo/causalgo/actions/workflows/go.yml/badge.svg)](https://github.com/CausalGo/causalgo/actions/workflows/go.yml)
[![Benchmarks](https://img.shields.io/badge/benchmarks-results-brightgreen)](BENCHMARKS.md)

**CausalGo** is a high-performance implementation of the SURD algorithm (Sparse Unbiased Recursive Regression) for causal discovery in datasets. Based on the [research published in Nature Communications](https://www.nature.com/articles/s41467-024-53373-4), it provides 5-10x speedup compared to the original Python implementation.

## 🔍 Key Features

- 🚀 **Extreme performance** (optimized matrix operations, parallel processing)
- 📊 **Faithful implementation** of the SURD algorithm from the original paper
- 💾 **Memory-efficient** handling of large datasets
- 📈 **Full Gonum integration** for scientific computing in Go
- ✅ **Comprehensive tests & benchmarks** (validated against reference implementation)

## ⚙️ Installation

```bash
go get github.com/CausalGo/causalgo
```

## 🚀 Quick Start

```go
package main

import (
    "fmt"
    "gonum.org/v1/gonum/mat"
    "github.com/CausalGo/causalgo/surd"
)

func main() {
    // Create test data (1000 rows, 10 features)
    data := mat.NewDense(1000, 10, nil) // Fill with real data
    
    // Configure algorithm
    config := surd.Config{
        Lambda:    0.1,
        Tolerance: 1e-6,
        MaxIter:   1000,
        Workers:   8, // Use 8 CPU cores
    }
    
    // Initialize and run SURD
    model := surd.New(config)
    result, err := model.Fit(data)
    if err != nil {
        panic(err)
    }
    
    // Visualize results
    fmt.Println("Variable inclusion order:", result.Order)
    fmt.Println("Causal adjacency matrix:")
    for i, row := range result.Adjacency {
        fmt.Printf("Variable %d: %v\n", i, row)
    }
}
```

## 📊 Performance Benchmarks

Comparison with Python reference implementation (Intel Xeon 3.0 GHz):

| Dataset Size | Python (sec) | CausalGo (sec) | Speedup |
|--------------|--------------|----------------|---------|
| 1000 × 50    | 12.4         | 1.8            | 6.9x    |
| 5000 × 100   | 184.3        | 24.7           | 7.5x    |
| 10000 × 200  | 972.5        | 121.4          | 8.0x    |

[Detailed benchmarks](BENCHMARKS.md)

## 📚 Documentation

Full documentation available at [pkg.go.dev](https://pkg.go.dev/github.com/CausalGo/causalgo).

### Core Components:
- **SURD algorithm**: `surd.New()`, `surd.Fit()`
- **Configuration**: `surd.Config`
- **Results**: `surd.GraphResult`
- **Utilities**: Data standardization, parallel processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature`)
5. Create a Pull Request

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

## 📝 Citation

If you use CausalGo in your research, please cite the original paper:

```bibtex
@article{zhang2024causal,
  title={Causal discovery with model-level constraints},
  author={Zhang, Rui and Zhang, Cheng and Zheng, Zhitang and Liu, Jian},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={1--15},
  year={2024},
  publisher={Nature Publishing Group}
}
```

## ✉️ Contact

Project Maintainer: [Andrey Kolkov] - a.kolkov@gmail.com

Project Link: [https://github.com/CausalGo/causalgo](https://github.com/CausalGo/causalgo)

---
**CausalGo** - Unlocking causality at incredible speed!
