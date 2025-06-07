package surd

import (
	"math/rand"
	"testing"
	"time"

	"gonum.org/v1/gonum/mat"
)

// BenchmarkSURD benchmarks performance for various dataset sizes
func BenchmarkSURD(b *testing.B) {
	rand.Seed(time.Now().UnixNano())

	benchmarks := []struct {
		name string
		rows int
		cols int
	}{
		{"100x10", 100, 10},     // Small dataset
		{"1000x50", 1000, 50},   // Medium dataset
		{"5000x100", 5000, 100}, // Large dataset
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			// Generate random data
			data := make([]float64, bm.rows*bm.cols)
			for i := range data {
				data[i] = rand.NormFloat64()
			}
			X := mat.NewDense(bm.rows, bm.cols, data)

			// Create SURD model
			model := New(Config{
				Lambda:    0.1,
				Workers:   8,
				MaxIter:   1000,
				Tolerance: 1e-5,
			})

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := model.Fit(X)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
