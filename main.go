package main

import (
	"fmt"
	"github.com/CausalGo/causalgo/surd"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Test with small dataset
	data := []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	X := mat.NewDense(3, 3, data)

	config := surd.Config{
		Lambda:    0.1,
		Tolerance: 1e-6,
		MaxIter:   1000,
		Workers:   4,
	}

	model := surd.New(config)
	result, err := model.Fit(X)
	if err != nil {
		panic(err)
	}

	fmt.Println("Order:", result.Order)
	fmt.Println("Residuals:", result.Residuals)
}
