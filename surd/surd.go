package surd

import (
	"math"
	"sync"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// Config parameters for SURD algorithm
type Config struct {
	Lambda    float64 // LASSO regularization parameter
	Tolerance float64 // Convergence criterion
	MaxIter   int     // Maximum iterations
	Workers   int     // Number of goroutines for parallel processing
	Verbose   bool    // Verbose output mode
}

// GraphResult represents causal discovery results
type GraphResult struct {
	Adjacency [][]bool    // Adjacency matrix of causal graph
	Order     []int       // Variable inclusion order
	Residuals []float64   // Residual variances
	Weights   [][]float64 // Causal weights
}

// SURD main type for causal discovery
type SURD struct {
	config Config
}

// New creates a new SURD instance
func New(cfg Config) *SURD {
	if cfg.Lambda <= 0 {
		cfg.Lambda = 0.01
	}
	if cfg.Tolerance <= 0 {
		cfg.Tolerance = 1e-5
	}
	if cfg.MaxIter <= 0 {
		cfg.MaxIter = 1000
	}
	if cfg.Workers <= 0 {
		cfg.Workers = 4
	}

	return &SURD{config: cfg}
}

// Fit performs causal discovery on input data
func (s *SURD) Fit(X *mat.Dense) (*GraphResult, error) {
	n, p := X.Dims()

	// Standardize data
	stdX := s.standardize(X)

	// Initialize result structures
	result := &GraphResult{
		Adjacency: make([][]bool, p),
		Order:     make([]int, 0, p),
		Residuals: make([]float64, p),
		Weights:   make([][]float64, p),
	}
	for i := range result.Adjacency {
		result.Adjacency[i] = make([]bool, p)
		result.Weights[i] = make([]float64, p)
	}

	// Track remaining variables
	remaining := make([]bool, p)
	for i := range remaining {
		remaining[i] = true
	}

	// Main recursive loop
	for len(result.Order) < p {
		// Handle last variable separately
		activeCount := 0
		for _, rem := range remaining {
			if rem {
				activeCount++
			}
		}
		if activeCount == 1 {
			for j := 0; j < p; j++ {
				if remaining[j] {
					result.Order = append(result.Order, j)
					col := make([]float64, n)
					mat.Col(col, j, stdX)
					result.Residuals[len(result.Order)-1] = computeMSE(col)
					remaining[j] = false
					break
				}
			}
			continue
		}

		type varResult struct {
			idx     int
			mse     float64
			weights []float64
		}

		results := make(chan varResult, p)
		var wg sync.WaitGroup
		sem := make(chan struct{}, s.config.Workers)

		for j := 0; j < p; j++ {
			if !remaining[j] {
				continue
			}

			wg.Add(1)
			sem <- struct{}{}

			go func(j int) {
				defer wg.Done()
				defer func() { <-sem }()

				// Prepare data for LASSO
				Xsub, y := s.prepareData(stdX, j, remaining)

				// Skip if no predictors available
				if Xsub == nil {
					results <- varResult{idx: j, mse: math.MaxFloat64}
					return
				}

				// Run LASSO regression
				weights := s.lassoRegression(Xsub, y)

				// Calculate residuals and MSE
				residuals := s.calculateResiduals(Xsub, y, weights)
				mse := computeMSE(residuals)

				results <- varResult{
					idx:     j,
					mse:     mse,
					weights: weights,
				}
			}(j)
		}

		go func() {
			wg.Wait()
			close(results)
		}()

		// Find best variable
		bestVar := -1
		bestMSE := math.MaxFloat64
		var bestWeights []float64

		for res := range results {
			if res.mse < bestMSE {
				bestMSE = res.mse
				bestVar = res.idx
				bestWeights = res.weights
			}
		}

		// Update results
		result.Order = append(result.Order, bestVar)
		result.Residuals[len(result.Order)-1] = bestMSE

		// Update weights and adjacency
		idx := 0
		for j := 0; j < p; j++ {
			if !remaining[j] || j == bestVar {
				continue
			}

			if math.Abs(bestWeights[idx]) > s.config.Tolerance {
				result.Adjacency[bestVar][j] = true
				result.Weights[bestVar][j] = bestWeights[idx]
			}
			idx++
		}

		// Mark variable as processed
		remaining[bestVar] = false
	}

	return result, nil
}

// prepareData prepares matrices for regression
func (s *SURD) prepareData(X *mat.Dense, target int, remaining []bool) (*mat.Dense, []float64) {
	n, p := X.Dims()

	// Count active variables
	activeCount := 0
	for _, rem := range remaining {
		if rem {
			activeCount++
		}
	}

	// Return nil if no predictors available
	predCount := activeCount - 1
	if predCount <= 0 {
		return nil, nil
	}

	// Create predictor matrix
	Xmat := mat.NewDense(n, predCount, nil)
	y := make([]float64, n)

	// Fill data
	colIdx := 0
	for j := 0; j < p; j++ {
		if !remaining[j] || j == target {
			continue
		}

		col := make([]float64, n)
		mat.Col(col, j, X)
		Xmat.SetCol(colIdx, col)
		colIdx++
	}

	// Target variable
	mat.Col(y, target, X)

	return Xmat, y
}

// lassoRegression implements LASSO with coordinate descent
func (s *SURD) lassoRegression(X *mat.Dense, y []float64) []float64 {
	n, p := X.Dims()
	weights := make([]float64, p)

	// Cache X^T for performance
	XT := mat.DenseCopyOf(X.T())

	for iter := 0; iter < s.config.MaxIter; iter++ {
		maxChange := 0.0

		// Process all features
		for j := 0; j < p; j++ {
			// Compute partial residual
			residual := s.calculatePartialResidual(X, y, weights, j)

			// Compute weight update
			col := XT.RawRowView(j)
			rDot := floats.Dot(col, residual)

			// Apply soft thresholding
			newWeight := softThreshold(rDot/float64(n), s.config.Lambda)

			// Update weight
			delta := math.Abs(newWeight - weights[j])
			if delta > maxChange {
				maxChange = delta
			}
			weights[j] = newWeight
		}

		// Check convergence
		if maxChange < s.config.Tolerance {
			break
		}
	}

	return weights
}

// calculatePartialResidual computes residual without feature j
func (s *SURD) calculatePartialResidual(X *mat.Dense, y, weights []float64, exclude int) []float64 {
	n, _ := X.Dims()
	residual := make([]float64, n)
	copy(residual, y)

	// Subtract contributions from all features except excluded
	for j := 0; j < len(weights); j++ {
		if j == exclude {
			continue
		}

		col := make([]float64, n)
		mat.Col(col, j, X)
		floats.AddScaled(residual, -weights[j], col)
	}

	return residual
}

// calculateResiduals computes regression residuals
func (s *SURD) calculateResiduals(X *mat.Dense, y, weights []float64) []float64 {
	n, _ := X.Dims()
	residuals := make([]float64, n)

	// Compute predictions
	predictions := mat.NewVecDense(n, nil)
	predictions.MulVec(X, mat.NewVecDense(len(weights), weights))

	// Compute residuals
	for i := 0; i < n; i++ {
		residuals[i] = y[i] - predictions.AtVec(i)
	}

	return residuals
}

// standardize normalizes data
func (s *SURD) standardize(X *mat.Dense) *mat.Dense {
	n, p := X.Dims()
	stdX := mat.DenseCopyOf(X)

	for j := 0; j < p; j++ {
		col := make([]float64, n)
		mat.Col(col, j, stdX)

		mean := stat.Mean(col, nil)
		stddev := stat.StdDev(col, nil)

		if stddev < 1e-8 {
			stddev = 1
		}

		for i := range col {
			col[i] = (col[i] - mean) / stddev
		}
		stdX.SetCol(j, col)
	}

	return stdX
}

// computeMSE calculates mean squared error
func computeMSE(residuals []float64) float64 {
	n := len(residuals)
	if n == 0 {
		return 0
	}
	sum := 0.0
	for _, r := range residuals {
		sum += r * r
	}
	return sum / float64(n)
}

// softThreshold applies soft thresholding operator
func softThreshold(z, lambda float64) float64 {
	if z > lambda {
		return z - lambda
	} else if z < -lambda {
		return z + lambda
	}
	return 0
}
