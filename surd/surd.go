package surd

import (
	"fmt"
	"math"
	"sync"

	"github.com/CausalGo/causalgo/regression"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// Config stores parameters for the SURD algorithm
type Config struct {
	Lambda    float64 // LASSO regularization parameter (passed to default regressor)
	Tolerance float64 // Convergence criterion for coordinate descent
	MaxIter   int     // Maximum iterations for coordinate descent
	Workers   int     // Number of parallel workers
	Verbose   bool    // Enable verbose logging
}

// GraphResult represents causal discovery results
type GraphResult struct {
	Adjacency [][]bool    // Adjacency matrix (true = causal link)
	Order     []int       // Variable ordering (causal sequence)
	Residuals []float64   // Residual variances at each step
	Weights   [][]float64 // Causal weights matrix
}

// SURD implements Sparse Unbiased Recursive Regression for causal discovery
type SURD struct {
	config    Config
	regressor regression.Regressor // Regression model (default: LASSO)
}

// New creates a new SURD instance with default LASSO regressor
func New(cfg Config) *SURD {
	// Set default values for missing parameters
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

	// Create default LASSO regressor with validated config
	lassoReg := regression.NewLASSO(regression.LASSOConfig{
		Lambda:    cfg.Lambda,
		Tolerance: cfg.Tolerance,
		MaxIter:   cfg.MaxIter,
	})

	return &SURD{
		config:    cfg,
		regressor: lassoReg,
	}
}

// SetRegressor sets a custom regressor implementation
// Allows extending SURD with different regression models
func (s *SURD) SetRegressor(r regression.Regressor) {
	s.regressor = r
}

// Fit performs causal discovery on input data
// X: n x p matrix (n samples, p variables)
// Returns: causal graph and computation results
func (s *SURD) Fit(x *mat.Dense) (*GraphResult, error) {
	// Validate input matrix
	if x == nil {
		return nil, fmt.Errorf("nil input matrix")
	}

	n, p := x.Dims()

	// Handle edge cases
	if n == 0 || p == 0 {
		return nil, fmt.Errorf("empty input matrix")
	}
	if n < 2 {
		return nil, fmt.Errorf("need at least 2 rows, got %d", n)
	}

	// Standardize data to zero mean and unit variance
	stdX := s.standardize(x)

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

	// Main recursive loop - select one variable per iteration
	for len(result.Order) < p {
		activeCount := countActive(remaining)

		// Handle last variable separately
		if activeCount == 1 {
			processLastVariable(stdX, result, remaining, n)
			continue
		}

		// Parallel processing of active variables
		results := s.processVariables(stdX, remaining, n, p)

		// Find best variable (lowest MSE)
		bestVar, bestMSE, bestWeights := findBestVariable(results)

		// Update results with best variable
		s.updateResults(result, bestVar, bestMSE, bestWeights, remaining, p)
	}

	return result, nil
}

// prepareData prepares matrices for regression
// target: index of target variable
// remaining: active variables mask
// Returns: predictor matrix and target vector
func (s *SURD) prepareData(x *mat.Dense, target int, remaining []bool) (*mat.Dense, []float64) {
	n, p := x.Dims()
	activeCount := countActive(remaining)

	// Get target vector
	y := make([]float64, n)
	mat.Col(y, target, x)

	// Handle case with no predictors
	predCount := activeCount - 1
	if predCount <= 0 {
		return nil, y
	}

	// Create predictor matrix
	xMat := mat.NewDense(n, predCount, nil)
	colIdx := 0

	// Collect all active predictors except target
	for j := 0; j < p; j++ {
		if !remaining[j] || j == target {
			continue
		}

		col := make([]float64, n)
		mat.Col(col, j, x)
		xMat.SetCol(colIdx, col)
		colIdx++
	}

	return xMat, y
}

// calculateResiduals computes regression residuals
func (s *SURD) calculateResiduals(X *mat.Dense, y, weights []float64) []float64 {
	n, _ := X.Dims()
	residuals := make([]float64, n)

	// Vectorized computation: residuals = y - Xw
	predictions := mat.NewVecDense(n, nil)
	predictions.MulVec(X, mat.NewVecDense(len(weights), weights))

	for i := 0; i < n; i++ {
		residuals[i] = y[i] - predictions.AtVec(i)
	}

	return residuals
}

// standardize normalizes data to zero mean and unit population variance
func (s *SURD) standardize(X *mat.Dense) *mat.Dense {
	n, p := X.Dims()
	stdX := mat.NewDense(n, p, nil)
	stdX.Copy(X)

	for j := 0; j < p; j++ {
		col := mat.Col(nil, j, stdX)
		mean := floats.Sum(col) / float64(n)

		// Calculate population standard deviation (division by n)
		variance := 0.0
		for _, v := range col {
			diff := v - mean
			variance += diff * diff
		}
		stddev := math.Sqrt(variance / float64(n))

		// Handle constant columns
		if stddev < 1e-12 {
			for i := 0; i < n; i++ {
				stdX.Set(i, j, 0)
			}
		} else {
			for i := 0; i < n; i++ {
				val := (stdX.At(i, j) - mean) / stddev
				stdX.Set(i, j, val)
			}
		}
	}
	return stdX
}

// Helper function: count active variables
func countActive(remaining []bool) int {
	count := 0
	for _, rem := range remaining {
		if rem {
			count++
		}
	}
	return count
}

// Helper function: process last remaining variable
func processLastVariable(stdX *mat.Dense, result *GraphResult, remaining []bool, n int) {
	for j := range remaining {
		if remaining[j] {
			result.Order = append(result.Order, j)
			col := make([]float64, n)
			mat.Col(col, j, stdX)
			result.Residuals[len(result.Order)-1] = computeMSE(col)
			remaining[j] = false
			break
		}
	}
}

// Helper function: parallel processing of variables
func (s *SURD) processVariables(stdX *mat.Dense, remaining []bool, n, p int) chan varResult {
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

			// Prepare data for regression
			xSub, y := s.prepareData(stdX, j, remaining)

			// Skip if no predictors available
			if xSub == nil {
				results <- varResult{idx: j, mse: math.MaxFloat64}
				return
			}

			// Run regression
			weights := s.regressor.Fit(xSub, y)

			// Calculate residuals and MSE
			residuals := s.calculateResiduals(xSub, y, weights)
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

	return results
}

// Helper function: find best variable
func findBestVariable(results chan varResult) (int, float64, []float64) {
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
	return bestVar, bestMSE, bestWeights
}

// Helper function: update results with best variable
func (s *SURD) updateResults(result *GraphResult, bestVar int, bestMSE float64,
	bestWeights []float64, remaining []bool, p int) {
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

// Internal struct for parallel results
type varResult struct {
	idx     int
	mse     float64
	weights []float64
}
