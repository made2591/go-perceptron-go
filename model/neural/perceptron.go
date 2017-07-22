package neural

// Perceptron struct represents a simple Perceptron network with a slice of n weights
type Perceptron struct {
	bias    float64
	weights []float64
	lrate   float64
}
