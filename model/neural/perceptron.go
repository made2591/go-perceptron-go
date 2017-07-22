package neural

// Perceptron struct represents a simple mn.mn.Perceptron network with a slice of n weights
type Perceptron struct {
	Bias    float64
	Weights []float64
	Lrate   float64
}
