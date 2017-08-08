// Neural provides struct to represents most common neural networks model and algorithms to train / test them.
package neural

import (

	// sys import
	"os"

	// third part import
	log "github.com/sirupsen/logrus"

)

// Level struct represents a simple Neurons network with a slice of n Neurons.
type Layer struct {

	// Neurons represents Neurons in layer
	Neurons []Neuron
	// Lrate represents number of Neuron in layer
	Length int

}

// #######################################################################################

func init() {

	// Output to stdout instead of the default stderr
	log.SetOutput(os.Stdout)
	// Only log the warning severity or above.
	log.SetLevel(log.InfoLevel)

}

// PrepareLayer create a Layer with n Neurons inside
// [n:int] is an int that specifies the number of neurons in the Layer
// [p:int] is an int that specifies the number of neurons in the previous Layer
// It returns a Layer object
func PrepareLayer(n int, p int) (l Layer) {

	l = Layer{Neurons: make([]Neuron, n), Length: n}

	for i := 0; i < n; i++ {
		RandomNeuronInit(&l.Neurons[i], p)
	}

	log.WithFields(log.Fields{
		"level":   "info",
		"msg":     "multilayer perceptron init completed",
		"neurons": len(l.Neurons),
		"lengthPreviousLayer": l.Length,
	}).Info("Complete Layer init.")

	return

}