// Neural provides struct to represents most common neural networks model and algorithms to train / test them.
package neural

import (

	// sys import
	"os"

	// third part import
	log "github.com/sirupsen/logrus"

)

// Level struct represents a simple Neuron network with a slice of n weights.
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

func PrepareLayer(n int, p int) (l Layer) {

	l = Layer{Neurons: make([]Neuron, n), Length: n}

	for i := 0; i < n; i++ {
		RandomNeuronInit(&l.Neurons[i], p)
	}

	return

}