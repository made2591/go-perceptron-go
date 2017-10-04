// Neural provides struct to represents most common neural networks model and algorithms to train / test them.
package neural

import (

	// sys import
	"os"

	// third part import
	log "github.com/sirupsen/logrus"

)

// Level struct represents a simple NeuronUnits network with a slice of n NeuronUnits.
type NeuralLayer struct {

	// NeuronUnits represents NeuronUnits in layer
	NeuronUnits []NeuronUnit
	// Lrate represents number of NeuronUnit in layer
	Length int

}

// #######################################################################################

func init() {

	// Output to stdout instead of the default stderr
	log.SetOutput(os.Stdout)
	// Only log the warning severity or above.
	log.SetLevel(log.InfoLevel)

}

// PrepareLayer create a NeuralLayer with n NeuronUnits inside
// [n:int] is an int that specifies the number of neurons in the NeuralLayer
// [p:int] is an int that specifies the number of neurons in the previous NeuralLayer
// It returns a NeuralLayer object
func PrepareLayer(n int, p int) (l NeuralLayer) {

	l = NeuralLayer{NeuronUnits: make([]NeuronUnit, n), Length: n}

	for i := 0; i < n; i++ {
		RandomNeuronInit(&l.NeuronUnits[i], p)
	}

	log.WithFields(log.Fields{
		"level":   "info",
		"msg":     "multilayer perceptron init completed",
		"neurons": len(l.NeuronUnits),
		"lengthPreviousLayer": l.Length,
	}).Info("Complete NeuralLayer init.")

	return

}

//make([]NeuronUnit, n)