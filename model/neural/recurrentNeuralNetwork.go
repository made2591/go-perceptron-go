// Neural provides struct to represents most common neural networks model and algorithms to train / test them.
package neural

import (

	// sys import
	"os"
	//"fmt"
	"math"

	// third part import
	log "github.com/sirupsen/logrus"
	//mu "github.com/made2591/go-perceptron-go/util"

)

func init() {
	// Output to stdout instead of the default stderr
	log.SetOutput(os.Stdout)
	// Only log the warning severity or above.
	log.SetLevel(log.InfoLevel)
}

type RecurrentNeuralNetwork struct {

	// Lrate represents learning rate of neuron
	L_rate float64

	// Layers represents layer of neurons
	Layers []Layer

	// Transfer function
	T_func transferFunction

	// Transfer function derivative
	T_func_d transferFunction

}

// TODO: RNN simple (Elman)