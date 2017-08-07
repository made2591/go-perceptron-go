// Neural provides struct to represents most common neural networks model and algorithms to train / test them.
package neural

import (

	// sys import
	"os"

	// third part import
	log "github.com/sirupsen/logrus"

	"math"
	//"fmt"
)

func init() {
	// Output to stdout instead of the default stderr
	log.SetOutput(os.Stdout)
	// Only log the warning severity or above.
	log.SetLevel(log.InfoLevel)
}

type MultiLayerPerceptron struct {
	L_rate float64
	Layers []Layer
	T_func transferFunction
	T_func_d transferFunction
}

// Create a multi layer Perceptron neural network.
// [layers:[]int] layers neurons number
// [learningRate:int] layers neurons number
// [layers:[]int] layers neurons number
func PrepareMLPNet(l []int, lr float64, tf transferFunction, trd transferFunction) (mlp MultiLayerPerceptron) {

	mlp.L_rate = lr
	mlp.T_func = tf
	mlp.T_func_d = trd

	mlp.Layers = make([]Layer, len(l))

	for il, ql := range l {

		if il != 0 {

			mlp.Layers[il] = PrepareLayer(ql, l[il-1])

		} else {

			mlp.Layers[il] = PrepareLayer(ql, 0)

		}

	}

	return

}

// Execute a multi layer Perceptron neural network.
// [mlp:MultiLayerPerceptron] input value		[s:Stimulus] input value
// Returns output values by network
func Execute(mlp *MultiLayerPerceptron, s *Stimulus) (r []float64) {

	new_value := 0.0

	r = make([]float64, mlp.Layers[len(mlp.Layers)-1].Length)

	// show stimulus to network
	for i := 0; i < mlp.Layers[0].Length; i++ {

		mlp.Layers[0].Neurons[i].Value = s.Dimensions[i];

	}

	// Execute - hiddens + output
	for k := 1; k < len(mlp.Layers); k++ {

		for i := 0; i < mlp.Layers[k].Length; i++ {

			new_value = 0.0

			for j := 0; j < mlp.Layers[k - 1].Length; j++ {

				new_value += mlp.Layers[k].Neurons[i].Weights[j] * mlp.Layers[k - 1].Neurons[j].Value

			}

			//fmt.Printf("len(mlp.Layers): %d, k: %d, i: %d\n", len(mlp.Layers), k, i)

			new_value += mlp.Layers[k].Neurons[i].Bias

			mlp.Layers[k].Neurons[i].Value = mlp.T_func(new_value)

		}

	}


	// Get output
	for i := 0; i < mlp.Layers[len(mlp.Layers)-1].Length; i++ {

		r[i] = mlp.Layers[len(mlp.Layers)-1].Neurons[i].Value

	}

	return r

}



// BackPropagation algorithm for assisted learning. Convergence is not guaranteed and very slow.
// Use as a stop criterion the average between previous and current errors and a maximum number of iterations.
// [mlp:MultiLayerPerceptron] input value		[s:Stimulus] input value (scaled between 0 and 1)
// [o:[]float64] expected output value (scaled between 0 and 1)
// return [r:float64] delta error between generated output and expected output
func BackPropagate(mlp *MultiLayerPerceptron, s *Stimulus, o []float64) (r float64) {

	no := Execute(mlp, s)
	e := 0.0

	// compute output error
	for i := 0; i < mlp.Layers[len(mlp.Layers)-1].Length; i++ {

		e = o[i] - no[i]

		mlp.Layers[len(mlp.Layers)-1].Neurons[i].Delta = e * mlp.T_func_d(no[i])

	}


	for k := len(mlp.Layers)-2; k >= 0; k-- {

		// compute actual layer errors and re-compute delta
		for i := 0; i < mlp.Layers[k].Length; i++ {

			e = 0.0

			for j := 0; j < mlp.Layers[k + 1].Length; j++ {

				e += mlp.Layers[k + 1].Neurons[j].Delta * mlp.Layers[k + 1].Neurons[j].Weights[i]

			}

			mlp.Layers[k].Neurons[i].Delta = e * mlp.T_func_d(mlp.Layers[k].Neurons[i].Value)

		}

		// compute weights in the next layer
		for i := 0; i < mlp.Layers[k + 1].Length; i++ {

			for j := 0; j < mlp.Layers[k].Length; j++ {

				mlp.Layers[k + 1].Neurons[i].Weights[j] +=
					mlp.L_rate * mlp.Layers[k + 1].Neurons[i].Delta * mlp.Layers[k].Neurons[j].Value

			}

			mlp.Layers[k + 1].Neurons[i].Bias += mlp.L_rate * mlp.Layers[k + 1].Neurons[i].Delta

		}

	}

	// compute errors
	for i := 0; i < len(o); i++ {

		r += math.Abs(no[i] - o[i])

	}

	r = r / float64(len(o))

	return

}