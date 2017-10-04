// Neural provides struct to represents most common neural networks model and algorithms to train / test them.
package neural

import (

	// sys import
	"os"
	//"fmt"
	"math"

	// third part import
	log "github.com/sirupsen/logrus"
	mu "github.com/made2591/go-perceptron-go/util"

	"time"
	"math/rand"
)

func init() {
	// Output to stdout instead of the default stderr
	log.SetOutput(os.Stdout)
	// Only log the warning severity or above.
	log.SetLevel(log.InfoLevel)
}

type MultiLayerNetwork struct {

	// Lrate represents learning rate of neuron
	L_rate float64

	// NeuralLayers represents layer of neurons
	NeuralLayers []NeuralLayer

	// Transfer function
	T_func transferFunction

	// Transfer function derivative
	T_func_d transferFunction

}

// PrepareMLPNet create a multi layer Perceptron neural network.
// [l:[]int] is an int array with layers neurons number [input, ..., output]
// [lr:int] is the learning rate of neural network
// [tr:transferFunction] is a transfer function
// [tr:transferFunction] the respective transfer function derivative
func PrepareMLPNet(l []int, lr float64, tf transferFunction, trd transferFunction) (mlp MultiLayerNetwork) {

	// setup learning rate and transfer function
	mlp.L_rate = lr
	mlp.T_func = tf
	mlp.T_func_d = trd

	// setup layers
	mlp.NeuralLayers = make([]NeuralLayer, len(l))

	// for each layers specified
	for il, ql := range l {

		// if it is not the first
		if il != 0 {

			// prepare the GENERIC layer with specific dimension and correct number of links for each NeuronUnits
			mlp.NeuralLayers[il] = PrepareLayer(ql, l[il-1])

		} else {

			// prepare the INPUT layer with specific dimension and No links to previous.
			mlp.NeuralLayers[il] = PrepareLayer(ql, 0)

		}

	}

	log.WithFields(log.Fields{
		"level":     "info",
		"msg":       "multilayer perceptron init completed",
		"layers":  len(mlp.NeuralLayers),
		"learningRate: ": mlp.L_rate,
	}).Info("Complete Multilayer Perceptron init.")

	return

}


// PrepareElmanNet create a recurrent neUral network neural network.
// [l:[]int] is an int array with layers neurons number [input, ..., output]
// [lr:int] is the learning rate of neural network
// [tr:transferFunction] is a transfer function
// [tr:transferFunction] the respective transfer function derivative
func PrepareElmanNet(i int, h int, o int, lr float64, tf transferFunction, trd transferFunction) (rnn MultiLayerNetwork) {

	// setup a three layer network with Input Context dimension
	rnn = PrepareMLPNet([]int{i, h, o}, lr, tf, trd);

	log.WithFields(log.Fields{
		"level":       "info",
		"msg":         "recurrent neural network init completed",
		"inputLayer":   i,
		"hiddenLayer":  h,
		"outputLayer":  o,
		"learningRate: ": rnn.L_rate,
	}).Info("Complete RNN init.")

	return

}

// Execute a multi layer Perceptron neural network.
// [mlp:MultiLayerNetwork] multilayer perceptron network pointer, [s:Pattern] input value
// It returns output values by network
func Execute(mlp *MultiLayerNetwork, s *Pattern, options ...int) (r []float64) {

	// new value
	nv := 0.0

	// result of execution for each OUTPUT NeuronUnit in OUTPUT NeuralLayer
	r = make([]float64, mlp.NeuralLayers[len(mlp.NeuralLayers)-1].Length)

	// show pattern to network =>
	for i := 0; i < len(s.Features); i++ {

		// setup value of each neurons in first layers to respective features of pattern
		mlp.NeuralLayers[0].NeuronUnits[i].Value = s.Features[i]

	}

	// init context
	for i := len(s.Features); i < mlp.NeuralLayers[0].Length; i++ {

		// setup value of each neurons in context layers to 0.5
		mlp.NeuralLayers[0].NeuronUnits[i].Value = 0.5

	}

	//OLD: TODO REMOVE
	// show pattern to network =>
	//for i := 0; i < mlp.NeuralLayers[0].Length; i++ {
	//
	//	// setup value of each neurons in first layers to respective features of pattern
	//	mlp.NeuralLayers[0].NeuronUnits[i].Value = s.Features[i]
	//
	//}
	//OLD: END REMOVE

	// execute - hiddens + output
	// for each layers from first hidden to output
	for k := 1; k < len(mlp.NeuralLayers); k++ {

		// for each neurons in focused level
		for i := 0; i < mlp.NeuralLayers[k].Length; i++ {

			// init new value
			nv = 0.0

			// for each neurons in previous level (for k = 1, INPUT)
			for j := 0; j < mlp.NeuralLayers[k - 1].Length; j++ {

				// sum output value of previous neurons multiplied by weight between previous and focused neuron
				nv += mlp.NeuralLayers[k].NeuronUnits[i].Weights[j] * mlp.NeuralLayers[k - 1].NeuronUnits[j].Value

				log.WithFields(log.Fields{
					"level":     "debug",
					"msg":       "multilayer perceptron execution",
					"len(mlp.NeuralLayers)":  len(mlp.NeuralLayers),
					"layer:  ": k,
					"neuron: ": i,
					"previous neuron: ": j,
				}).Debug("Compute output propagation.")

			}

			// add neuron bias
			nv += mlp.NeuralLayers[k].NeuronUnits[i].Bias

			// compute activation function to new output value
			mlp.NeuralLayers[k].NeuronUnits[i].Value = mlp.T_func(nv)

			// save output of hidden layer to context if nextwork is RECURRENT
			if k == 1 && len(options) > 0 && options[0] == 1 {

				for z := len(s.Features); z < mlp.NeuralLayers[0].Length; z++ {

					log.WithFields(log.Fields{
						"level"				: "debug",
						"len z" 			: z,
						"s.Features"		: s.Features,
						"len(s.Features)" : len(s.Features),
						"len mlp.NeuralLayers[0].NeuronUnits" : len(mlp.NeuralLayers[0].NeuronUnits),
						"len mlp.NeuralLayers[k].NeuronUnits" : len(mlp.NeuralLayers[k].NeuronUnits),
					}).Debug("Save output of hidden layer to context.")

					mlp.NeuralLayers[0].NeuronUnits[z].Value = mlp.NeuralLayers[k].NeuronUnits[z-len(s.Features)].Value

				}

			}

			log.WithFields(log.Fields{
				"level":     "debug",
				"msg":       "setup new neuron output value after transfer function application",
				"len(mlp.NeuralLayers)":  len(mlp.NeuralLayers),
				"layer:  ": k,
				"neuron: ": i,
				"outputvalue" : mlp.NeuralLayers[k].NeuronUnits[i].Value,
			}).Debug("Setup new neuron output value after transfer function application.")

		}

	}


	// get ouput values
	for i := 0; i < mlp.NeuralLayers[len(mlp.NeuralLayers)-1].Length; i++ {

		// simply accumulate values of all neurons in last level
		r[i] = mlp.NeuralLayers[len(mlp.NeuralLayers)-1].NeuronUnits[i].Value

	}

	return r

}



// BackPropagation algorithm for assisted learning. Convergence is not guaranteed and very slow.
// Use as a stop criterion the average between previous and current errors and a maximum number of iterations.
// [mlp:MultiLayerNetwork] input value		[s:Pattern] input value (scaled between 0 and 1)
// [o:[]float64] expected output value (scaled between 0 and 1)
// return [r:float64] delta error between generated output and expected output
func BackPropagate(mlp *MultiLayerNetwork, s *Pattern, o []float64, options ...int) (r float64) {

	var no []float64;
	// execute network with pattern passed over each level to output
	if len(options) == 1 {
		no = Execute(mlp, s, options[0])
	} else {
		no = Execute(mlp, s)
	}

	// init error
	e := 0.0

	// compute output error and delta in output layer
	for i := 0; i < mlp.NeuralLayers[len(mlp.NeuralLayers)-1].Length; i++ {

		// compute error in output: output for given pattern - output computed by network
		e = o[i] - no[i]

		// compute delta for each neuron in output layer as:
		// error in output * derivative of transfer function of network output
		mlp.NeuralLayers[len(mlp.NeuralLayers)-1].NeuronUnits[i].Delta = e * mlp.T_func_d(no[i])

	}

	// backpropagate error to previous layers
	// for each layers starting from the last hidden (len(mlp.NeuralLayers)-2)
	for k := len(mlp.NeuralLayers)-2; k >= 0; k-- {

		// compute actual layer errors and re-compute delta
		for i := 0; i < mlp.NeuralLayers[k].Length; i++ {

			// reset error accumulator
			e = 0.0

			// for each link to next layer
			for j := 0; j < mlp.NeuralLayers[k + 1].Length; j++ {

				// sum delta value of next neurons multiplied by weight between focused neuron and all neurons in next level
				e += mlp.NeuralLayers[k + 1].NeuronUnits[j].Delta * mlp.NeuralLayers[k + 1].NeuronUnits[j].Weights[i]

			}

			// compute delta for each neuron in focused layer as error * derivative of transfer function
			mlp.NeuralLayers[k].NeuronUnits[i].Delta = e * mlp.T_func_d(mlp.NeuralLayers[k].NeuronUnits[i].Value)

		}

		// compute weights in the next layer
		// for each link to next layer
		for i := 0; i < mlp.NeuralLayers[k + 1].Length; i++ {

			// for each neurons in actual level (for k = 0, INPUT)
			for j := 0; j < mlp.NeuralLayers[k].Length; j++ {

				// sum learning rate * next level next neuron Delta * actual level actual neuron output value
				mlp.NeuralLayers[k + 1].NeuronUnits[i].Weights[j] +=
					mlp.L_rate * mlp.NeuralLayers[k + 1].NeuronUnits[i].Delta * mlp.NeuralLayers[k].NeuronUnits[j].Value

			}

			// learning rate * next level next neuron Delta * actual level actual neuron output value
			mlp.NeuralLayers[k + 1].NeuronUnits[i].Bias += mlp.L_rate * mlp.NeuralLayers[k + 1].NeuronUnits[i].Delta

		}

		// copy hidden output to context
		if k == 1 && len(options) > 0 && options[0] == 1 {

			for z := len(s.Features); z < mlp.NeuralLayers[0].Length; z++ {

				// save output of hidden layer to context
				mlp.NeuralLayers[0].NeuronUnits[z].Value = mlp.NeuralLayers[k].NeuronUnits[z-len(s.Features)].Value

			}

		}

	}

	// compute global errors as sum of abs difference between output execution for each neuron in output layer
	// and desired value in each neuron in output layer
	for i := 0; i < len(o); i++ {

		r += math.Abs(no[i] - o[i])

	}

	// average error
	r = r / float64(len(o))

	return

}

// MLPTrain train a mlp MultiLayerNetwork with BackPropagation algorithm for assisted learning.
func MLPTrain(mlp *MultiLayerNetwork, patterns []Pattern, mapped []string, epochs int) {

	epoch := 0
	output := make([]float64, len(mapped))

	// for fixed number of epochs
	for {

		// for each pattern in training set
		for _, pattern := range patterns {

			// setup desired output for each unit
			for io, _ := range output {
				output[io] = 0.0
			}
			// setup desired output for specific class of pattern focused
			output[int(pattern.SingleExpectation)] = 1.0
			// back propagation
			BackPropagate(mlp, &pattern, output)

		}

		log.WithFields(log.Fields{
			"level":             "info",
			"place":             "validation",
			"method":            "MLPTrain",
			"epoch":        	 epoch,
		}).Debug("Training epoch completed.")

		// if max number of epochs is reached
		if epoch > epochs {
			// exit
			break
		}
		// increase number of epoch
		epoch++

	}

}

// ElmanTrain train a mlp MultiLayerNetwork with BackPropagation algorithm for assisted learning.
func ElmanTrain(mlp *MultiLayerNetwork, patterns []Pattern, epochs int) {

	epoch := 0

	// for fixed number of epochs
	for {

		rand.Seed(time.Now().UTC().UnixNano())
		p_i_r := rand.Intn(len(patterns))

		// for each pattern in training set
		for p_i, pattern := range patterns {

			// back propagation
			BackPropagate(mlp, &pattern, pattern.MultipleExpectation, 1)

			if (epoch % 100 == 0 && p_i == p_i_r) {

				// get output from network
				o_out := Execute(mlp, &pattern, 1)
				for o_out_i, o_out_v := range(o_out) {
					o_out[o_out_i] = mu.Round(o_out_v, .5, 0)
				}
				log.WithFields(log.Fields{
					"SUM":	"  ==========================",
				}).Info()
				log.WithFields(log.Fields{
					"a_n_1":	mu.ConvertBinToInt(pattern.Features[0:int(len(pattern.Features)/2)]),
					"a_n_2":	pattern.Features[0:int(len(pattern.Features)/2)],
				}).Info()
				log.WithFields(log.Fields{
					"b_n_1":	mu.ConvertBinToInt(pattern.Features[int(len(pattern.Features)/2):]),
					"b_n_2":	pattern.Features[int(len(pattern.Features)/2):],
				}).Info()
				log.WithFields(log.Fields{
					"sum_1":	mu.ConvertBinToInt(pattern.MultipleExpectation),
					"sum_2":	pattern.MultipleExpectation,
				}).Info()
				log.WithFields(log.Fields{
					"sum_1":	mu.ConvertBinToInt(o_out),
					"sum_2":	o_out,
				}).Info()
				log.WithFields(log.Fields{
					"END":	"  ==========================",
				}).Info()

			}

		}

		log.WithFields(log.Fields{
			"level":             "info",
			"place":             "validation",
			"method":            "ElmanTrain",
			"epoch":        	 epoch,
		}).Debug("Training epoch completed.")

		// if max number of epochs is reached
		if epoch > epochs {
			// exit
			break
		}
		// increase number of epoch
		epoch++

	}

}