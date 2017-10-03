// Neural provides struct to represents most common neural networks model and algorithms to train / test them.
package neural

import (

	// sys import
	"os"
	//"fmt"
	"math"

	// third part import
	log "github.com/sirupsen/logrus"

	// this repo internal import
	mu "github.com/made2591/go-perceptron-go/util"

)

func init() {
	// Output to stdout instead of the default stderr
	log.SetOutput(os.Stdout)
	// Only log the warning severity or above.
	log.SetLevel(log.InfoLevel)
}

// TODO: RNN simple (Elman)

// PrepareRNNNet create a recurrent neUral network neural network.
// [l:[]int] is an int array with layers neurons number [input, ..., output]
// [lr:int] is the learning rate of neural network
// [tr:transferFunction] is a transfer function
// [tr:transferFunction] the respective transfer function derivative
func PrepareRNNNet(i int, h int, o int, lr float64, tf transferFunction, trd transferFunction) (rnn MultiLayerPerceptron) {

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

// ExecuteElman a multi layer Perceptron neural network.
// [mlp:MultiLayerPerceptron] multilayer perceptron network pointer, [s:Pattern] input value
// It returns output values by network
func ExecuteElman(mlp *MultiLayerPerceptron, s *Pattern) (r []float64) {

	// new value
	nv := 0.0

	// result of execution for each OUTPUT Neuron in OUTPUT Layer
	r = make([]float64, mlp.Layers[len(mlp.Layers)-1].Length)

	// show pattern to network =>
	for i := 0; i < len(s.Dimensions); i++ {

		// setup value of each neurons in first layers to respective features of pattern
		mlp.Layers[0].Neurons[i].Value = s.Dimensions[i]

	}

	// init context 
	for i := len(s.Dimensions); i < mlp.Layers[0].Length; i++ {

		// setup value of each neurons in context layers to 0.5
		mlp.Layers[0].Neurons[i].Value = 0.5

	}

	// executeelman - hiddens + output
	// for each layers from first hidden to output
	for k := 1; k < len(mlp.Layers); k++ {

		// for each neurons in focused level
		for i := 0; i < mlp.Layers[k].Length; i++ {

			// init new value
			nv = 0.0

			// for each neurons in previous level (for k = 1, INPUT)
			for j := 0; j < mlp.Layers[k - 1].Length; j++ {

				// sum output value of previous neurons multiplied by weight between previous and focused neuron
				nv += mlp.Layers[k].Neurons[i].Weights[j] * mlp.Layers[k - 1].Neurons[j].Value

				log.WithFields(log.Fields{
					"level":     "debug",
					"msg":       "elman training execution",
					"len(mlp.Layers)":  len(mlp.Layers),
					"layer:  ": k,
					"neuron: ": i,
					"previous neuron: ": j,
				}).Debug("Compute output propagation.")

			}

			// add neuron bias
			nv += mlp.Layers[k].Neurons[i].Bias

			// compute activation function to new output value
			mlp.Layers[k].Neurons[i].Value = mlp.T_func(nv)

			if k == 1 {

				for z := len(s.Dimensions); z < mlp.Layers[0].Length; z++ {

					// // save output of hidden layer to context
					// log.WithFields(log.Fields{
					// 	"level"				: "info",
					// 	"len z" 			: z,
					// 	"s.Dimensions"		: s.Dimensions,
					// 	"len(s.Dimensions)" : len(s.Dimensions),
					// 	"len mlp.Layers[0].Neurons" : len(mlp.Layers[0].Neurons),
					// 	"len mlp.Layers[k].Neurons" : len(mlp.Layers[k].Neurons),						
					// }).Info("DEBUG");

					mlp.Layers[0].Neurons[z].Value = mlp.Layers[k].Neurons[z-len(s.Dimensions)].Value

				}

			}

			log.WithFields(log.Fields{
				"level":     "debug",
				"msg":       "setup new neuron output value after transfer function application",
				"len(mlp.Layers)":  len(mlp.Layers),
				"layer:  ": k,
				"neuron: ": i,
				"outputvalue" : mlp.Layers[k].Neurons[i].Value,
			}).Debug("Setup new neuron output value after transfer function application.")

		}

	}


	// get ouput values
	for i := 0; i < mlp.Layers[len(mlp.Layers)-1].Length; i++ {

		// simply accumulate values of all neurons in last level
		r[i] = mlp.Layers[len(mlp.Layers)-1].Neurons[i].Value

	}

	return r

}



// BackPropagation algorithm for assisted learning. Convergence is not guaranteed and very slow.
// Use as a stop criterion the average between previous and current errors and a maximum number of iterations.
// [mlp:MultiLayerPerceptron] input value		[s:Pattern] input value (scaled between 0 and 1)
// [o:[]float64] expected output value (scaled between 0 and 1)
// return [r:float64] delta error between generated output and expected output
func BackPropagateElman(mlp *MultiLayerPerceptron, s *Pattern, o []float64) (r float64) {

	// executeelman network with pattern passed over each level to output
	no := ExecuteElman(mlp, s)

	// init error
	e := 0.0

	// compute output error and delta in output layer
	for i := 0; i < mlp.Layers[len(mlp.Layers)-1].Length; i++ {

		// compute error in output: output for given pattern - output computed by network
		e = o[i] - no[i]

		// compute delta for each neuron in output layer as:
		// error in output * derivative of transfer function of network output
		mlp.Layers[len(mlp.Layers)-1].Neurons[i].Delta = e * mlp.T_func_d(no[i])

	}

	// backpropagateelman error to previous layers
	// for each layers starting from the last hidden (len(mlp.Layers)-2)
	for k := len(mlp.Layers)-2; k >= 0; k-- {

		// compute actual layer errors and re-compute delta
		for i := 0; i < mlp.Layers[k].Length; i++ {

			// reset error accumulator
			e = 0.0

			// for each link to next layer
			for j := 0; j < mlp.Layers[k + 1].Length; j++ {

				// sum delta value of next neurons multiplied by weight between focused neuron and all neurons in next level
				e += mlp.Layers[k + 1].Neurons[j].Delta * mlp.Layers[k + 1].Neurons[j].Weights[i]

			}

			// compute delta for each neuron in focused layer as error * derivative of transfer function
			mlp.Layers[k].Neurons[i].Delta = e * mlp.T_func_d(mlp.Layers[k].Neurons[i].Value)

		}

		// compute weights in the next layer
		// for each link to next layer
		for i := 0; i < mlp.Layers[k + 1].Length; i++ {

			// for each neurons in actual level (for k = 0, INPUT)
			for j := 0; j < mlp.Layers[k].Length; j++ {

				// sum learning rate * next level next neuron Delta * actual level actual neuron output value
				mlp.Layers[k + 1].Neurons[i].Weights[j] +=
					mlp.L_rate * mlp.Layers[k + 1].Neurons[i].Delta * mlp.Layers[k].Neurons[j].Value

			}

			// learning rate * next level next neuron Delta * actual level actual neuron output value
			mlp.Layers[k + 1].Neurons[i].Bias += mlp.L_rate * mlp.Layers[k + 1].Neurons[i].Delta

		}

		// copy hidden output to context
		if k == 1 {

			for z := len(s.Dimensions); z < mlp.Layers[0].Length; z++ {

				// save output of hidden layer to context
				mlp.Layers[0].Neurons[z].Value = mlp.Layers[k].Neurons[z-len(s.Dimensions)].Value

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

// RNNTrain train a mlp MultiLayerPerceptron with BackPropagation algorithm for assisted learning.
func RNNTrain(mlp *MultiLayerPerceptron, patterns []Pattern, epochs int) {

	epoch := 0
	output := make([]float64, len(patterns[0].Expected))

	// for fixed number of epochs
	for {

		// for each pattern in training set
		for _, pattern := range patterns {

			// setup desired output for each unit
			for io, _ := range output {
				output[io] = pattern.Expected[io]
			}
			// setup desired output for specific class of pattern focused
			//output[int(pattern.Expected)] = 1.0

			// log.WithFields(log.Fields{
			// 	"level":             "info",
			// 	"place":             "train",
			// 	"method":            "RNNTrain",
			// 	"pattern":        	 pattern.Dimensions,
			// }).Info("DEBUG EXT")

			// back propagation
			BackPropagateElman(mlp, &pattern, pattern.Expected)

			if (epoch % 100 == 0) {

				// get output from network
				o_out := ExecuteElman(mlp, &pattern)
				for o_out_i, o_out_v := range(o_out) {
					o_out[o_out_i] = mu.Round(o_out_v, .5, 0)
				}
				log.WithFields(log.Fields{
					"a_p_b":	pattern.Dimensions,
				}).Info()
				log.WithFields(log.Fields{
					"rea_c":	pattern.Expected,
				}).Info()
				log.WithFields(log.Fields{
					"pre_c":	o_out,
				}).Info()

			}

		}

		log.WithFields(log.Fields{
			"level":             "info",
			"place":             "validation",
			"method":            "RNNTrain",
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