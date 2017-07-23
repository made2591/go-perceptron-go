// Neural provides struct to represents most common neural networks model and algorithms to train / test them.
package neural

import (

	// sys import
	"os"
	"math/rand"

	// third part import
	log "github.com/sirupsen/logrus"

	// this repo internal import
	mu "github.com/made2591/go-perceptron-go/util"

)

// Perceptron struct represents a simple Perceptron network with a slice of n weights.
type Perceptron struct {

	// Weights represents Perceptron vector representation
	Weights 		[]float64
	// Bias represents Perceptron natural propensity to spread signal
	Bias    		float64
	// Lrate represents learning rate of perceptron
	Lrate   		float64

}

// #######################################################################################

func init() {
	// Output to stdout instead of the default stderr
	log.SetOutput(os.Stdout)
	// Only log the warning severity or above.
	log.SetLevel(log.InfoLevel)
}

// RandomPerceptronInit initialize perceptron weight, bias and learning rate using NormFloat64 random value.
func RandomPerceptronInit(perceptron *Perceptron) {

	// init random weights
	for index, _ := range perceptron.Weights {
		// init random threshold weight
		perceptron.Weights[index] = rand.NormFloat64()
	}

	// init random bias and lrate
	perceptron.Bias = rand.NormFloat64()
	perceptron.Lrate = rand.NormFloat64() * 0.01

	log.WithFields(log.Fields{
		"level" : "debug",
		"place" : "perceptron",
		"func" : "RandomPerceptronInit",
		"msg" : "random perceptron weights init",
		"weights" : perceptron.Weights,
	}).Debug()

}

// UpdateWeights performs update in perceptron weights with respect to passed stimulus.
// It returns error of prediction before and after updating weights.
func UpdateWeights(perceptron *Perceptron, stimulus *Stimulus) (float64, float64) {

	// compute prediction value and error for stimulus given perceptron BEFORE update (actual state)
	var predictedValue, prevError, postError float64 = Predict(perceptron, stimulus), 0.0, 0.0
	prevError = stimulus.Expected - predictedValue

	// performs weights update for perceptron
	perceptron.Bias = perceptron.Bias + perceptron.Lrate * prevError

	// performs weights update for perceptron
	for index, _ := range perceptron.Weights {
		perceptron.Weights[index] = perceptron.Weights[index] + perceptron.Lrate * prevError * stimulus.Dimensions[index]
	}

	// compute prediction value and error for stimulus given perceptron AFTER update (actual state)
	predictedValue = Predict(perceptron, stimulus)
	postError = stimulus.Expected - predictedValue

	log.WithFields(log.Fields{
		"level" : "debug",
		"place" : "perceptron",
		"func" : "UpdateWeights",
		"msg" : "updating weights of perceptron",
		"weights" : perceptron.Weights,
	}).Debug()

	// return errors
	return prevError, postError

}

// TrainPerceptron trains a passed perceptron with stimuli passed, for specified number of epoch.
// If init is 0, leaves weights unchanged before training.
// If init is 1, reset weights and bias of perceptron before training.
func TrainPerceptron(perceptron *Perceptron, stimuli *Stimuli, epochs int, init int) {

	// init weights if specified
	if init == 1 {
		perceptron.Weights = make([]float64, len(stimuli.Training[0].Dimensions))
		perceptron.Bias = 0.0
	}

	// init counter
	var epoch int = 0

	// accumulator errors prev and post weights updates
	var squaredPrevError, squaredPostError float64 = 0.0, 0.0

	// in each epoch
	for epoch < epochs {

		// update weight using each stimulus in training set
		for _, stimulus := range stimuli.Training {
			prevError, postError := UpdateWeights(perceptron, &stimulus)
			// NOTE: in each step, use weights already updated by previous
			squaredPrevError = squaredPrevError + (prevError * prevError)
			squaredPostError = squaredPostError + (postError * postError)
		}

		log.WithFields(log.Fields{
			"level" : "debug",
			"place" : "error evolution in epoch",
			"method" : "TrainPerceptron",
			"msg" : "epoch and squared errors reached before and after updating weights",
			"epochReached" : epoch+1,
			"squaredErrorPrev" : squaredPrevError,
			"squaredErrorPost" : squaredPostError,
		}).Debug()

		// increment epoch counter
		epoch++

	}

}

// Predict performs a perceptron prediction to passed stimulus.
// It returns a float64 binary predicted value.
func Predict(perceptron *Perceptron, stimulus *Stimulus) float64 {

	if mu.ScalarProduct(perceptron.Weights, stimulus.Dimensions) + perceptron.Bias < 0.0 {
		return 0.0
	}
	return 1.0

}

// Accuracy calculate percentage of equal values between two float64 based slices.
// It returns int number and a float64 percentage value of corrected values.
func Accuracy(actual []float64, predicted []float64) (int, float64) {

	// if slices have different number of elements
	if len(actual) != len(predicted) {
		log.WithFields(log.Fields{
			"level" : "error",
			"place" : "perceptron",
			"method" : "Accuracy",
			"msg" : "accuracy between actual and predicted slices of values",
			"actualLen" : len(actual),
			"predictedLen" : len(predicted),
		}).Error("Failed to compute accuracy between actual values and predictions: different length.")
		return -1, -1.0
	}

	// init result
	var correct int = 0

	for index, value := range actual {
		if value == predicted[index] {
			correct++
		}
	}

	// return correct
	return correct, float64(correct) / float64(len(actual)) * 100.0

}