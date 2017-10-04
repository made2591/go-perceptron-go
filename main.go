// Main package provide main to test library
package main

import (
	// sys import
	"os"

	// third part import
	log "github.com/sirupsen/logrus"

	// this repo internal import
	mn "github.com/made2591/go-perceptron-go/model/neural"
	mu "github.com/made2591/go-perceptron-go/util"
	v "github.com/made2591/go-perceptron-go/validation"
)

func init() {
	// Output to stdout instead of the default stderr
	log.SetOutput(os.Stdout)
	// Only log the warning severity or above.
	log.SetLevel(log.InfoLevel)
}

//############################ MAIN ############################

func main() {

	// #############################################################################################################
	// ######################################  Single layer perceptron model  ######################################
	// #############################################################################################################

	if true {

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "main",
			"msg": "single layer perceptron train and test over sonar dataset",
		}).Info("Compute single layer perceptron on sonar data set (binary classification problem)")

		// percentage and shuffling in dataset
		var filePath string = "./res/sonar.all_data.csv"
		var percentage float64 = 0.67
		var shuffle = 1

		// single layer neuron parameters
		var bias float64 = 0.0
		var learningRate float64 = 0.01

		// training parameters
		var epochs int = 500
		var folds int = 5

		// Patterns initialization
		var patterns, _, _ = mn.LoadPatternsFromCSVFile(filePath)

		// NeuronUnit initialization
		var neuron mn.NeuronUnit = mn.NeuronUnit{Weights: make([]float64, len(patterns[0].Features)), Bias: bias, Lrate: learningRate}

		// compute scores for each folds execution
		var scores []float64 = v.KFoldValidation(&neuron, patterns, epochs, folds, shuffle)

		// use simpler validation
		var neuron2 mn.NeuronUnit = mn.NeuronUnit{Weights: make([]float64, len(patterns[0].Features)), Bias: bias, Lrate: learningRate}
		var scores2 []float64 = v.RandomSubsamplingValidation(&neuron2, patterns, percentage, epochs, folds, shuffle)

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "main",
			"scores": scores,
		}).Info("Scores reached: ", scores)

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "main",
			"scores": scores2,
		}).Info("Scores reached: ", scores2)

	}

	// #############################################################################################################
	// ######################################  Multilayer perceptron model  ########################################
	// #############################################################################################################

	if false {

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "main",
			"msg": "multi layer perceptron train and test over iris dataset",
		}).Info("Compute backpropagation multi layer perceptron on sonar data set (binary classification problem)")

		// percentage and shuffling in dataset
		var filePath = "./res/iris.all_data.csv"
		//filePath = "./res/sonar.all_data.csv"

		// single layer neuron parameters
		var learningRate = 0.01
		var percentage = 0.67
		var shuffle = 1

		// training parameters
		var epochs = 500
		var folds  = 3

		// Patterns initialization
		var patterns, _ , mapped = mn.LoadPatternsFromCSVFile(filePath)

		//input  layer : 4 neuron, represents the feature of Iris, more in general dimensions of pattern
		//hidden layer : 3 neuron, activation using sigmoid, number of neuron in hidden level
		// 2° hidden l : * neuron, insert number of level you want
		//output layer : 3 neuron, represents the class of Iris, more in general dimensions of mapped values
		var layers []int = []int{len(patterns[0].Features), 20, len(mapped)}

		//Multilayer perceptron model, with one hidden layer.
		var mlp mn.MultiLayerNetwork = mn.PrepareMLPNet(layers, learningRate, mn.SigmoidalTransfer, mn.SigmoidalTransferDerivate)

		// compute scores for each folds execution
		var scores = v.MLPKFoldValidation(&mlp, patterns, epochs, folds, shuffle, mapped)

		// use simpler validation
		var mlp2 mn.MultiLayerNetwork = mn.PrepareMLPNet(layers, learningRate, mn.SigmoidalTransfer, mn.SigmoidalTransferDerivate)
		var scores2 = v.MLPRandomSubsamplingValidation(&mlp2, patterns, percentage, epochs, folds, shuffle, mapped)

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "main",
			"scores": scores,
		}).Info("Scores reached: ", scores)

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "main",
			"scores": scores2,
		}).Info("Scores reached: ", scores2)

	}

	// #############################################################################################################
	// #########################################  Recurrent Neural Network  ########################################
	// #############################################################################################################

	if true {

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "main",
			"msg": "multi layer perceptron train and test over iris dataset",
		}).Info("Compute training algorithm on elman network using iris data set (binary classification problem)")

		// percentage and shuffling in dataset
		//var filePath = ".\\res\\iris.all_data.csv"
		//var filePath = "./res/sonar.all_data.csv"

		// single layer neuron parameters
		var learningRate = 0.01
		var shuffle = 1

		// training parameters
		var epochs = 500

		// Patterns initialization
		var patterns = mn.CreateRandomPatternArray(8, 30)

		//log.Info(patterns[0].Features[:int(len(patterns[0].Features)/2)])
		//n := mu.ConvertBinToInt(patterns[0].Features[:int(len(patterns[0].Features)/2)])
		//log.Info(n)
		//os.Exit(1)

		//input  layer : 4 neuron, represents the feature of Iris, more in general dimensions of pattern
		//hidden layer : 3 neuron, activation using sigmoid, number of neuron in hidden level
		// 2° hidden l : * neuron, insert number of level you want
		//output layer : 3 neuron, represents the class of Iris, more in general dimensions of mapped values

		//Multilayer perceptron model, with one hidden layer.
		var mlp mn.MultiLayerNetwork =
				mn.PrepareElmanNet(len(patterns[0].Features)+10,
				10, len(patterns[0].MultipleExpectation), learningRate,
				mn.SigmoidalTransfer, mn.SigmoidalTransferDerivate)

		// compute scores for each folds execution
		var mean, _ = v.RNNValidation(&mlp, patterns, epochs, shuffle)

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "main",
			"precision": mu.Round(mean, .5, 2),
		}).Info("Scores reached: ", mu.Round(mean, .5, 2))

	}

}
