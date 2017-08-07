// Main package provide main to test library
package main

import (
	// sys import
	"os"

	// third part import
	log "github.com/sirupsen/logrus"

	// this repo internal import
	mu "github.com/made2591/go-perceptron-go/util"
	mn "github.com/made2591/go-perceptron-go/model/neural"
	//v "github.com/made2591/go-perceptron-go/validation"
	"fmt"
)

func init() {
	// Output to stdout instead of the default stderr
	log.SetOutput(os.Stdout)
	// Only log the warning severity or above.
	log.SetLevel(log.InfoLevel)
}

//############################ MAIN ############################

func main() {

	//// percentage and shuffling in dataset
	//var filePath string = "./res/sonar.all_data.csv"
	//var percentage float64 = 0.67
	//var shuffle = 1
	//
	//// single layer neuron parameters
	//var bias float64 = 0.0
	//var learningRate float64 = 0.01
	//
	//// training parameters
	//var epochs int = 500
	//var folds int = 5
	//
	//// Stimuli initialization
	//stimuli, _ := mn.LoadStimuliFromCSVFile(filePath)
	//
	//// Neuron initialization
	//var neuron mn.Neuron = mn.Neuron{Weights: make([]float64, len(stimuli[0].Dimensions)), Bias: bias, Lrate: learningRate}
	//
	//// compute scores for each folds execution
	//var scores []float64 = v.KFoldValidation(&neuron, stimuli, epochs, folds, shuffle)
	//
	//// use simpler validation
	//var neuron2 mn.Neuron = mn.Neuron{Weights: make([]float64, len(stimuli[0].Dimensions)), Bias: bias, Lrate: learningRate}
	//var scores2 []float64 = v.RandomSubsamplingValidation(&neuron2, stimuli, percentage, epochs, folds, shuffle)
	//
	//log.WithFields(log.Fields{
	//	"level":  "info",
	//	"place":  "main",
	//	"scores": scores,
	//}).Info("Scores reached: ", scores)
	//
	//log.WithFields(log.Fields{
	//	"level":  "info",
	//	"place":  "main",
	//	"scores": scores2,
	//}).Info("Scores reached: ", scores2)


	//SECTION 2 : Build and Train Model
	//Multilayer perceptron model, with one hidden layer.

	// percentage and shuffling in dataset
	var filePath string = "./res/iris.all_data.csv"

	// single layer neuron parameters
	var learningRate float64 = 0.005

	// training parameters
	var epochs int = 500

	// Stimuli initialization
	stimuli, _ , mapped := mn.LoadStimuliFromCSVFile(filePath)

	//input layer : 4 neuron, represents the feature of Iris
	//hidden layer : 3 neuron, activation using sigmoid
	//output layer : 3 neuron, represents the class of Iris
	var layers []int = []int{4, 3, 3}

	//Multilayer perceptron model, with one hidden layer.
	var mlp mn.MultiLayerPerceptron = mn.PrepareMLPNet(layers, learningRate, mn.SigmoidalTransfer, mn.SigmoidalTransferDerivate)

	//fmt.Println(mlp)

	i := 0
	s := stimuli[mu.Random(0, len(stimuli)-1)]
	o := []float64{0, 0, 0}

	for {

		s = stimuli[mu.Random(0, len(stimuli)-1)]
		o[int(s.Expected)] = 1
		mn.BackPropagate(&mlp, &s, o)

		if i > epochs {
			break
		}
		i++

	}

	s = stimuli[mu.Random(0, len(stimuli)-1)]

	o = mn.Execute(&mlp, &s)

	fmt.Printf("Stimulus:      %v\n", s.RawExpected)
	fmt.Printf("Mapped values: %v\n", mapped)
	fmt.Printf("Predicted:     %v\n", o)

}
