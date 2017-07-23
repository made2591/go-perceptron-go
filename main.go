// Main package provide main to test library
package main

import (
	// sys import
	"fmt"

	// this repo internal import
	mn "github.com/made2591/go-perceptron-go/model/neural"
	// mu "github.com/made2591/go-perceptron-go/util"
)

// Evaluate perform evaluation on perceptron algorithm.
// It returns scores reached for each fold iteration.
func Evaluate(perceptron *mn.Perceptron, stimuli *mn.Stimuli, percentage float64, epochs int, folds int) []float64 {
	var scores []float64
	for {
		// TODO: fix rotation for cross validation.
		mn.SeparateSet(stimuli, percentage, 0)
		fmt.Printf("fold: %d, trains: %d, tests: %d\n", len(scores), len(stimuli.Training), len(stimuli.Testing))
		var predicted []float64
		mn.TrainingPerceptron(perceptron, stimuli, epochs)
		for _, stm := range stimuli.Testing {
			predicted = append(predicted, mn.Predict(perceptron, &stm))
		}
		var actual []float64
		var i int = 0
		for i < len(stimuli.Testing) {
			actual = append(actual, stimuli.Testing[i].Expected)
			i++
		}
		scores = append(scores, mn.Accuracy(actual, predicted))
		if len(scores) == folds {
			return scores
		}
	}
}

//############################ MAIN ############################

func main() {

	// percentage and shuffling in dataset
	var filePath    string   = "./res/sonar.all_data.csv"
	var percentange float64  = 0.67
	var shuffling   int      = 0

	var bias float64		 = 0.0
	var learningRate float64 = 0.01

	var epochs 		int		 = 500
	var folds 		int		 = 3

	// Stimuli initialization
	stimuli, _ := mn.LoadStimuliFromCSVFile(filePath, percentange, shuffling)

	fmt.Println(stimuli)

	// Perceptron initialization
	var perceptron mn.Perceptron = mn.Perceptron{Weights: make([]float64, len(stimuli.Training[0].Dimensions)), Bias: bias, Lrate: learningRate}

	fmt.Printf("\nscores reached: %2.4v\n", Evaluate(&perceptron, &stimuli, percentange, epochs, folds))

}
