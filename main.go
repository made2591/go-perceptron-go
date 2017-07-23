// Main package provide main to test library
package main

import (
	// sys import
	"os"

	// third part import
	log "github.com/sirupsen/logrus"

	// this repo internal import
	mn "github.com/made2591/go-perceptron-go/model/neural"

)

func init() {
	// Output to stdout instead of the default stderr
	log.SetOutput(os.Stdout)
	// Only log the warning severity or above.
	log.SetLevel(log.InfoLevel)
}

// Evaluate perform evaluation on perceptron algorithm.
// It returns scores reached for each fold iteration.
func Evaluate(perceptron *mn.Perceptron, stimuli *mn.Stimuli, percentage float64, epochs int, folds int) []float64 {

	// results and predictions vars init
	var scores, actual, predicted []float64

	for {

		// TODO: fix rotation for cross validation.
		// mn.SeparateSet(stimuli, percentage, 0)

		// train perceptron with set of stimuli, for specified number of epochs
		mn.TrainPerceptron(perceptron, stimuli, epochs, 1)

		// compute predictions for each stimulus in testing set
		for _, stimulus := range stimuli.Testing {
			actual 	  = append(actual, stimulus.Expected)
			predicted = append(predicted, mn.Predict(perceptron, &stimulus))
		}

		// compute score
		_, percentageCorrect := mn.Accuracy(actual, predicted)
		scores = append(scores, percentageCorrect)

		log.WithFields(log.Fields{
			"level": "info",
			"place" : "main",
			"method" : "Evaluate",
			"fold": len(scores),
			"trainSetLen" : len(stimuli.Training),
			"testSetLen" : len(stimuli.Testing),
			"percentageCorrect" : percentageCorrect,
		}).Info("Evaluation completed for actual fold.")

		// returns scores if ended
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

	// single layer perceptron parameters
	var bias float64		 = 0.0
	var learningRate float64 = 0.01

	// training parameters
	var epochs 		int		 = 500
	var folds 		int		 = 3

	// Stimuli initialization
	stimuli, _ := mn.LoadStimuliFromCSVFile(filePath, percentange, shuffling)

	// Perceptron initialization
	var perceptron mn.Perceptron = mn.Perceptron{Weights: make([]float64, len(stimuli.Training[0].Dimensions)), Bias: bias, Lrate: learningRate}

	// compute scores for each folds execution
	var scores []float64 = Evaluate(&perceptron, &stimuli, percentange, epochs, folds)

	log.WithFields(log.Fields{
		"level" : "info",
		"place" : "main",
		"scores" : scores,
	}).Info("Scores reached: ", scores)

}
