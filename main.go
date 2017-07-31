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
func Evaluate(perceptron *mn.Perceptron, stimuli []mn.Stimulus, percentage float64, epochs int, folds int) []float64 {

	// results and predictions vars init
	var scores, actual, predicted []float64
	var train, test []mn.Stimulus

	scores = make([]float64, folds)

	for t := 0; t < folds; t++ {
		// split the dataset with shuffling
		train, test = mn.SeparateSet(stimuli, percentage, 1)

		// train perceptron with set of stimuli, for specified number of epochs
		mn.TrainPerceptron(perceptron, train, epochs, 1)

		// compute predictions for each stimulus in testing set
		for _, stimulus := range test {
			actual = append(actual, stimulus.Expected)
			predicted = append(predicted, mn.Predict(perceptron, &stimulus))
		}

		// compute score
		_, percentageCorrect := mn.Accuracy(actual, predicted)
		scores[t] = percentageCorrect

		log.WithFields(log.Fields{
			"level":             "info",
			"place":             "main",
			"method":            "Evaluate",
			"foldNumber":        t,
			"trainSetLen":       len(train),
			"testSetLen":        len(test),
			"percentageCorrect": percentageCorrect,
		}).Info("Evaluation completed for current fold.")
	}

	// compute average score
	acc := 0.0
	for i := 0; i < len(scores); i++ {
		acc += scores[i]
	}

	mean := acc / float64(len(scores))

	log.WithFields(log.Fields{
		"level":       "info",
		"place":       "main",
		"method":      "Evaluate",
		"folds":       folds,
		"trainSetLen": len(train),
		"testSetLen":  len(test),
		"meanScore":   mean,
	}).Info("Evaluation completed for all folds.")

	return scores

}

//############################ MAIN ############################

func main() {

	// percentage and shuffling in dataset
	var filePath string = "./res/sonar.all_data.csv"
	var percentange float64 = 0.67

	// single layer perceptron parameters
	var bias float64 = 0.0
	var learningRate float64 = 0.01

	// training parameters
	var epochs int = 500
	var folds int = 3

	// Stimuli initialization
	stimuli, _ := mn.LoadStimuliFromCSVFile(filePath)

	// Perceptron initialization
	var perceptron mn.Perceptron = mn.Perceptron{Weights: make([]float64, len(stimuli[0].Dimensions)), Bias: bias, Lrate: learningRate}

	// compute scores for each folds execution
	var scores []float64 = Evaluate(&perceptron, stimuli, percentange, epochs, folds)

	log.WithFields(log.Fields{
		"level":  "info",
		"place":  "main",
		"scores": scores,
	}).Info("Scores reached: ", scores)

}
