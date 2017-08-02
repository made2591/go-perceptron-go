// Main package provide main to test library
package main

import (
	// sys import
	"os"

	// third part import
	log "github.com/sirupsen/logrus"

	// this repo internal import
	mn "github.com/made2591/go-perceptron-go/model/neural"
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

	// percentage and shuffling in dataset
	var filePath string = "./res/sonar.all_data.csv"
	var percentage float64 = 0.67
	var shuffle = 1

	// single layer perceptron parameters
	var bias float64 = 0.0
	var learningRate float64 = 0.01

	// training parameters
	var epochs int = 500
	var folds int = 5

	// Stimuli initialization
	stimuli, _ := mn.LoadStimuliFromCSVFile(filePath)

	// Perceptron initialization
	var perceptron mn.Perceptron = mn.Perceptron{Weights: make([]float64, len(stimuli[0].Dimensions)), Bias: bias, Lrate: learningRate}

	// compute scores for each folds execution
	var scores []float64 = v.KFoldValidation(&perceptron, stimuli, epochs, folds, shuffle)

	// use simpler validation
	var perceptron2 mn.Perceptron = mn.Perceptron{Weights: make([]float64, len(stimuli[0].Dimensions)), Bias: bias, Lrate: learningRate}
	var scores2 []float64 = v.RandomSubsamplingValidation(&perceptron2, stimuli, percentage, epochs, folds, shuffle)

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
