// Neural provides struct to represents most common neural networks model and algorithms to train / test them.
package neural

import (
	// sys import
	"os"
	"io"
	"strings"
	"io/ioutil"
	"math/rand"
	"encoding/csv"

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

// Stimuli struct represents a set of stimulus splitted in training and testing set
type Stimuli struct {

	// Set of stimulus for training
	Training 		[]Stimulus
	// Set of stimulus for testing
	Testing  		[]Stimulus
	// Percentage of strimulus for training and testing
	SplitPercentage	float64

}

// Stimulus struct represents one stimulus with dimensions and desired value
type Stimulus struct {

	// Features that describe the stimulus
	Dimensions  	[]float64
	// Raw (usually string) expected value
	RawExpected 	string
	// Numeric representation of expected value
	Expected    	float64

}

// #######################################################################################

// LoadStimuliFromCSVFile load a CSV dataset into Stimuli struct.
// If mix is 0, no shuffling will be performed.
// If 1, pseudo-random shuffling of line will be performed.
// It returns Stimuli struct splitting record in file with passed percentage.
func LoadStimuliFromCSVFile(filePath string, splitPercentage float64, mix int) (Stimuli, error) {

	// init stimuli set
	var stimuli Stimuli = Stimuli{Training: []Stimulus{}, Testing: []Stimulus{}, SplitPercentage: splitPercentage}

	// read content ([]byte), check error (error)
	fileContent, error := ioutil.ReadFile(filePath)

	if error != nil {
		log.WithFields(log.Fields{
			"level" : "fatal",
			"place" : "stimuli",
			"method" : "LoadStimuliFromCSVFile",
			"msg" : "reading file in specific path",
			"filePath" : filePath,
			"error" : error,
		}).Fatal("Failed to read file in specified path.")
		return stimuli, error
	}

	// create pointer to read file
	pointer := csv.NewReader(strings.NewReader(string(fileContent)))

	var lineCounter int = 0
	// for each record in file
	for {

		// read line, check error
		line, error := pointer.Read()

		log.WithFields(log.Fields{
			"level" : "debug",
			"place" : "stimuli",
			"method" : "LoadStimuliFromCSVFile",
			"line" : line,
		}).Debug()

		// if end of file reached, exit loop
		if error == io.EOF {
			log.WithFields(log.Fields{
				"level" : "info",
				"place" : "stimuli",
				"method" : "LoadStimuliFromCSVFile",
				"msg" : "reached end of file during reading",
			}).Info("Reached end of file during reading.")
			break
		}

		// if another error encountered, exit program
		if error != nil {
			log.WithFields(log.Fields{
				"level" : "error",
				"place" : "stimuli",
				"method" : "LoadStimuliFromCSVFile",
				"msg": "parsing file in specific line number",
				"lineCounter" : lineCounter,
				"error" : error,
			}).Error("Failed to parse line.")
			return stimuli, error
		}

		// line values cast to float64
		var floatingValues []float64 = mu.StringToFloat(line, 1, -1.0)

		// add casted stimulus to training set
		stimuli.Training = append(
			stimuli.Training,
			Stimulus{Dimensions: floatingValues, RawExpected: line[len(line)-1]})

		lineCounter = lineCounter + 1

	}

	// cast expected values to float64 numeric values
	RawExpectedConversion(&stimuli)

	// create training and testing set with percentange and shuffling mode passed.
	SeparateSet(&stimuli, splitPercentage, mix)

	// return stimuli
	return stimuli, nil

}

// RawExpectedConversion converts (string) raw expected values in stimuli training / testing sets to float64 values
// It works on stimule struct (pointer) passed. It doens't returns nothing
func RawExpectedConversion(stimuli *Stimuli) {

	// collect expected string values
	var rawExpectedValues []string

	// for each stimulus in training set
	for _, stimulus := range stimuli.Training {
		check, _ := mu.StringInSlice(stimulus.RawExpected, rawExpectedValues)
		if !check {
			rawExpectedValues = append(rawExpectedValues, stimulus.RawExpected)
		}
		log.WithFields(log.Fields{
			"level" : "debug",
			"place" : "stimuli",
			"msg" : "raw class exctraction",
			"rawExpectedAdded" : stimulus.RawExpected,
		}).Debug()
	}

	log.WithFields(log.Fields{
		"level" : "info",
		"place" : "stimuli",
		"msg" : "raw class exctraction completed",
		"numberOfRawUnique" : len(rawExpectedValues),
	}).Info("Complete RawExpected value set filling.")

	// for each stimulus in training set
	for index, _ := range stimuli.Training {
		// find mapped int value (using index of rawExpectedValues slice)
		for mapped, value := range rawExpectedValues {
			if strings.Compare(value, stimuli.Training[index].RawExpected) == 0 {
				// conversion to float64 value
				stimuli.Training[index].Expected = float64(mapped)
			}
		}
	}

}

// separateSet Stimuli training set in training and testing.
// TODO: create in-place rotation
// If passed mix is 0, training and testing set will be first merged, then splitted with given percentage.
// If passed mix is 1, training and testing set will be first merged, then shuffled, then splitted with given percentage.
func SeparateSet(stimuli *Stimuli, percentage float64, mix int) {

	// create a ordered copy of Stimulus in training and testing set
	var availableStimuli []Stimulus = make([]Stimulus, len(stimuli.Training)+len(stimuli.Testing))

	// create splitting pivot
	var i, k, splitPivot int = 0, 0, int(float64(len(availableStimuli)) * percentage)

	// if mixed mode, shuffled copy inside availableStimuli
	if mix == 1 {
		// create perm
		perm := rand.Perm(len(availableStimuli))

		// copy each element in training set in random position
		for i < len(stimuli.Training) {
			availableStimuli[perm[i]] = stimuli.Training[i]
			i++
		}
		// copy each element in testing set in random position
		for k < len(stimuli.Testing) {
			availableStimuli[perm[i+k]] = stimuli.Testing[i+k]
			k++
		}
		// else, copy inside availableStimuli without perm
	} else {
		for i < len(stimuli.Training) {
			availableStimuli[i] = stimuli.Training[i]
			i++
		}
		for k < len(stimuli.Testing) {
			availableStimuli[i+k] = stimuli.Testing[k]
			k++
		}
	}

	// init indexes
	i, k = 0, 0

	// empty struct
	stimuli.Training = make([]Stimulus, splitPivot)
	stimuli.Testing  = make([]Stimulus, len(availableStimuli)-splitPivot)

	// fill struct with specified perc
	for i < splitPivot {
		stimuli.Training[i] = availableStimuli[i]
		i++
	}
	for k < len(availableStimuli)-splitPivot {
		stimuli.Testing[k] = availableStimuli[i+k]
		k++
	}

	log.WithFields(log.Fields{
		"level" : "info",
		"msg" : "splitting completed",
		"trainSet" : len(stimuli.Training),
		"testSet: " : len(stimuli.Testing),
	}).Info("Complete splitting train/test set.")

}