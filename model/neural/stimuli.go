// Neural provides struct to represents most common neural networks model and algorithms to train / test them.
package neural

import (
	// sys import
	"encoding/csv"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"strings"
	"time"

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

// Stimulus struct represents one stimulus with dimensions and desired value
type Stimulus struct {

	// Features that describe the stimulus
	Dimensions []float64
	// Raw (usually string) expected value
	RawExpected string
	// Numeric representation of expected value
	Expected float64
}

// #######################################################################################

// LoadStimuliFromCSVFile load a CSV dataset into an array of Stimulus.
func LoadStimuliFromCSVFile(filePath string) ([]Stimulus, error) {

	// init stimuli
	var stimuli []Stimulus

	// read content ([]byte), check error (error)
	fileContent, error := ioutil.ReadFile(filePath)

	if error != nil {
		log.WithFields(log.Fields{
			"level":    "fatal",
			"place":    "stimuli",
			"method":   "LoadStimuliFromCSVFile",
			"msg":      "reading file in specific path",
			"filePath": filePath,
			"error":    error,
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
			"level":  "debug",
			"place":  "stimuli",
			"method": "LoadStimuliFromCSVFile",
			"line":   line,
		}).Debug()

		// if end of file reached, exit loop
		if error == io.EOF {
			log.WithFields(log.Fields{
				"level":    "info",
				"place":    "stimuli",
				"method":   "LoadStimuliFromCSVFile",
				"readData": len(stimuli),
				"msg":      "File reading completed.",
			}).Info("File reading completed.")
			break
		}

		// if another error encountered, exit program
		if error != nil {
			log.WithFields(log.Fields{
				"level":       "error",
				"place":       "stimuli",
				"method":      "LoadStimuliFromCSVFile",
				"msg":         "parsing file in specific line number",
				"lineCounter": lineCounter,
				"error":       error,
			}).Error("Failed to parse line.")
			return stimuli, error
		}

		// line values cast to float64
		var floatingValues []float64 = mu.StringToFloat(line, 1, -1.0)

		// add casted stimulus to training set
		stimuli = append(
			stimuli,
			Stimulus{Dimensions: floatingValues, RawExpected: line[len(line)-1]})

		lineCounter = lineCounter + 1

	}

	// cast expected values to float64 numeric values
	RawExpectedConversion(stimuli)

	// return stimuli
	return stimuli, nil

}

// RawExpectedConversion converts (string) raw expected values in stimuli
// training / testing sets to float64 values
// It works on stimule struct (pointer) passed. It doens't returns nothing
func RawExpectedConversion(stimuli []Stimulus) {

	// collect expected string values
	var rawExpectedValues []string

	// for each stimulus in training set
	for _, stimulus := range stimuli {
		check, _ := mu.StringInSlice(stimulus.RawExpected, rawExpectedValues)
		if !check {
			rawExpectedValues = append(rawExpectedValues, stimulus.RawExpected)
		}
		log.WithFields(log.Fields{
			"level":            "debug",
			"place":            "stimuli",
			"msg":              "raw class exctraction",
			"rawExpectedAdded": stimulus.RawExpected,
		}).Debug()
	}

	log.WithFields(log.Fields{
		"level":             "info",
		"place":             "stimuli",
		"msg":               "raw class exctraction completed",
		"numberOfRawUnique": len(rawExpectedValues),
	}).Info("Complete RawExpected value set filling.")

	// for each stimulus in training set
	for index, _ := range stimuli {
		// find mapped int value (using index of rawExpectedValues slice)
		for mapped, value := range rawExpectedValues {
			if strings.Compare(value, stimuli[index].RawExpected) == 0 {
				// conversion to float64 value
				stimuli[index].Expected = float64(mapped)
			}
		}
	}

}

// SeparateSet split an array of stimuli in training and testing.
// if shuffle is 0 the function takes the first percentage items as train and the other as test
// otherwise the stimuli array is shuffled before partitioning
func SeparateSet(stimuli []Stimulus, percentage float64, shuffle int) (train []Stimulus, test []Stimulus) {

	// create splitting pivot
	var splitPivot int = int(float64(len(stimuli)) * percentage)
	train = make([]Stimulus, splitPivot)
	test = make([]Stimulus, len(stimuli)-splitPivot)

	// if mixed mode, split with shuffling
	if shuffle == 1 {
		// create random indexes permutation
		rand.Seed(time.Now().UTC().UnixNano())
		perm := rand.Perm(len(stimuli))

		// copy training data
		for i := 0; i < splitPivot; i++ {
			train[i] = stimuli[perm[i]]
		}
		// copy test data
		for i := 0; i < len(stimuli)-splitPivot; i++ {
			test[i] = stimuli[perm[i]]
		}

	} else {
		// else, split without shuffle
		train = stimuli[:splitPivot]
		test = stimuli[splitPivot:]
	}

	log.WithFields(log.Fields{
		"level":     "info",
		"msg":       "splitting completed",
		"trainSet":  len(train),
		"testSet: ": len(test),
	}).Info("Complete splitting train/test set.")

	return train, test
}
