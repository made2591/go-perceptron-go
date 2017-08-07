// Neural provides struct to represents most common neural networks model and algorithms to train / test them.
package neural

import (
	// sys import
	"encoding/csv"
	"io"
	"io/ioutil"
	"os"
	"strings"

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
func LoadStimuliFromCSVFile(filePath string) ([]Stimulus, error, []string) {

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
		return stimuli, error, nil
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
			return stimuli, error, nil
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
	mapped := RawExpectedConversion(stimuli)

	// return stimuli
	return stimuli, nil, mapped

}

// RawExpectedConversion converts (string) raw expected values in stimuli
// training / testing sets to float64 values
// It works on stimule struct (pointer) passed. It doens't returns nothing
func RawExpectedConversion(stimuli []Stimulus) []string {

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

	return rawExpectedValues

}
