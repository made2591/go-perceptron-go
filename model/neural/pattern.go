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

// Pattern struct represents one pattern with dimensions and desired value
type Pattern struct {

	// Features that describe the pattern
	Features []float64
	// Raw (usually string) expected value
	SingleRawExpectation string
	// Numeric representation of expected value
	SingleExpectation float64
	// Numeric representation of expected value
	MultipleExpectation []float64

}

// #######################################################################################

// LoadPatternsFromCSVFile load a CSV dataset into an array of Pattern.
func LoadPatternsFromCSVFile(filePath string) ([]Pattern, error, []string) {

	// init patterns
	var patterns []Pattern

	// read content ([]byte), check error (error)
	fileContent, error := ioutil.ReadFile(filePath)

	if error != nil {
		log.WithFields(log.Fields{
			"level":    "fatal",
			"place":    "patterns",
			"method":   "LoadPatternsFromCSVFile",
			"msg":      "reading file in specific path",
			"filePath": filePath,
			"error":    error,
		}).Fatal("Failed to read file in specified path.")
		return patterns, error, nil
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
			"place":  "patterns",
			"method": "LoadPatternsFromCSVFile",
			"line":   line,
		}).Debug()

		// if end of file reached, exit loop
		if error == io.EOF {
			log.WithFields(log.Fields{
				"level":    "info",
				"place":    "patterns",
				"method":   "LoadPatternsFromCSVFile",
				"readData": len(patterns),
				"msg":      "File reading completed.",
			}).Info("File reading completed.")
			break
		}

		// if another error encountered, exit program
		if error != nil {
			log.WithFields(log.Fields{
				"level":       "error",
				"place":       "patterns",
				"method":      "LoadPatternsFromCSVFile",
				"msg":         "parsing file in specific line number",
				"lineCounter": lineCounter,
				"error":       error,
			}).Error("Failed to parse line.")
			return patterns, error, nil
		}

		// line values cast to float64
		var floatingValues []float64 = mu.StringToFloat(line, 1, -1.0)

		// add casted pattern to training set
		patterns = append(
			patterns,
			Pattern{Features: floatingValues, SingleRawExpectation: line[len(line)-1]})

		lineCounter = lineCounter + 1

	}

	// cast expected values to float64 numeric values
	mapped := RawExpectedConversion(patterns)

	// return patterns
	return patterns, nil, mapped

}

// RawExpectedConversion converts (string) raw expected values in patterns
// training / testing sets to float64 values
// It works on pattern struct (pointer) passed. It doens't returns nothing
func RawExpectedConversion(patterns []Pattern) []string {

	// collect expected string values
	var rawExpectedValues []string

	// for each pattern in training set
	for _, pattern := range patterns {
		check, _ := mu.StringInSlice(pattern.SingleRawExpectation, rawExpectedValues)
		if !check {
			rawExpectedValues = append(rawExpectedValues, pattern.SingleRawExpectation)
		}
		log.WithFields(log.Fields{
			"level":            "debug",
			"place":            "patterns",
			"msg":              "raw class exctraction",
			"rawExpectedAdded": pattern.SingleRawExpectation,
		}).Debug()
	}

	log.WithFields(log.Fields{
		"level":             "info",
		"place":             "patterns",
		"msg":               "raw class exctraction completed",
		"numberOfRawUnique": len(rawExpectedValues),
	}).Info("Complete SingleRawExpectation value set filling.")

	// for each pattern in training set
	for index, _ := range patterns {
		// find mapped int value (using index of rawExpectedValues slice)
		for mapped, value := range rawExpectedValues {
			if strings.Compare(value, patterns[index].SingleRawExpectation) == 0 {
				// conversion to float64 value
				patterns[index].SingleExpectation = float64(mapped)
			}
		}
	}

	return rawExpectedValues

}

// CreateRandomPatternArray load a CSV dataset into an array of Pattern.
func CreateRandomPatternArray(d int, k int) ([]Pattern) {

	// init patterns
	var patterns []Pattern;

	// for i times
	var i = 0
	for i < k {

		a := mu.GenerateRandomIntWithBinaryDim(d)
		b := mu.GenerateRandomIntWithBinaryDim(d)
		c := a+b

		log.WithFields(log.Fields{
			"ai":	a,
			"as":	mu.ConvertIntToBinary(a, d),
			"bi":	b,
			"bs":	mu.ConvertIntToBinary(b, d),
			"ci":	c,
			"cs":	mu.ConvertIntToBinary(c, d+1),
		}).Debug()

		ab := mu.ConvertIntToBinary(a, d)
		bb := mu.ConvertIntToBinary(b, d)
		for _, v := range(bb) {
			ab = append(ab, v)
		}

		// add casted pattern to training set
		patterns = append(
			patterns,
			Pattern{Features: ab,
				MultipleExpectation: 	mu.ConvertIntToBinary(c, d+1)})

		i = i + 1

	}

	// return patterns
	return patterns

}