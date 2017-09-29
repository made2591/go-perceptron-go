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
	Dimensions []float64

}

// #######################################################################################

// CreaTerandomPattERNArray load a CSV dataset into an array of Pattern.
func CreaTerandomPattERNArray(int d, int q) ([]Pattern) {

	// init patterns
	var patterns []Pattern

	var c = 0
	// for d times
	for {

		n := Int63n(int64(100))
		s := strconv.FormatInt(n, 2)
		var
		for ci, cs in range(len(s)) {
			patterns.Dimensions = s[ci]
		}

		log.WithFields(log.Fields{
			"level":  "debug",
			"place":  "patterns",
			"method": "CreaTerandomPattERNArray",
			"line":   line,
		}).Debug()

		// if end of file reached, exit loop
		if error == io.EOF {
			log.WithFields(log.Fields{
				"level":    "info",
				"place":    "patterns",
				"method":   "CreaTerandomPattERNArray",
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
				"method":      "CreaTerandomPattERNArray",
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
			Pattern{Dimensions: floatingValues})

		lineCounter = lineCounter + 1

	}

	// cast expected values to float64 numeric values
	mapped := RawExpectedConversion(patterns)

	// return patterns
	return patterns, nil, mapped

}