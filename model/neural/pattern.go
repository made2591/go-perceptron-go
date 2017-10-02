// Neural provides struct to represents most common neural networks model and algorithms to train / test them.
package neural

import (
	// sys import
	"os"
	"math/rand"
	"strconv"

	// third part import
	log "github.com/sirupsen/logrus"

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
func CreaTerandomPattERNArray(d int, q int) ([]Pattern) {

	// init patterns
	var patterns []Pattern;

	var c = 0
	// for d times
	for c < q {

		n := rand.Int63n(int64(2^d))
		s := strconv.FormatInt(n, 2)
		var floatingValues = make([]float64, d)

		for ci, cs := range(s) {
			floatingValues[ci], _ = strconv.ParseFloat(string(cs), 64)
		}

		log.WithFields(log.Fields{
			"level":  "debug",
			"place":  "patterns",
			"method": "CreaTerandomPattERNArray",
			"c":   c,
		}).Debug()

		// add casted pattern to training set
		patterns = append(
			patterns,
			Pattern{Dimensions: floatingValues})

		c = c + 1

	}

	// return patterns
	return patterns

}