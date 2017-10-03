// Neural provides struct to represents most common neural networks model and algorithms to train / test them.
package neural

import (
	// sys import
	"os"
	// "math/rand"
	// "strconv"

	// third part import
	log "github.com/sirupsen/logrus"
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

	Expected []float64

}

// #######################################################################################

// CreaTerandomPattERNArray load a CSV dataset into an array of Pattern.
func CreaTerandomPattERNArray(d int, k int) ([]Pattern) {

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
			Pattern{Dimensions: ab, 
					Expected: 	mu.ConvertIntToBinary(c, d)})

		i = i + 1

	}

	// return patterns
	return patterns

}