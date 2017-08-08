// Util provides util to handle common tasks: file and struct operations, string manipulation, etc.
package util

import (

	// sys import
	"os"
	"time"
	"strconv"
	"math/rand"

	// github import
	log "github.com/sirupsen/logrus"

)

func init() {
	// Output to stdout instead of the default stderr
	log.SetOutput(os.Stdout)
	// Only log the warning severity or above.
	log.SetLevel(log.DebugLevel)
}

// Random return pseudo random number in [min, max]
func Random(min, max int) int {
	max = max + 1
	rand.Seed(time.Now().Unix())
	return rand.Intn(max-min) + min
}

// StringInSlice looks for a string in slice.
// It returns true or false and position of string in slice (false, -1 if not found).
func StringInSlice(element string, slice []string) (bool, int) {

	// for element in slice
	for index, value := range slice {
		if value == element {
			return true, index
		}
	}

	// return false, placeholder
	return false, -1

}

// StringToFloat cast a slice of string element to a slice of float64 element.
// If passed mode is 0, for each error encountered in casting, passed default will be inserted
// in slice.
// If passed mode is 1, output slice will contain only correctly casted element.
// It returns slice converted.
func StringToFloat(slice []string, mode int, def float64) []float64 {

	// result declaration
	var result []float64

	// if mode is 0, pre-alloc struct
	if mode == 0 {
		// pre-alloc struct
		result = make([]float64, len(slice))
	}

	for index, value := range slice {

		// cast and error
		casted, err := strconv.ParseFloat(value, 64)

		if mode == 0 {
			if err == nil {
				// add casted
				result[index] = casted
			} else {
				// add default
				result[index] = def
			}
		} else {
			if err == nil {
				// add casted
				result = append(result, casted)
			}
		}
	}

	// return result
	return result

}

// ScalarProduct compute scalar product between two float64 based slices.
// It returns a float64 value.
func ScalarProduct(a []float64, b []float64) float64 {

	// if slices have different number of elements
	if len(a) != len(b) {
		log.WithFields(log.Fields{
			"level":  "error",
			"place":  "mixed",
			"method": "ScalarProduct",
			"msg":    "scalar product between slices",
			"aLen":   len(a),
			"bLen":   len(b),
		}).Error("Failed to compute scalar product between slices: different length.")
		return -1.0
	}

	// init result
	var result float64 = 0.0

	// for each element compute product
	for index, value := range a {
		result = result + (value * b[index])
	}

	// return value
	return result

}

// MaxInSlice return max value in float64 slice
// It returns the max float64 value and index of max in slice.
func MaxInSlice(v []float64) (float64, int) {
	mv := 0.0
	mi := 0
	for i, e := range v {
		if e > mv {
			mv = e
			mi = i
		}
	}
	return mv, mi
}