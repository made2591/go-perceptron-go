// Util provides util to handle common tasks: file and struct operations, string manipulation, etc.
package util

import (
	// sys import
	"os"
	"strconv"

	// github import
	log "github.com/sirupsen/logrus"
)

func init() {

	// Log as JSON instead of the default ASCII formatter.
	// log.SetFormatter(&log.JSONFormatter{})

	// Output to stdout instead of the default stderr
	// Can be any io.Writer
	log.SetOutput(os.Stdout)

	// Only log the warning severity or above.
	log.SetLevel(log.DebugLevel)

	//log.Debug("Useful debugging information.")
	//log.Info("Something noteworthy happened!")
	//log.Warn("You should probably take a look at this.")
	//log.Error("Something failed but I'm not quitting.")
	//// Calls os.Exit(1) after logging
	//log.Fatal("Bye.")
	//// Calls panic() after logging
	//log.Panic("I'm bailing.")

}

// StringInSlice looks for a string in slice.
// It returns true or false and position of string in slice (false, -1 if not found).
func StringInSlice(element string, slice []string) (bool, int) {

	for index, value := range slice {
		if value == element {
			return true, index
		}
	}

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
		casted, err := strconv.ParseFloat(value, 64);

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
	if len(a) != len(b) {
		return -1.0
	}
	var i int = 0
	var t float64 = 0.0
	for i < len(a) {
		t = t + (a[i] * b[i])
		i = i + 1
	}
	return t
}