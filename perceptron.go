package main

//########################## IMPORT ############################

import (
	//"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	//"os"
	"strconv"
	"strings"
)

//########################## STRUCTS ###########################

// Stimuli struct represents a stimuli training and testing set
type Stimuli struct {
	training []Stimulus
	testing  []Stimulus
}

// Stimulus struct represents one stimulus with dimension and desired value
type Stimulus struct {
	dimensions  []float64
	rawexpected string
	expected    float64
}

// Perceptron struct represents a simple Perceptron network with a slice of n weights
type Perceptron struct {
	bias    float64
	weights []float64
	lrate   float64
}

//########################## METHODS ###########################

// search string in slice
func stringInSlice(a string, list []string) bool {
	for _, b := range list {
		if b == a {
			return true
		}
	}
	return false
}

// string slice to float slice cast
func stringToFloat(strrecord []string) []float64 {
	var fltrecord []float64
	for _, strval := range strrecord {
		if fltval, err := strconv.ParseFloat(strval, 64); err == nil {
			fltrecord = append(fltrecord, fltval)
		}
	}
	return fltrecord
}

// string slice to float slice cast
func rawExpectedConversion(stimuli *Stimuli) {
	// expected string values
	var rawexpected []string
	for _, stimulus := range stimuli.training {
		if !stringInSlice(stimulus.rawexpected, rawexpected) {
			rawexpected = append(rawexpected, stimulus.rawexpected)
		}
	}
	// expected string values
	var stmindex int = 0
	for stmindex < len(stimuli.training) {
		for intvalue, strvalue := range rawexpected {
			if strings.Compare(strvalue, stimuli.training[stmindex].rawexpected) == 0 {
				// conversion to float64 value
				stimuli.training[stmindex].expected = float64(intvalue)
			}
		}
		stmindex++
	}
}

// load csv dataset file
func loadCSVFile(path string) Stimuli {
	// read content, check error
	content, error := ioutil.ReadFile(path)
	check(error)
	pointer := csv.NewReader(strings.NewReader(string(content)))
	// init stimuli set
	var stimuli Stimuli = Stimuli{training: []Stimulus{}, testing: []Stimulus{}}
	// read record in file
	for {
		record, error := pointer.Read()
		if error == io.EOF {
			break
		}
		if error != nil {
			log.Fatal(error)
		}
		// conversion
		var fltrecord []float64 = stringToFloat(record)
		// add record to training set
		stimuli.training = append(
			stimuli.training,
			Stimulus{dimensions: fltrecord, rawexpected: record[len(record)-1]})
	}
	// cast expected value to numeric
	rawExpectedConversion(&stimuli)
	return stimuli
}

// separate training set in training and testing
func separateSet(s *Stimuli, perc float64) {
	var datasetCopy []Stimulus = make([]Stimulus, len(s.training))
	perm := rand.Perm(len(s.training))
	for i, v := range perm {
		datasetCopy[v] = s.training[i]
	}
	var i, k, split int = 0, 1, int(float64(len(s.training)) * perc)
	s.training = []Stimulus{}
	for i < split {
		s.training = append(s.training, datasetCopy[i])
		i++
	}
	for k < len(datasetCopy)-split {
		s.testing = append(s.testing, datasetCopy[i+k])
		k++
	}
}

// calculate accuracy percentage
func accuracy(actual []float64, predicted []float64) float64 {
	var i int = 0
	var correct float64 = 0.0
	for i < len(actual) {
		if actual[i] == predicted[i] {
			correct++
		}
		i++
	}
	return correct / float64(len(actual)) * 100.0
}

// compute scalar product
func scalarProduct(a []float64, b []float64) float64 {
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

// initializeWeight get Perceptron pointer
func randomPerceptronInit(p *Perceptron) {
	var i int = 0
	for i < len(p.weights) {
		// init random threshold weight
		p.weights[i] = rand.NormFloat64()
		i++
	}
	// init random bias weight and lrate
	p.bias = rand.NormFloat64()
	p.lrate = rand.NormFloat64() * 0.01
}

// update weights in perceptron
func updateWeights(p *Perceptron, s *Stimulus) {
	// dummies
	var i int = 0
	// activation and error
	var v, e float64 = predict(p, s), 0.0
	v = predict(p, s)
	e = s.expected - v
	// bias updating
	p.bias = p.bias + p.lrate*e
	// weights updating
	for i < len(p.weights) {
		p.weights[i] = p.weights[i] + s.dimensions[i]*p.lrate*e
		i++
	}
}

// perceptron training
func trainingPerceptron(p *Perceptron, s *Stimuli, epochs int) {
	// init counter
	var epoch, stmindex int = 0, 0
	// for #epoch times
	for epoch < epochs {
		// var prev int = stimuliCorrectlyClassified(p, s)
		for stmindex < len(s.training) {
			updateWeights(p, &s.training[stmindex])
			stmindex++
		}
		// var post int = stimuliCorrectlyClassified(p, s)
		// fmt.Println(epoch, prev, post)
		// bufio.NewReader(os.Stdin).ReadBytes('\n')
		stmindex = 0
		epoch++
	}
}

// compute perceptron activation
func predict(p *Perceptron, s *Stimulus) float64 {
	if scalarProduct(p.weights, s.dimensions)+p.bias <= 0.0 {
		return -1.0
	}
	return 1.0
}

// perceptron
func perceptron(p *Perceptron, s *Stimuli, epochs int) []float64 {
	var predictions []float64
	trainingPerceptron(p, s, epochs)
	for _, stm := range s.testing {
		predictions = append(predictions, predict(p, &stm))
	}
	//fmt.Println(predictions)
	return predictions
}

// compute perceptron activation
func isStimulusCorrectlyClassified(p *Perceptron, s *Stimulus) bool {
	if predict(p, s) == s.expected {
		return true
	}
	return false
}

// areStimuliCorrectlyClassified get Perceptron pointer, Stimuli pointer and see if Perceptron correctly find solution for all Stimulus inside
func stimuliCorrectlyClassified(p *Perceptron, s *Stimuli) int {
	var i, c, l int = 0, 0, len(s.training)
	for i < l {
		if isStimulusCorrectlyClassified(p, &s.training[i]) {
			c++
		}
		i++
	}
	return c
}

// compute error
func check(e error) {
	if e != nil {
		panic(e)
	}
}

// evaluate algorithm
func evaluateAlgorithm(p *Perceptron, s *Stimuli, perc float64, epochs int) []float64 {
	var scores []float64
	separateSet(s, perc)
	fmt.Println(len(s.training), len(s.testing))
	var predicted []float64 = perceptron(p, s, epochs)
	var actual []float64
	var i int = 0
	for i < len(s.testing) {
		actual = append(actual, s.testing[i].expected)
		i++
	}
	scores = append(scores, accuracy(actual, predicted))
	return scores
}

//############################ MAIN ############################

func main() {

	// Stimuli initialization
	var stimuli Stimuli = loadCSVFile("sonar.all_data.csv")
	// Perceptron initialization
	var perceptron Perceptron = Perceptron{weights: make([]float64, len(stimuli.training[0].dimensions))}
	randomPerceptronInit(&perceptron)
	perceptron.lrate = 0.01
	fmt.Printf("%v\n", evaluateAlgorithm(&perceptron, &stimuli, 0.9, 500))

}
