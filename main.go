package main

//########################## IMPORT ############################

import (
	//"bufio"
	"encoding/csv"
	"fmt"
	"github.com/made2591/go-perceptron-go/model"
	mu "github.com/made2591/go-perceptron-go/util"
	log "github.com/sirupsen/logrus"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	//"strconv"
	"strings"
)

//########################## METHODS ###########################

// string slice to float slice cast
func rawExpectedConversion(stimuli *Stimuli) {
	// expected string values
	var rawexpected []string
	for _, stimulus := range stimuli.training {
		if !mu.StringInSlice(stimulus.rawexpected, rawexpected) {
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
		var fltrecord []float64 = mu.StringToFloat(record)
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
func separateSet(s *Stimuli, perc float64) Stimuli {
	var dataset Stimuli = Stimuli{}
	var datasetCopy []Stimulus = make([]Stimulus, len(s.training))
	perm := rand.Perm(len(s.training))
	for i, v := range perm {
		datasetCopy[v] = s.training[i]
	}
	i := 0
	// for test purpose
	// for i < len(s.training) {
	// 	datasetCopy[i] = s.training[i]
	// 	i++
	// }
	// i = 0
	var k, split int = 0, int(float64(len(s.training)) * perc)
	for i < split {
		dataset.training = append(dataset.training, datasetCopy[i])
		i++
	}
	for k < len(datasetCopy)-split {
		dataset.testing = append(dataset.testing, datasetCopy[i+k])
		k++
	}
	//fmt.Println(len(dataset.training), len(dataset.testing))
	//bufio.NewReader(os.Stdin).ReadBytes('\n')
	return dataset
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
func updateWeights(p *Perceptron, s *Stimulus) float64 {
	// if false {
	// 	fmt.Println("row")
	// 	bufio.NewReader(os.Stdin).ReadBytes('\n')
	// 	fmt.Println(s.dimensions)
	// }
	// dummies
	var i int = 0
	// activation and error
	var v, e float64 = predict(p, s), 0.0
	e = s.expected - v
	// bias updating
	p.bias = p.bias + p.lrate*e
	// weights updating
	for i < len(p.weights) {
		p.weights[i] = p.weights[i] + p.lrate*e*s.dimensions[i]
		i++
	}
	// if false {
	// 	fmt.Println("weights")
	// 	bufio.NewReader(os.Stdin).ReadBytes('\n')
	// 	fmt.Println(p.weights)
	// }
	return e
}

// perceptron training
func trainingPerceptron(p *Perceptron, s *Stimuli, epochs int) {
	p.weights = make([]float64, len(s.training[0].dimensions))
	p.bias = 0.0

	// init counter
	var epoch, stmindex int = 0, 0
	// for #epoch times
	var sumerror float64 = 0.0
	for epoch < epochs {
		// var prev int = stimuliCorrectlyClassified(p, s)
		for stmindex < len(s.training) {
			e := updateWeights(p, &s.training[stmindex])
			sumerror = sumerror + (e * e)
			stmindex++
		}
		//fmt.Printf(">epoch: %d, lrate: %.3f, error: %.3f\n", epoch, p.lrate, sumerror)
		// var post int = stimuliCorrectlyClassified(p, s)
		// fmt.Println(epoch, prev, post)
		// bufio.NewReader(os.Stdin).ReadBytes('\n')
		stmindex = 0
		epoch++
	}
	//fmt.Printf("---------------------\n%v\n---------------------\n", s.training[0])
	//fmt.Printf("---------------------\n%v\n---------------------\n", p.weights)
	//fmt.Printf("---------------------\n%v\n---------------------\n", s.testing[0])
	//bufio.NewReader(os.Stdin).ReadBytes('\n')
}

// compute perceptron activation
func predict(p *Perceptron, s *Stimulus) float64 {
	if scalarProduct(p.weights, s.dimensions)+p.bias < 0.0 {
		return 0.0
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
func evaluateAlgorithm(p *Perceptron, s *Stimuli, perc float64, epochs int, folds int) []float64 {
	var scores []float64
	for {
		var ssep Stimuli = separateSet(s, perc)
		fmt.Printf("fold: %d, trains: %d, tests: %d\n", len(scores), len(ssep.training), len(ssep.testing))
		var predicted []float64 = perceptron(p, &ssep, epochs)
		var actual []float64
		var i int = 0
		for i < len(ssep.testing) {
			actual = append(actual, ssep.testing[i].expected)
			i++
		}
		scores = append(scores, accuracy(actual, predicted))
		if len(scores) == folds {
			return scores
		}
	}
}

//############################ MAIN ############################

func init() {
	// Log as JSON instead of the default ASCII formatter.
	log.SetFormatter(&log.JSONFormatter{})

	// Output to stdout instead of the default stderr
	// Can be any io.Writer, see below for File example
	log.SetOutput(os.Stdout)

	// Only log the warning severity or above.
	log.SetLevel(log.DebugLevel)
}

func main() {

	// Stimuli initialization
	var stimuli Stimuli = loadCSVFile("sonar.all_data.csv")
	fmt.Printf("start  rotation...\n")
	// Perceptron initialization
	var perceptron Perceptron = Perceptron{weights: make([]float64, len(stimuli.training[0].dimensions))}
	//randomPerceptronInit(&perceptron)
	perceptron.weights = make([]float64, len(stimuli.training[0].dimensions))
	perceptron.bias = 0.0
	perceptron.lrate = 0.01
	perctraintest := 0.67
	epochs := 500
	folds := 3

	fmt.Printf("ending rotation...\n")
	fmt.Printf("\nscores reached: %2.4v\n",
		evaluateAlgorithm(&perceptron, &stimuli, perctraintest, epochs, folds))

}
