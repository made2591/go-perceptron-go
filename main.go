package main

//########################## IMPORT ############################

import (
	//"bufio"
	"encoding/csv"
	"fmt"
	m "github.com/made2591/go-perceptron-go/model"
	mn "github.com/made2591/go-perceptron-go/model/neural"
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
func rawExpectedConversion(stimuli *m.Stimuli) {
	// expected string values
	var rawexpected []string
	for _, stimulus := range stimuli.Training {
		if !mu.StringInSlice(stimulus.Rawexpected, rawexpected) {
			rawexpected = append(rawexpected, stimulus.Rawexpected)
		}
	}
	// expected string values
	var stmindex int = 0
	for stmindex < len(stimuli.Training) {
		for intvalue, strvalue := range rawexpected {
			if strings.Compare(strvalue, stimuli.Training[stmindex].Rawexpected) == 0 {
				// conversion to float64 value
				stimuli.Training[stmindex].Expected = float64(intvalue)
			}
		}
		stmindex++
	}
}

// load csv dataset file
func loadCSVFile(path string) m.Stimuli {
	// read content, check error
	content, error := ioutil.ReadFile(path)
	check(error)
	pointer := csv.NewReader(strings.NewReader(string(content)))
	// init stimuli set
	var stimuli m.Stimuli = m.Stimuli{Training: []m.Stimulus{}, Testing: []m.Stimulus{}}
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
		stimuli.Training = append(
			stimuli.Training,
			m.Stimulus{Dimensions: fltrecord, Rawexpected: record[len(record)-1]})
	}
	// cast expected value to numeric
	rawExpectedConversion(&stimuli)
	return stimuli
}

// separate training set in training and testing
func separateSet(s *m.Stimuli, perc float64) m.Stimuli {
	var dataset m.Stimuli = m.Stimuli{}
	var datasetCopy []m.Stimulus = make([]m.Stimulus, len(s.Training))
	perm := rand.Perm(len(s.Training))
	for i, v := range perm {
		datasetCopy[v] = s.Training[i]
	}
	i := 0
	// for test purpose
	// for i < len(s.Training) {
	// 	datasetCopy[i] = s.Training[i]
	// 	i++
	// }
	// i = 0
	var k, split int = 0, int(float64(len(s.Training)) * perc)
	for i < split {
		dataset.Training = append(dataset.Training, datasetCopy[i])
		i++
	}
	for k < len(datasetCopy)-split {
		dataset.Testing = append(dataset.Testing, datasetCopy[i+k])
		k++
	}
	//fmt.Println(len(dataset.Training), len(dataset.Testing))
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
func randomPerceptronInit(p *mn.Perceptron) {
	var i int = 0
	for i < len(p.Weights) {
		// init random threshold weight
		p.Weights[i] = rand.NormFloat64()
		i++
	}
	// init random bias weight and lrate
	p.Bias = rand.NormFloat64()
	p.Lrate = rand.NormFloat64() * 0.01
}

// update weights in perceptron
func updateWeights(p *mn.Perceptron, s *m.Stimulus) float64 {
	// if false {
	// 	fmt.Println("row")
	// 	bufio.NewReader(os.Stdin).ReadBytes('\n')
	// 	fmt.Println(s.Dimensions)
	// }
	// dummies
	var i int = 0
	// activation and error
	var v, e float64 = predict(p, s), 0.0
	e = s.Expected - v
	// bias updating
	p.Bias = p.Bias + p.Lrate*e
	// weights updating
	for i < len(p.Weights) {
		p.Weights[i] = p.Weights[i] + p.Lrate*e*s.Dimensions[i]
		i++
	}
	// if false {
	// 	fmt.Println("weights")
	// 	bufio.NewReader(os.Stdin).ReadBytes('\n')
	// 	fmt.Println(p.Weights)
	// }
	return e
}

// perceptron training
func trainingPerceptron(p *mn.Perceptron, s *m.Stimuli, epochs int) {
	p.Weights = make([]float64, len(s.Training[0].Dimensions))
	p.Bias = 0.0

	// init counter
	var epoch, stmindex int = 0, 0
	// for #epoch times
	var sumerror float64 = 0.0
	for epoch < epochs {
		// var prev int = stimuliCorrectlyClassified(p, s)
		for stmindex < len(s.Training) {
			e := updateWeights(p, &s.Training[stmindex])
			sumerror = sumerror + (e * e)
			stmindex++
		}
		//fmt.Printf(">epoch: %d, lrate: %.3f, error: %.3f\n", epoch, p.Lrate, sumerror)
		// var post int = stimuliCorrectlyClassified(p, s)
		// fmt.Println(epoch, prev, post)
		// bufio.NewReader(os.Stdin).ReadBytes('\n')
		stmindex = 0
		epoch++
	}
	//fmt.Printf("---------------------\n%v\n---------------------\n", s.Training[0])
	//fmt.Printf("---------------------\n%v\n---------------------\n", p.Weights)
	//fmt.Printf("---------------------\n%v\n---------------------\n", s.Testing[0])
	//bufio.NewReader(os.Stdin).ReadBytes('\n')
}

// compute perceptron activation
func predict(p *mn.Perceptron, s *m.Stimulus) float64 {
	if scalarProduct(p.Weights, s.Dimensions)+p.Bias < 0.0 {
		return 0.0
	}
	return 1.0
}

// perceptron
func perceptron(p *mn.Perceptron, s *m.Stimuli, epochs int) []float64 {
	var predictions []float64
	trainingPerceptron(p, s, epochs)
	for _, stm := range s.Testing {
		predictions = append(predictions, predict(p, &stm))
	}
	//fmt.Println(predictions)
	return predictions
}

// compute perceptron activation
func isStimulusCorrectlyClassified(p *mn.Perceptron, s *m.Stimulus) bool {
	if predict(p, s) == s.Expected {
		return true
	}
	return false
}

// areStimuliCorrectlyClassified get mn.Perceptron pointer, m.Stimuli pointer and see if mn.Perceptron correctly find solution for all m.Stimulus inside
func stimuliCorrectlyClassified(p *mn.Perceptron, s *m.Stimuli) int {
	var i, c, l int = 0, 0, len(s.Training)
	for i < l {
		if isStimulusCorrectlyClassified(p, &s.Training[i]) {
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
func evaluateAlgorithm(p *mn.Perceptron, s *m.Stimuli, perc float64, epochs int, folds int) []float64 {
	var scores []float64
	for {
		var ssep m.Stimuli = separateSet(s, perc)
		fmt.Printf("fold: %d, trains: %d, tests: %d\n", len(scores), len(ssep.Training), len(ssep.Testing))
		var predicted []float64 = perceptron(p, &ssep, epochs)
		var actual []float64
		var i int = 0
		for i < len(ssep.Testing) {
			actual = append(actual, ssep.Testing[i].Expected)
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

	// m.Stimuli initialization
	var stimuli m.Stimuli = loadCSVFile("sonar.all_data.csv")
	fmt.Printf("start  rotation...\n")
	// mn.Perceptron initialization
	var perceptron mn.Perceptron = mn.Perceptron{Weights: make([]float64, len(stimuli.Training[0].Dimensions))}
	//randomPerceptronInit(&perceptron)
	perceptron.Weights = make([]float64, len(stimuli.Training[0].Dimensions))
	perceptron.Bias = 0.0
	perceptron.Lrate = 0.01
	perctraintest := 0.67
	epochs := 500
	folds := 3

	fmt.Printf("ending rotation...\n")
	fmt.Printf("\nscores reached: %2.4v\n",
		evaluateAlgorithm(&perceptron, &stimuli, perctraintest, epochs, folds))

}
