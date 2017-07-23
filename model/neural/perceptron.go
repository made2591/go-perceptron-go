// Neural provides struct to represents most common neural networks model and algorithms to train / test them.
package neural

import (
	// sys import
	"math/rand"

	// third part import
	// log "github.com/sirupsen/logrus"

	// this repo internal import
	mu "github.com/made2591/go-perceptron-go/util"
)

// Perceptron struct represents a simple Perceptron network with a slice of n weights.
type Perceptron struct {
	// Weights represents Perceptron vector representation
	Weights 		[]float64
	// Bias represents Perceptron natural propensity to spread signal
	Bias    		float64
	// Lrate represents learning rate of perceptron
	Lrate   		float64
}

// #######################################################################################

// RandomPerceptronInit initialize perceptron weight, bias and learning rate.
func RandomPerceptronInit(p *Perceptron) {
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

// UpdateWeights performs update in perceptron weights with respect to passed stimulus.
// It returns error of prediction before updating weights.
func UpdateWeights(p *Perceptron, s *Stimulus) float64 {
	// if false {
	// 	fmt.Println("row")
	// 	bufio.NewReader(os.Stdin).ReadBytes('\n')
	// 	fmt.Println(s.Dimensions)
	// }
	// dummies
	var i int = 0
	// activation and error
	var v, e float64 = Predict(p, s), 0.0
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

// TrainingPerceptron trains a passed perceptron with stimuli passed, for specified number of epoch.
func TrainingPerceptron(p *Perceptron, s *Stimuli, epochs int) {
	p.Weights = make([]float64, len(s.Training[0].Dimensions))
	p.Bias = 0.0

	// init counter
	var epoch, stmindex int = 0, 0
	// for #epoch times
	var sumerror float64 = 0.0
	for epoch < epochs {
		// var prev int = stimuliCorrectlyClassified(p, s)
		for stmindex < len(s.Training) {
			e := UpdateWeights(p, &s.Training[stmindex])
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

// Predict performs a perceptron prediction to passed stimulus.
// It returns a float64 binary predicted value.
func Predict(p *Perceptron, s *Stimulus) float64 {
	if mu.ScalarProduct(p.Weights, s.Dimensions)+p.Bias < 0.0 {
		return 0.0
	}
	return 1.0
}

// Accuracy calculate percentage of equal values between two float64 based slices.
// It returns a float64 percentage value of equal values.
func Accuracy(actual []float64, predicted []float64) float64 {
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