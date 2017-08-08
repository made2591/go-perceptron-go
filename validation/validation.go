package validation

import (
	"math/rand"
	"time"

	// third part import
	log "github.com/sirupsen/logrus"

	// internal import
	mn "github.com/made2591/go-perceptron-go/model/neural"
	mu "github.com/made2591/go-perceptron-go/util"

	//"fmt"
)

// TrainTestSplit split an array of stimuli in training and testing.
// if shuffle is 0 the function takes the first percentage items as train and the other as test
// otherwise the stimuli array is shuffled before partitioning
func TrainTestSplit(stimuli []mn.Stimulus, percentage float64, shuffle int) (train []mn.Stimulus, test []mn.Stimulus) {

	// create splitting pivot
	var splitPivot int = int(float64(len(stimuli)) * percentage)
	train = make([]mn.Stimulus, splitPivot)
	test = make([]mn.Stimulus, len(stimuli)-splitPivot)

	// if mixed mode, split with shuffling
	if shuffle == 1 {
		// create random indexes permutation
		rand.Seed(time.Now().UTC().UnixNano())
		perm := rand.Perm(len(stimuli))

		// copy training data
		for i := 0; i < splitPivot; i++ {
			train[i] = stimuli[perm[i]]
		}
		// copy test data
		for i := 0; i < len(stimuli)-splitPivot; i++ {
			test[i] = stimuli[perm[i]]
		}

	} else {
		// else, split without shuffle
		train = stimuli[:splitPivot]
		test = stimuli[splitPivot:]
	}

	log.WithFields(log.Fields{
		"level":     "info",
		"msg":       "splitting completed",
		"trainSet":  len(train),
		"testSet: ": len(test),
	}).Info("Complete splitting train/test set.")

	return train, test
}

// KFoldSplit split an array of stimuli in k subsets.
// if shuffle is 0 the function partitions the items maintaining the order
// otherwise the stimuli array is shuffled before partitioning
func KFoldSplit(stimuli []mn.Stimulus, k int, shuffle int) [][]mn.Stimulus {

	// get the size of each fold
	var size = int(len(stimuli) / k)
	var freeElements = int(len(stimuli) % k)

	folds := make([][]mn.Stimulus, k)

	var perm []int
	// if mixed mode, split with shuffling
	if shuffle == 1 {
		// create random indexes permutation
		rand.Seed(time.Now().UTC().UnixNano())
		perm = rand.Perm(len(stimuli))
	}

	// start splitting
	currSize := 0
	foldStart := 0
	curr := 0
	for f := 0; f < k; f++ {
		curr = foldStart
		currSize = size
		if f < freeElements {
			// add another
			currSize++
		}

		// create array
		folds[f] = make([]mn.Stimulus, currSize)

		// copy elements

		for i := 0; i < currSize; i++ {
			if shuffle == 1 {
				folds[f][i] = stimuli[perm[curr]]
			} else {
				folds[f][i] = stimuli[curr]
			}
			curr++
		}

		foldStart = curr

	}

	log.WithFields(log.Fields{
		"level":              "info",
		"msg":                "splitting completed",
		"numberOfFolds":      k,
		"meanFoldSize: ":     size,
		"consideredElements": (size * k) + freeElements,
	}).Info("Complete folds splitting.")

	return folds
}

// RandomSubsamplingValidation perform evaluation on neuron algorithm.
// It returns scores reached for each fold iteration.
func RandomSubsamplingValidation(neuron *mn.Neuron, stimuli []mn.Stimulus, percentage float64, epochs int, folds int, shuffle int) []float64 {

	// results and predictions vars init
	var scores, actual, predicted []float64
	var train, test []mn.Stimulus

	scores = make([]float64, folds)

	for t := 0; t < folds; t++ {
		// split the dataset with shuffling
		train, test = TrainTestSplit(stimuli, percentage, shuffle)

		// train neuron with set of stimuli, for specified number of epochs
		mn.TrainNeuron(neuron, train, epochs, 1)

		// compute predictions for each stimulus in testing set
		for _, stimulus := range test {
			actual = append(actual, stimulus.Expected)
			predicted = append(predicted, mn.Predict(neuron, &stimulus))
		}

		// compute score
		_, percentageCorrect := mn.Accuracy(actual, predicted)
		scores[t] = percentageCorrect

		log.WithFields(log.Fields{
			"level":             "info",
			"place":             "validation",
			"method":            "RandomSubsamplingValidation",
			"foldNumber":        t,
			"trainSetLen":       len(train),
			"testSetLen":        len(test),
			"percentageCorrect": percentageCorrect,
		}).Info("Evaluation completed for current fold.")
	}

	// compute average score
	acc := 0.0
	for i := 0; i < len(scores); i++ {
		acc += scores[i]
	}

	mean := acc / float64(len(scores))

	log.WithFields(log.Fields{
		"level":       "info",
		"place":       "validation",
		"method":      "RandomSubsamplingValidation",
		"folds":       folds,
		"trainSetLen": len(train),
		"testSetLen":  len(test),
		"meanScore":   mean,
	}).Info("Evaluation completed for all folds.")

	return scores
}

// RandomSubsamplingValidation perform evaluation on neuron algorithm.
// It returns scores reached for each fold iteration.
func KFoldValidation(neuron *mn.Neuron, stimuli []mn.Stimulus, epochs int, k int, shuffle int) []float64 {

	// results and predictions vars init
	var scores, actual, predicted []float64
	var train, test []mn.Stimulus

	scores = make([]float64, k)

	// split the dataset with shuffling
	folds := KFoldSplit(stimuli, k, shuffle)

	// the t-th fold is used as test
	for t := 0; t < k; t++ {
		// prepare train
		train = nil
		for i := 0; i < k; i++ {
			if i != t {
				train = append(train, folds[i]...)
			}
		}
		test = folds[t]

		// train neuron with set of stimuli, for specified number of epochs
		mn.TrainNeuron(neuron, train, epochs, 1)

		// compute predictions for each stimulus in testing set
		for _, stimulus := range test {
			actual = append(actual, stimulus.Expected)
			predicted = append(predicted, mn.Predict(neuron, &stimulus))
		}

		// compute score
		_, percentageCorrect := mn.Accuracy(actual, predicted)
		scores[t] = percentageCorrect

		log.WithFields(log.Fields{
			"level":             "info",
			"place":             "validation",
			"method":            "KFoldValidation",
			"foldNumber":        t,
			"trainSetLen":       len(train),
			"testSetLen":        len(test),
			"percentageCorrect": percentageCorrect,
		}).Info("Evaluation completed for current fold.")
	}

	// compute average score
	acc := 0.0
	for i := 0; i < len(scores); i++ {
		acc += scores[i]
	}

	mean := acc / float64(len(scores))

	log.WithFields(log.Fields{
		"level":       "info",
		"place":       "validation",
		"method":      "KFoldValidation",
		"folds":       k,
		"trainSetLen": len(train),
		"testSetLen":  len(test),
		"meanScore":   mean,
	}).Info("Evaluation completed for all folds.")

	return scores

}

// It returns scores reached for each fold iteration.
func MLPRandomSubsamplingValidation(mlp *mn.MultiLayerPerceptron, stimuli []mn.Stimulus, percentage float64, epochs int, folds int, shuffle int, mapped []string) []float64 {

	// results and predictions vars init
	var scores, actual, predicted []float64
	var train, test []mn.Stimulus

	scores = make([]float64, folds)

	for t := 0; t < folds; t++ {
		// split the dataset with shuffling
		train, test = TrainTestSplit(stimuli, percentage, shuffle)

		// train mlp with set of stimuli, for specified number of epochs
		mn.MLPTrain(mlp, stimuli, mapped, epochs)

		// compute predictions for each stimulus in testing set
		for _, stimulus := range test {
			// get actual
			actual = append(actual, stimulus.Expected)
			// get output from network
			o_out := mn.Execute(mlp, &stimulus)
			// get index of max output
			_, indexMaxOut := mu.MaxInSlice(o_out)
			// add to predicted values
			predicted = append(predicted, float64(indexMaxOut))
		}

		// compute score
		_, percentageCorrect := mn.Accuracy(actual, predicted)
		scores[t] = percentageCorrect

		log.WithFields(log.Fields{
			"level":             "info",
			"place":             "validation",
			"method":            "RandomSubsamplingValidation",
			"foldNumber":        t,
			"trainSetLen":       len(train),
			"testSetLen":        len(test),
			"percentageCorrect": percentageCorrect,
		}).Info("Evaluation completed for current fold.")
	}

	// compute average score
	acc := 0.0
	for i := 0; i < len(scores); i++ {
		acc += scores[i]
	}

	mean := acc / float64(len(scores))

	log.WithFields(log.Fields{
		"level":       "info",
		"place":       "validation",
		"method":      "RandomSubsamplingValidation",
		"folds":       folds,
		"trainSetLen": len(train),
		"testSetLen":  len(test),
		"meanScore":   mean,
	}).Info("Evaluation completed for all folds.")

	return scores
}

// RandomSubsamplingValidation perform evaluation on neuron algorithm.
// It returns scores reached for each fold iteration.
func MLPKFoldValidation(mlp *mn.MultiLayerPerceptron, stimuli []mn.Stimulus, epochs int, k int, shuffle int, mapped []string) []float64 {

	// results and predictions vars init
	var scores, actual, predicted []float64
	var train, test []mn.Stimulus

	scores = make([]float64, k)

	// split the dataset with shuffling
	folds := KFoldSplit(stimuli, k, shuffle)

	// the t-th fold is used as test
	for t := 0; t < k; t++ {
		// prepare train
		train = nil
		for i := 0; i < k; i++ {
			if i != t {
				train = append(train, folds[i]...)
			}
		}
		test = folds[t]

		// train mlp with set of stimuli, for specified number of epochs
		mn.MLPTrain(mlp, stimuli, mapped, epochs)

		// compute predictions for each stimulus in testing set
		for _, stimulus := range test {
			// get actual
			actual = append(actual, stimulus.Expected)
			// get output from network
			o_out := mn.Execute(mlp, &stimulus)
			// get index of max output
			_, indexMaxOut := mu.MaxInSlice(o_out)
			// add to predicted values
			predicted = append(predicted, float64(indexMaxOut))
		}

		// compute score
		_, percentageCorrect := mn.Accuracy(actual, predicted)
		scores[t] = percentageCorrect

		log.WithFields(log.Fields{
			"level":             "info",
			"place":             "validation",
			"method":            "KFoldValidation",
			"foldNumber":        t,
			"trainSetLen":       len(train),
			"testSetLen":        len(test),
			"percentageCorrect": percentageCorrect,
		}).Info("Evaluation completed for current fold.")
	}

	// compute average score
	acc := 0.0
	for i := 0; i < len(scores); i++ {
		acc += scores[i]
	}

	mean := acc / float64(len(scores))

	log.WithFields(log.Fields{
		"level":       "info",
		"place":       "validation",
		"method":      "KFoldValidation",
		"folds":       k,
		"trainSetLen": len(train),
		"testSetLen":  len(test),
		"meanScore":   mean,
	}).Info("Evaluation completed for all folds.")

	return scores

}