package model

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
