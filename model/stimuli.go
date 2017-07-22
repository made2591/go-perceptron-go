package model

// Stimuli struct represents a stimuli training and testing set
type Stimuli struct {
	Training []Stimulus
	Testing  []Stimulus
}

// Stimulus struct represents one stimulus with dimension and desired value
type Stimulus struct {
	Dimensions  []float64
	Rawexpected string
	Expected    float64
}
