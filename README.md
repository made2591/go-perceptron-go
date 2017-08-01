# Go Perceptron

A single level perceptron classifier with weights estimated from sonar training data set using stochastic gradient descent.
The implementation is in dev. Planned features:

- complete future features XD (see above)
- find co-workers
- dev a three (then k-parameter) level networks with backprop
- create a ml library in openqasm (just kidding)
- brainstorming / devtesting other algorithms in ml

### Updates

2017-08-01: Introduced validation package and k-fold cross validation.

2017-07-31: I started working on ```mlp``` branch for MLP + back prop. It doens't work yet but...I push first commit after some exp in dev. I delete ```dev``` because of some structs optimization.

2017-07-31: we started working on k-fold validation.

## Dependencies

- [logrus](https://github.com/sirupsen/logrus)

## Run test

To run a simple test just open a shell and run the following:

```
git clone https://github.com/made2591/go-perceptron-go
cd go-perceptron-go
go get https://github.com/sirupsen/logrus
go run main.go
```

## To complete yet

- test methods

## Future features

- [mathgl](https://github.com/go-gl/mathgl.git) for better vector space handling
- multilevel (3 and then parametric) level perceptron to resolve non-linearly separable problems
- some other cool neural model XD
