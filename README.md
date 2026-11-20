# A Single Neuron with Sigmoid Activation: Logistic Regression as the Smallest Neural Network

What is the smallest possible neural network? **One neuron**: a weighted sum of the inputs, squashed by a sigmoid so the output reads as a probability. This notebook builds that neuron from scratch in NumPy, trains it with gradient descent, and uses it to answer a simple question: *is this picture a cat?*

It is the best way to see what every deep-learning framework does under the hood, because nothing is hidden: no autograd, no layers, only the math.

## The problem

The data is a small image set of 64 × 64 RGB pictures labelled **cat (1)** or **non-cat (0)**: 209 training images and 50 test images. Each image becomes a vector of 64 · 64 · 3 = **12,288 numbers**, scaled to [0, 1].

## How the neuron works

For one image `x`:

```
z = wᵀx + b          the weighted sum (12,288 weights + 1 bias)
ŷ = σ(z) = 1 / (1 + e^(-z))      a probability between 0 and 1
loss = -[ y·log(ŷ) + (1 - y)·log(1 - ŷ) ]     cross-entropy
```

The notebook builds the learning algorithm in four small, testable pieces:

1. **Initialise** the weights and bias to zero.
2. **Forward propagation**: compute the predictions and the cost over all images.
3. **Backward propagation**: compute the gradients of the cost with respect to `w` and `b`. For this model they have a clean closed form, `dw = X(A - Y)ᵀ / m` and `db = mean(A - Y)`.
4. **Optimise**: repeat "predict, measure the cost, step against the gradient" for many iterations.

These are then combined in a single `model()` function that returns training and test accuracy.

## What the results show

The cost starts at **0.693 (= ln 2)**, exactly what a model that guesses 50/50 should score, and falls steadily as training proceeds (0.616 after 100 iterations, 0.556 after 400, 0.518 after 700, and so on).

| | Accuracy |
|---|---|
| Train | **86.6 %** |
| Test | **62.0 %** |

That gap is the real lesson. A single neuron has enough capacity to fit 209 training images well, but it cannot learn what "a cat" looks like in a way that carries over to new pictures: it sees only pixel intensities, with no notion of shape or texture. On a tiny dataset it **overfits** and generalises poorly. This is the motivation for deeper networks and convolutional layers.

## What you can take from this

- Logistic regression and a one-neuron network are the same model.
- A falling training cost does not mean a good model; always check held-out data.
- Careful tracking of matrix shapes is what prevents most deep-learning bugs.

## Files

- `A_Single_Neuron_with_Sigmoid_Activation_Function.ipynb`: the full walkthrough with explanations and plots
- `datasets.zip`: the cat / non-cat image data (`train_catvnoncat.h5`, `test_catvnoncat.h5`)
