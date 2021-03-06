### About
Simple use-cases inspired from the paper ['ResNet with one-neuron hidden layers is a Universal Approximator'](https://arxiv.org/abs/1806.10909)

### Things I tried

1. Using resnet with one-neuron hidden layer to approximate circular regions in a 2d plane. Here's a look as the network learns to approximate the boundary:

![](https://github.com/vinsis/points-in-2d/blob/master/images/resnet_1h.gif?raw=true)

Note how the boundary seems to have four sides. Using a ResNet with 4 neurons in hidden layer gives a nice hexagon-ish boundary which is a better approximation:

![](https://github.com/vinsis/points-in-2d/blob/master/images/resnet_4h.gif?raw=true)

2. Instead of an identity function `f(x) + x`, I tried adding x scaled by a factor: `f(x) + scale * x`. I added `scale` as a __learnable parameter__ to see if the network things a scaled version is better than an unscaled one. I might be biased, but I did get a faster convergence with a resnet having one-neuron hidden layer.

![](https://github.com/vinsis/points-in-2d/blob/master/images/resnet_v2_1h.gif?raw=true)

### How a resnet with one hidden neuron approximates the boundary:

#### Actual boundary (`x`)

![](https://github.com/vinsis/points-in-2d/blob/master/images/resnet_1.png?raw=true)

#### `f(x)` (forward)

![](https://github.com/vinsis/points-in-2d/blob/master/images/resnet_2.png?raw=true)

#### `f(x) = f(x) + x` (adding identity)

![](https://github.com/vinsis/points-in-2d/blob/master/images/resnet_3.png?raw=true)

#### `g( f(x) )` (forward again)

![](https://github.com/vinsis/points-in-2d/blob/master/images/resnet_4.png?raw=true)

#### `g( f(x) ) + f(x)` (adding identity again)

![](https://github.com/vinsis/points-in-2d/blob/master/images/resnet_5.png?raw=true)

#### `h( g() )` (last linear layer)

![](https://github.com/vinsis/points-in-2d/blob/master/images/resnet_6.png?raw=true)

Note how the two classes are linearly separable. Vanilla networks with at most 2 hidden neurons fail at this task.
