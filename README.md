# Recurrent Neural Network with Target Propagation Through Time

This code implements the TPTT-RNN network as described in the paper

Manchev, N. and Spratling, M., "Target Propagation in Recurrent Neural Networks", Journal of Machine Learning Research 21 (2020) 1-33

## Training a TPTT SRNN on the synthetic problems

The pathological synthetic problems are known to be very challenging for SRNs to solve, as they require the memorization of long-term correlations. Here is a brief description of the four individual problems selected for testing TPTT:

* **The Temporal Order Problem** The goal in this problem is sequence classification. A sequence of length T is generated using a set of randomly chosen symbols {a,b,c,d}. Two additional symbols -- X and Y are selected at random and presented at positions t1 in [T/10, 2\*T/10] and t2 in [4\*T/10, 5\*T/10]. The network must predict the correct order of appearance of X and Y out of four possible options: XX, XY, YX, YY
	
* **The 3-bit Temporal Order Problem** This problem is similar to the Temporal Order Problem, but the positions of interest are increased to three -- t1 in [T/10, 2\*T/10], t2 in [3\*T/10, 4\*T/10], and t3 in [6\*T/10, 7\*T/10]. This also leads to an increased number of possible outcomes that the network must learn to predict: XXX, XXY, XYX, XYY, YXX, YXY, YYX, YYY
	
* **The Adding Problem** The problem presents the network with two input channels of length T. The first channel is a sequence of randomly selected numbers from [0,1]. The second channel is a series of zeros, with the exception of two positions t1 in [1, T/10] and t2 in [T/10, T\*2], where its values are ones. The ones at positions t1 and t2 act as markers that select two values from the first channel: X1 and X2. The target that the network must learn to predict is the result of (X1 + X2)/2
	
* **The Random Permutation Problem** This task receives a sequence of symbols T, with the symbol at t1 being either 1 or 0 and also being identical to the symbol at tmax. All the other symbols in the sequence are randomly sampled from [3,100]. This condition produces two types of sequences -- (0, a\_t2, a\_t3, ... , a\_(tmax-1)}, 0) and (1, a\_t2, a\_t3, ... , a_(tmax-1), 1) where a\_t is randomly sampled from [3,100]. The goal is to predict the symbol at tmax, which only depends on the symbol at t1, while the other symbols in the sequence act as distractors.

To run this network start the RNN\_dtp.py script. Here is an example run that trains the network on the Temporal Order Problem with T=10

```
$ python RNN_dtp.py --task temporal --min 10 --max 10
```

## Training a TPTT SRNN on MNIST

The MNIST data was used to define the MNIST classification from a sequence of pixels problem, originally devised by [Yann LeCun](http://yann.lecun.com/exdb/mnist/). In this challenge, the images are presented to the network one pixel at a time in a scanline order. This results in a very long range dependency problem as 784 pixels translate to 784 time steps (i.e. T=784).

To run this network start the MNIST\_dtp.py script.

```
$ python MNIST_dtp.py
```

This script doesn't accept arguments so parameters have to be tweaked directly in main().

## License and citations

All code in this repository is is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License. (C) 2018 Nikolay Manchev. To cite use of the code please use 

Manchev, N. and Spratling, M., "Target Propagation in Recurrent Neural Networks", Journal of Machine Learning Research 21 (2020) 1-33
