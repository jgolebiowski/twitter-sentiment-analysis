# graphAttack

Sentiment analysis of text based on a dataset of Annotated Tweets. 


### Dataset

In a variation on the popular task of sentiment analysis, this dataset contains labels for the emotional content (such as happiness, sadness, and anger) of texts. 40 thousands of examples across 13 labels can be found [here]. The labels have been concentrated to 6 basic Ekman's emotions.

### Pre-processing

Individual words were extracted from Tweets using the Twitter Tokenizer from [NLTK] and later converted to vectors using the [GloVe] embedding scheme.

### Model

The model is composed of a Recurrent NN with 512 LSTM hidden units connected to 6 output units by a linear layer. Throughput the training, dropout ration of 0.5 was used on the LSTM units and Linear Layer.

### Training

Training was performed using an Adam SGD algorithm with early stopping. Best results achieved after 10 epochs.

### Dependencies
* [Python] 3.5 or above
* [pyTorch] - Machine Learning Toolkit
* [torchtext] - Data loaders and abstractions for text and NLP
* [NLTK] - The Natural Language Toolkit



License
----

GPL


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [here]: <https://data.world/crowdflower/sentiment-analysis-in-text>
   [NLTK]: <http://www.nltk.org/>
   [GloVe]: <https://nlp.stanford.edu/projects/glove/>
   [pyTorch]: <http://pytorch.org/>
   [torchtext]: <https://github.com/pytorch/text>




[![Build Status](https://travis-ci.org/jgolebiowski/graphAttack.svg?branch=master)](https://travis-ci.org/jgolebiowski/graphAttack)

Computational graph library for machine learning. The main point is to combine mathematical operation together to form a workflow of choice. The graph takes care of evaluating the gradient of all the inputs to ease up setting up the minimizer.

I have aimed for the library to be simple and transparent so that it would be easy to understand and modify to fit individual needs. The performance was not the primary objective as there are plenty of fast alternatives; the aim was smaller, educational models.

All feedback and ideas for improvement welcome.


### Tutorial
 - To be found in [tutorial.ipynb], covers basic usage and a simple linear regression model

### Examples
Setting up and running / training a dense neural network model
 - [controlDense.py]
 - [controlTrainDense.py]

Setting up and running / training a Convolution neural network model
 - [controlCNN.py]
 - [controlTrainCNN.py]

Setting up and running / training a Recurrent neural network
 - [controlRNN.py]
 - [controlTrainRNN.py]
 
### Features
Matrix operations
- Dot product
- Hadamard product
- element-wise division
- addition

Regularization
- Dropout
- Batch Normalisation

Activations
- ReLU
- Sigmoid
- Tanh
- Softmax

Convolution
- 2d convolution
- Max Pooling

Cost operations
- Quadratic cost
- Cross-entropy for softmax activation

Optimisation
- Adam SGD minimizer, support for batches and continuous series

Misc
- Reshape / flatten
- Slice
- Sum all elements/axis
- Sum all elements squared
- element-wise exponent




### GPU Support
Limited GPU support can be made avaliable by modifying the file:
[graphAttack/gaUtilities/graphAttackFunctions.py]

Commenting out lines 1-2 and uncommenting lines 4-22 will send the dot product computations to your GPU

##### Requirements
 - [pyCUDA] for GPU computations
 - [scikit-cuda] for easy access to cuBLAS

## Additional Resources

http://www.deeplearningbook.org/
section 6.5.1 for more information on computational graphs and the rest of the book for more details about ML/deep learning.


### Dependencies
* [Python] 3.5 or above
* [numpy] - linear algebra for Python
* [scipy] - Scientific Python library, here used for utilities



License
----

MIT


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [numpy]: <http://www.numpy.org/>
   [python]: <https://www.python.org/>
   [scipy]: <https://www.scipy.org/index.html>
   [controlCNN.py]: <https://github.com/jgolebiowski/graphAttack/blob/master/controlCNN.py>
   [controlDense.py]: <https://github.com/jgolebiowski/graphAttack/blob/master/controlDense.py>
   [controlRNN.py]: <https://github.com/jgolebiowski/graphAttack/blob/master/controlRNN.py>
   [controlTrainCNN.py]: <https://github.com/jgolebiowski/graphAttack/blob/master/controlTrainCNN.py>
   [controlTrainRNN.py]: <https://github.com/jgolebiowski/graphAttack/blob/master/controlTrainRNN.py>
   [controlTrainDense.py]: <https://github.com/jgolebiowski/graphAttack/blob/master/controlTrainDense.py>
   [tutorial.ipynb]: <https://github.com/jgolebiowski/graphAttack/blob/master/tutorial.ipynb>
   [graphAttack/gaUtilities/graphAttackFunctions.py]: <https://github.com/jgolebiowski/graphAttack/blob/master/graphAttack/gaUtilities/graphAttackFunctions.py>
   [pyCUDA]: <https://developer.nvidia.com/pycuda>
   [scikit-cuda]: <http://scikit-cuda.readthedocs.io/en/latest/index.html#>