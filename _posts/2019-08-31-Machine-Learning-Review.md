---
layout: post
title: "Machine Learning Review"
date: 2019-08-31
---
What is Machine Learning?

We begin with a programming friendly definition. Why are we able to code 
addition of two numbers but unable to code handwritten digit detection 
by computer algorithm? Take a couple of minutes to think about this.

In the case of the first task, we know the exact formula for calculating 
addition of two numbers.  $F(a, b)= a + b$.  However in case of handwritten 
digit recognition through computer program is hard.  We as humans are good 
at recognizing these digits. But we are unable to express the recognition 
process as steps in an algorithm. At best we can come up with rules that 
may work in some cases but it fails to generalize on different ways of 
writing even in limited setup.

ML comes to our rescue here.  The traditional computer algorithms take 
data and rules as input as return the desired output by applying rules to 
the input data.  In the case of ML, we have a bunch of examples or data 
points for which the desired output is known. With these pieces of 
information, ML algorithm learns the rules that maps input to the 
desired output.  The process of learning the rules of mapping is known 
as *training* of ML algorithm.

Once the model is trained, we essentially know the mapping or formula 
to map input to the output. We can then use this formula to predict 
output for any new input.  This process is called *inference* or 
*prediction*.

Now that we have high level understanding how ML differs from the 
traditional programming and we know two stages of ML algorithms, 
let's understand ML terminology in a bit more detail. 

Mapping between input and output is called *model* in ML terminology. 
Model is nothing but a function with certain parameters.

$y = b + w_1 x_1$ is the simplest model mapping input $x_1$ to output 
$y$.  This is called a *linear model*.  Geometrically this equation 
denotes a line with bias b.

The most important prerequisite for ML is data.  There is a famous 
saying "Data is the new oil".  This data is called *training data*. 
The training data has two components: 
* Features which represents characteristics of data item and 
* Label is the outcome that we expect the model to predict

<span style="color:blue">Question: Give an example of data point.</span>

We classify ML algorithms broadly based on availability of label in 
the training data. 
* ML algorithms that are trained in based on label 
information are called **supervised learning algorithm**.  
* The ones that are trained in absence of label information are 
called **unsupervised learning algorithms**.

Within supervised learning, we have two subtypes:
* Label is a real number, we have a **regression** problem.
* Label is a discrete quantity, we have a **classification** problem.

<span style="color:blue">Question: Provide an example each for supervised, 
  unsupervised, regression and classification tasks.</span>

Let's understand more about features: Each data item is represented 
by a number of it's attributes or features.  The model provides mapping 
between these features with the output.

<span style="color:blue">Question: Provide an example of data and it's 
  features.</span>

The features for data item is often determined by the domain experts.  
The features can be of different types.
* Numeric features
* Categorical features.

We need to convert each feature to number somehow so that the ML 
algorithm can consume it in training. Example of numeric feature 
and categorical feature.  Later in this course we will understand 
how to convert or represent categorical attributes to numbers.  

We make sure that all the features are on the same scale.  This 
helps us in getting faster convergence of parameters during training.

These steps are together called as **data pre-processing** step. It 
involves feature transformation like normalization, log transformation 
etc.  It also involved outlier detection and their removal so that the 
training data is not affected by their presence. The outliers are 
results of errors in data collection, or presence of unusual data 
points that are very different from most of the other points.

Now that the data is ready for training, next we fix a model.  There 
are different types of model. The simplest of them is a linear model.  
* **Linear regression**: $y = b + w_1 x_1 + w_2 x_2 + ... + w_n x_n$ for 
data with $n$ features ${x_1, x_2, \ldots, x_n}$.   This is an 
equation of a hyperplane.  <span style="color:blue">Task: 
  Look at the video and understand how does a model on a single 
  feature look like.</span>
* **Logistic regression**: Instead of predicting a real number, we are 
interested in predicting a probability of an input data item belonging 
to a class.  The simplest model that can be used here is a logistic 
regression model.  This model first performs linear combination of 
features and weights: $z = b + w_1 x_1 + w_2 x_2 + …. + w_n x_n = 
b + \sum_i=1^n x_i w_i$, and then apply logistic function to this 
quantity, which returns a number between 0 to 1. 
<span style="color:blue">Question: Draw the logistic function 
  by checking the video.</span> 
It is represented by the following equation 
$Pr(y=1|x) = \frac{1}{1 + exp(-z)}$ 
<span style="color:blue">Question: How does the 
decision boundary between two classes look like?</span>

This form of linear model is unable to separate classes that have 
non-linear decision boundary.  In such cases, we need to perform 
feature crosses and use them as new features so that we can build 
a classifier for non-linearly separable classes.  
* **Polynomial regression**: $y = b + w_1 x_1 + w_2 x_2 +  w_3 x_1^2 + w_4 x_2^2 + w_5 x_1 x_2$.  
Here data has two features $x_1$ and $x_2$.  $x_1 x_2$ is obtained via feature crossing.  
<span style="color:blue">Question: How will this model look like 
  geometrically?<span style="color:blue">
* **Feedforward neural network (NN) models**: We just saw that feature 
crossing is one way of building models for non-linearly separable 
classes.  However we need to construct all the features by hand.  
NN provides a way to automatically construct these features from 
the original input features.  NN can be thought of as learning a 
complex function by breaking it down into smaller functions and 
then we combine the output of these functions to learn a complex 
function.  Check out the neural network architecture in the video.  
We will study NN in detail in the next section. 

Once we define the model, the next task is training.  Training 
involves learning the parameters or weights corresponding to each 
input feature.  In order to train the model, we need to first 
define a loss function.  The loss is a function of parameters 
chosen in the model.  We use optimization techniques to calculate 
optimal values of parameters so that the loss is minimized. 

In case of linear regression, we use the least square as a loss 
function.  The total loss is the sum of loss across all the points. 
This is one of the possible loss functions that can be used.  The 
choice of loss function also depends on the domain knowledge and 
mathematical convenience in optimization.  Other loss function 
could be sum of absolute errors for the regression task.

In binary classification task, we often use cross-entropy loss function.  
Check out the video to understand it intuitively and the equation. This 
loss function is generalized to multi-class classification and is called 
as categorical_crossentropy loss or sparse_categorical_cross entropy loss.  
The categorical cross entropy loss is used when we denote the output with 
one-hot encoding.  The sparse categorical cross entropy loss is used when 
we denote the output as integers. Having defined a loss function, we use 
techniques from mathematical optimization to come up with the optimal 
parameter values.  One of the most widely used techniques is Gradient 
Descent.  

Let’s try to understand it through a simple regression example: Let’s say 
we are trying to learn parameter for a regression problem with a single 
feature.  Also assume that the bias term is 0.  Please check out the 
video on gradient descent to understand more details.  The steps in 
gradient descent are as follows:
* First randomly initialize w_1 to some value. Let’s find the loss at 
this point. 
* Calculate the gradient or slope of loss function at this point.
Move in the negative direction of the slope and update the parameter 
value. Control the movement in the negative direction of slope through 
learning rate.  Specifically,
$ w_1 (new) := w_1 (old) - \alpha gradient$
Mathematically it is written as 
$w_1 (new) := w_1 (old) - \alpha d/dw_1 J(w, b)$
* Repeat these steps till the parameter continues to update.

How do we know if the model is learning?
Learning curve.

Effect of learning rate on convergence?
* Small learning rate - takes a long time to reach the minima.
* Large learning rate - chances of missing the minima.
