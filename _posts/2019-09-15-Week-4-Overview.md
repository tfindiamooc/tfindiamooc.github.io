---
layout: post
title: "Overview of Week 4"
date: 2019-09-15
---

Welcome to the week 4 of the course! In this week, we will learn how to build basic ML models with TensorFlow 2.0.  We will be building the following models:

* [Image Classification](https://www.youtube.com/watch?v=toduAqaz_EA&list=PLOzRYVm0a65cTV_t0BYj-nV8VX_Me6Es3&index=16)
* [Regression](https://www.youtube.com/watch?v=AO8zuIcx0Aw&list=PLOzRYVm0a65cTV_t0BYj-nV8VX_Me6Es3&index=17)
* [Classification of Structured Data](https://youtu.be/lCopG4tDSok?list=PLOzRYVm0a65cTV_t0BYj-nV8VX_Me6Es3)
* [Text classification](https://youtu.be/bvYIicaVNTE?list=PLOzRYVm0a65cTV_t0BYj-nV8VX_Me6Es3)

As we discussed in Week 1, after building the model, we will encounter that we have achieved one of the following: (i) **overfitting**, (ii) **underfitting** or (iii) **just right fit model**.  We will concretely understand the concept of 
underfitting and overfitting through practical examples.  Here is [the video for overfitting and underfitting](https://www.youtube.com/watch?v=j6uL6c14pUY&list=PLOzRYVm0a65cTV_t0BYj-nV8VX_Me6Es3&index=20). 

Finally we often need to save the trained ML model so that we can use it later for prediction or use it to initialize weights of model in the next run.  We will cover the mechanism of storing a model in TensorFlow 2.0. We can either store 
only weights of the model, store only architecture of the model or the both.  There are variety of formats in which we 
can store the model like JSON, YMAL and HDF5.  The model can be stored at the end of training or after every few epochs during the training.  The later is very useful for models that take long time to train.  In such models, we can 
take the intermediate model and use it to get a sense of how the model is performing on the given task. We will demonstrate how to restore the model from the latest checkpoint or from any checkpoint in the past.  Here is the [video 
for saving and restoring models](https://youtu.be/Wi44C1sDBqk?list=PLOzRYVm0a65cTV_t0BYj-nV8VX_Me6Es3).

## Note on teaching style/method

In this week, we will be writing code for building ML models using concepts learnt so far in the course.  The lectures mainly focus on walking you through the colab notebooks and explaining how the model looks like.  We will not be 
explaining any of the concepts explained so far in the course and we advise you to revisit the respective videos again. This has been in order to make more time to learn new concepts.

# Broad steps in training ML model

As discussed in week 1, most of these models will have the following broad steps:

* Load training data from files or from inbuilt datasets from keras or TensorFlow datasets.
* Data Exploration including visualization, dataset statistics, etc.
* Data Preprocessing which involves normalization, outlier removal etc.
* Model construction - We choose the appropriate model depending on the problem class and data exploration.  There are two steps in tf.keras API: (i) Model building - Choose the model and write code to build it.  (ii) Model Compilation where we specify loss function, optimization algorithm to use for training and metrics to track during training process.
* Model training by specifying training data and also validation data for diagnosing problems with training (like underfitting/overfitting, inappropriate learning rates etc.)
* Model evaluation on test data.
* Prediction on new data.
* Error analysis (in some cases), where we analyze errors made by model.  This learning is usually fed back in making changes in the mdoel for obtaining better performance. 

## Training Data


| Problem   |      Dataset      |  Source | ML problem |
|-----------|-----------------|--------|---------------|
| Image Classification | [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)  | `keras.dataset.fashion_mnist` | Classification |
| Regression | [AutoMPG](https://archive.ics.uci.edu/ml/) | [UCI ML repository](https://archive.ics.uci.edu/ml/) | Regresson |
| Structured Data | [Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+Disease) | Cleveland Clinic Foundation | Classification |
| Text Classification | [IMDB](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb) | `tensorflow_datasets`| Classification |


As you can see that we have datasets from mixed sources so that we can demonstrate how to load data from different sources. 

## Data Visualization
We will be using `matplotlib` for visualization of structured data as well as image related visualizations (`imshow`). 

## Data Preprocessing

* Structured data: We apply normalization on data in case of continous attributes.  We convert discrete attributes into appropriate representations like one-hot encoding, interger encoding or embeddings (which is handled as part of model.)
* Image Data: We augment image data through rotation, translation and by adding noise.  This enables us to obtain more training examples from the existing ones.
* Text Data: We construct vocabulary, discard in-frequent words. We also transform strings into appropriate feature representation like integer encoding or one-hot encoding.

### Model Construction


**Specification**

We fix the architecture of the models and for problems we will be using feed-forward neural networks (FFNN).  We will be using `tf.keras.models.Sequential` model for building FFNNs.  Typical model specifciation looks like - 

```
keras.Sequential([
    keras.layers.Dense(num_hidden_units, kernel_regularizer=regularization, activation=activation),  #hidden layer
    keras.layers.Dense(num_hidden_units, kernel_regularizer=regularization, activation=activation),  #hidden layer
    keras.layers.Dense(num_hidden_units, activation=activation)                                      #output layer
])
```

We usually use `relu` activation in the hidden layers.  The activation for the output layer depends on the problem: 
* For regression problem, we use `linear` activation, which is the default activation.  
* For binary classification problems, we use `sigmoid` activation.
* For multi-class classification problems, we use `softmax` activation. Here we get a probability distribution over all labels.

The number of hidden layers and units within them are part of model configuration or hyperparameters.  The number of units in the output layer depends on the problem:
* For a single output regression or binary classification problem, we have a single unit in the output layer.
* For multiclass classification problem, like MNIST or Fashion MNIST, we have number of units equal to the number of classes.

**Compilation**

After specifying the model, we need to compile them where we specify loss function, optimization algorithm and metrics to track during model training.

```
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=list_of_metrics)
```

Depending of the problem, we provide the loss function.  
* For regression, we use mean squared error `mse` or mean absolute error `mae`.
* For binary classification, we use `binary_cross_entropy_loss`
* For multi-class classification, we use either `categorical_cross_entropy_loss` or `sparse_categorical_cross_entropy_loss`.

## Model Training

We perfom model training with `model.fit` function where we give training data - features and labels, validation data - features and labels(optional), and number of epochs for training.

## Model Evaluation

We use `model.evaluate` function for evaluating model performance on test data.  We provide both test data and labels to the evaluate function.

## Model Training

Finally, we use `model.predict` function to predict labels for new data.  Here we only specify the data and the function returns its prediction in terms of labels.

## Review

The following is the list of important functions and concepts to remember from this week

| Task | API/Function |
| -----| ------------ |
| Model Specification | [`tf.keras.Sequential`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Sequential?hl=en) |
| Add layers in model | [`layers`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers?hl=en), [`layers.Dense`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/Dense?hl=en)|
| Activation | [`relu`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/activations/relu?hl=en), [`sigmoid`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/activations/sigmoid?hl=en), [`linear`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/activations/linear?hl=en), [`softmax`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/activations/softmax?hl=en) |
| Loss functions | [`mse`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/losses/MSE?hl=en), [`binary_cross_entropy`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/losses/binary_crossentropy?hl=en), [`sparse_categorical_cross_entropy`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy?hl=en) |
| Optimizers | [`adam`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/optimizers/Adam?hl=en) |
| Metrics | [`accuracy`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/metrics/Accuracy?hl=en) |
| Data load | [`tf.data.Dataset`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/data/Dataset?hl=en) |
