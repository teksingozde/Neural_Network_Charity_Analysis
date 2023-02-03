# Neural Network Charity Analysis
## Overview of Project 

A client wishes to designate an institution for donation who wants his donation to be given to an institution that will be successful. For this reason, the client wants to know which of the institutions asking for donations deserves help.

### Resources:
#### Data Sources:
- charity_data.csv 
- AlphabetSoupCharity.ipynb

#### Software:
- Python 3.9.12
- Jupyter Notebook 6.4.8
- Tensorflow 2.11.0

## Overview of Analysis

First of all, these two columns were not included, as it was seen that the applicant's identity and name were not needed when the data transferred to the DataFrame with the help of the Pandas library were examined. Application_type and Classification columns are columns with two categorical degrees and a density graph is drawn. Columns with numerical values less than 500 for Application_type values and less than 1000 degrees for Classification values are labeled as "Other".

After these operations, the categorical columns are listed and it is possible to fit them under the OneHotEncoder method. In the data preprocessing part, the data set is divided into training and test data, and then it becomes fit with the StandardScaler method.

### 1.	What is Neural Network?
Artificial neural networks are computer systems developed with the aim of automatically performing the abilities of the human brain, such as deriving new information, creating and discovering new information through learning, without any help.
Artificial neural networks emerged as a result of mathematical modeling of the learning process by taking the human brain as an example. It mimics the structure of biological neural networks in the brain and their ability to learn, remember and generalize. Learning process in artificial neural networks is carried out using examples. During learning, input and output information is given and rules are set.

#### 1.1. Advantages of Artificial Neural Networks
- Artificial Neural Networks consist of many cells and these cells work simultaneously and perform complex tasks.
- They have the ability to learn and can learn with different learning algorithms.
- They can produce results (information) for unseen outputs. There is unsupervised learning.
- They can make pattern recognition and classification. They can complete the missing patterns.
- They have fault tolerance. They can work with incomplete or ambiguous information. They show gradual deterioration in faulty situations.
- They can work in parallel and process real-time information.
Artificial neural networks are mainly used in areas such as diagnosis, classification, prediction, control, data association, data filtering and interpretation. It is necessary to compare the properties of the networks with the properties of the problems in order to determine which mesh is more suitable for which problem.

#### 1.2. Usage Areas of Artificial Neural Networks
- Computational Finance: Credit scoring, Algorithmic Trading
- Image processing and computer vision: Face recognition, gesture recognition, object recognition
- Computational biology: Tumor detection, Drug discovery, DNA sequencing
- Energy production: Price and load forecasting
- Automotive, aerospace and manufacturing: Predictive maintenance
- Natural language processing: Voice Assistant, Sentiment analysis

#### 1.3. Biological Fundamentals of Artificial Neural Networks
Artificial neural networks are made up of neurons (nerve cells). Neurons have the ability to process information. Neurons connect with each other to form functions. It is estimated that there are 100 billion neurons in our brain. A neuron can make between 50,000 and 250,000 connections with other neurons, and it is estimated that there are more than 6×10^13 connections in our brain.
Examining the behavior of living things, modeling them mathematically and producing similar artificial models is called cybernetics. The aim is to try to model the learning and application structure of the human brain with neural networks that can be trained, self-organized, learn, and evaluate. In order to perform a job on a computer, it is necessary to know its algorithm. The algorithm is the complete set of basic scripts for converting input to output. However, there may not be a known algorithm for solving some problems. Applications that may change over time in desired and undesirable situations or vary according to the user do not have fixed algorithms. Even if our knowledge is lacking, our data can be plentiful. We can easily make the system learn from thousands of samples, both desirable and undesirable.
Since data collection devices are digital in today's technology, the fact that the data can be accessed, stored and processed reliably gives us an advantage.

#### 1.4. Structure of Artificial Neural Network
Artificial neural networks are structures formed by connecting artificial nerve cells to each other.
Artificial neural networks are examined in three main layers; Input Layer, Hidden Layers and Output Layer.
Information is transmitted to the network through the input layer. They are processed in the intermediate layers and sent from there to the output layer. What is meant by information processing is to convert the information coming to the network into output by using the weight values of the network. In order for the network to produce the correct outputs for the inputs, the weights must have the correct values.
If it consists of many neurons and hidden layers, it is called a multilayer artificial neural network. If it consists of a single layer, it is called a single layer artificial neural network.

## Comparing Before and After Optimization
#### Graph 1. Application Type Density
<img width="530" alt="Screen Shot 2023-02-02 at 1 51 38 PM" src="https://user-images.githubusercontent.com/26927158/216455741-a95fc1c1-4151-4618-939f-6fbef2dc60f8.png">

Looking at the application type values distribution chart, it is seen that the current values are concentrated between -10000 and 1000. There are values higher than 1000, albeit in a small amount.

#### Graph 2. Classification Density
<img width="485" alt="Screen Shot 2023-02-02 at 1 54 02 PM" src="https://user-images.githubusercontent.com/26927158/216456057-86f08f9c-34c3-46bd-9120-eaeefef14eba.png">

Looking at the distribution chart of Classification values, it is seen that the current values are concentrated between --5000 and 5000. There are values higher than 5000, albeit in a small amount.

#### Table 1. Before and After Optimization Neural Network Model
<img width="800" alt="Screen Shot 2023-02-02 at 2 13 12 PM" src="https://user-images.githubusercontent.com/26927158/216456245-070cf6d6-93d6-4ee5-81cc-8795417f4c52.png">

Before the optimization, a two-layer hidden layer and 80 nodes in the first layer and 30 nodes in the second layer were created in the formation of the model. While the weight of the first layer with 80 nodes is 3520, the weight of the first layer with 30 nodes is 2430. The layer called Dense_2 is the output layer and its node is 1. In the 1st and 2nd layers, the activation occurs with relu, in the 3rd layer, that is, in dense_2, the activation occurs with sigmoid and the connection weight of the parameters is 31. The algorithm itself (and the input data) sets these parameters. Hyperparameters are typically learning rate, stack size, or number of epochs. The total number of connections of the parameters is 5,981 and all of these parameters have been learned during the training phase.

After the optimization, if the layers and nodes of the model are interpreted, there are 80 nodes in the first layer, 30 nodes in the second layer, 20 nodes in the third layer and 1 node in the fourth layer. The parameter connection weights of the first three layers are 3200, 2430 and 620, respectively. In the last layer whose activation is sigmoid, the connection weight of the parameter is 21. The total number of connections of the parameters is 6,271 and all of these parameters were used in the training phase.

While there were 5,981 total parameters before the optimization, this number increased to 6,271 after the optimization. The reason is that one more layer and 20 nodes are added to that layer. While the connection weight of the parameters in the first layer was 3520 before the optimization, this number decreased to 3200 after the optimization. However, there was no change in the second layer. After the optimization, a third layer was added and the connection weights of the parameters in 20 nodes in this layer were measured as 620.

The difference between the output layers before and after optimization is only in the connection weights of the parameters. Before optimization, this weight has a higher value.

#### Table 2. Loss and Accuracy with Before and After Optimization
<img width="592" alt="Screen Shot 2023-02-02 at 3 45 01 PM" src="https://user-images.githubusercontent.com/26927158/216456548-c27fcd9c-84ac-4b79-9ce8-5162c61abd9e.png">

Generally, an inverse relationship between loss and accuracy is expected. However, no such relationship was found mathematically. When we look at the table values determined before and after optimization in our model, while the loss before optimization was 0.5534, it increased to 0.5539 in the post-optimization model. While the accuracy value before optimization was 0.728, it was observed as 0.73 after optimization and it is not said that there is much difference.

Contrary to the truth, the loss is not a percentage. It is the sum of the mistakes made for each sample in the training and test sets. Loss is usually used in the training process to find the best parameter value for the model, namely the weights in the neural network. The aim is to minimize this value.
The low accuracy and high loss means that the model made large errors in most of the data. If both the loss and the accuracy are low, the model can be said to have made minor errors in most of the data. However, if both are high, it means that large errors are made in some data. Finally, if the accuracy is high and the loss low, the model is making low errors on only a portion of the data, which is what it should be.

Considering the values in the table, the loss is above 0.55 in both, which is actually lower than 0.50. Accuracy values are close to 75%, but they are not sufficient and this value is normally expected to be higher than 90%.

In our model, accuracy is low and loss is high, which means that the model makes large errors in most of the data.

### 2. How to Increase Accuracy in Neural Networks?

#### 2.1. Add More Dataset

The first thing that we can do to enhance a model accuracy is to add more data to train your model. Having more data is always a good idea. For instance, we do not get a choice to increase the size of training data because we haven’t more data and we can’t find more data from outside.

#### 2.2. Feature Selection

Feature Selection is a process of finding out the best subset of attributes which better explains the relationship of independent variables with target variable.

- Domain Knowledge
Based on domain experience, we select feature(s) which may have higher impact on target variable. For example, an ID is an unique value and a string that have a lot of differentiation between one and another like a human’s name.

- Visualization
It helps to visualize the relationship between variables, which makes your variable selection process easier. Make a visualization of all attribute in graph and visualize the attribute one by one on the graph.

#### 2.3. Normalization

Normalization is a term to ensure that the data haven’t high different value between one and another. Normalization refers to rescaling real valued numeric attributes into the range 0 and 1. It is useful to scale the input attributes for a model that relies on the magnitude of values, such as salary in Indonesia is a millions and value of gender (if you’ve label it) is 0 of 1. It was an obvious example that the value of salary is too big and the other side the value of gender label is too small.

Why would we normalize in the first place?
- Normalization makes training less sensitive to the scale of features, so we can better solve for coefficients.
- The use of a normalization method will improve analysis from multiple models.
- Normalizing will ensure that a convergence problem does not have a massive variance, making optimization feasible.

#### 2.4. Using Hidden Layer

I used loss=binary_crossentropy, optimizer=adam, epoch = 50, verbose=2. 
The number of hidden units are 80, 30 and the accuracy is about 72.8%.
For optimization, the number of hidden units are 80, 30 and 20, the accuracy is about 73%.
It’s not about the higher the value of units will enhance the accuracy of model. But, we can try for many time to get highest accuracy of our model, and so for another parameters. 

#### 2.5. Tunning Algorithm

Tuning Algorithm is about tune the parameters of machine learning algorithm to get the optimum value of each parameter. it will improve the accuracy of model to predict data. To tune these parameters, you must have a good understanding of these meaning and their individual impact on model.
for instance, in neural network you can tune parameter like hidden layer, activation function, epoch, optimizer, batch_size, learning rate, Verbose, dropout, Cross Validation.
This is the real example of enhancing model accuracy using one of parameters tuning algorithm.

## Results
Firstly, unlike our previous pre- and post-optimization study, the “Status” and “Special Considerations” columns were included in our data in this study. Thus, an increase occurred in our data set. For Optimization 3, a "Name" column was added to the data set and only the "EIN" column was not included in the study.

#### Graph 3. Split Type Count for Application Type
<img width="480" alt="Screen Shot 2023-02-02 at 6 16 36 PM" src="https://user-images.githubusercontent.com/26927158/216495333-7be7a52c-f5d5-43f8-a388-abfe20f6c63e.png">

The graphic above is formed by filtering the overweight data as a result of dividing the values in the application_type column into subcategories. Since the only value with overweight is T3, all values outside of it are evaluated in the “Other” category. This graph also shows us the combination graph of the values in the "Other" category. Again, those values less than 500 from these values were not included in the study in any way, that is, restructuring of the data took place. Because data sets with a small volume will reduce the reliability, it was found appropriate to be removed.

#### Graph 4. Split Type Count for Classification
<img width="480" alt="Screen Shot 2023-02-02 at 6 28 09 PM" src="https://user-images.githubusercontent.com/26927158/216496678-7c0a4d35-c23b-4a00-a9eb-18bcaea3884b.png">

It is divided into sub-categories to see the volumes of the values in the Classification column. No filtering is required for the Classification column. The graph above is also the distribution graph of the data.

#### Table 3. Optimization 2 and Optimization 3 Neural Network Models
<img width="823" alt="Screen Shot 2023-02-02 at 7 02 24 PM" src="https://user-images.githubusercontent.com/26927158/216496811-6e7bc97a-5c11-4212-8413-c4ec0ef3ce17.png">

There is no change in the hidden layer values in the optimization 2 and optimization 3 tables. In the first neural network in Optimization 3, the correlation between the weights of the parameters is quite high compared to Optimization 2. And in other neural networks, the connection values between the weights of the parameters do not change. For this reason, the correlation between the weights of the total parameters is also very high in Optimization 3 in general.

#### Table 4. Comparing Optimization 2 and Optimization 3 Loss and Accuracy 
<img width="590" alt="Screen Shot 2023-02-02 at 6 14 07 PM" src="https://user-images.githubusercontent.com/26927158/216497004-e5f92d1c-f871-4378-876f-96a33a9c54b1.png">

In the Optimization 2 value, while the loss value increases, an increase in the accuracy value is observed. However, in Optimization 3, the loss and accuracy values increase together. In Optimization 3, there is an additional "Name" column in the data set compared to Optimization 2. This means that while accuracy increases, it also causes loss. In Optimization 2, some of the data has low rate of mistakes, while in Optimization 3, some big mistakes are made in some data. The increase in the loss value in Optimization 3 is due to the "Name" column.

However, in general, it can be said that the best result among the optimization values is Optimization 2 with low loss and high accuracy.
