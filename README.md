# Machine-Learning-IPT-ParrotAI

# Week 1

Here is a my short summary of what I have achieved to learn in my first week of training under ParrotAi.

Introduction to Machine learning , I have achieved to know a good intro into Machine Learning which include the history of ML ,the types of ML  such supervised, unsupervised, Reinforcement learning. And also answers to questions such why machine learning? , challenges facing machine learning which include insufficient data, irrelevant on data, overfitting, underfitting and there solutions in general.

Supervised Machine algorithms, here I learnt the theory and intuition  behind the common used supervised ML including the KNN, Linear Regressions, Logistic, Regression, and Ensemble algorithm the Random forest. Also not only the intuition but their implementation in python using the sklearn library and parameter tuning them to achieve a best model with stunning accuracy(here meaning the way to regularize the model to avoid overfitting and underfitting).And also the intuition on where to use/apply the algorithms basing on the problem I.e classification or regression. Also which model performs better on what and poor on what based on circumstances.

Data preprocessing and representation here I learnt on the importance of preprocessing the data, also the techniques involved such scaling(include Standard Scaling, RobustScaling  and MinMaxScaler) ,handling the missing data either by ignoring(technical termed as dropping) the data which is not recommended since one could loose important patterns on the data and by fitting the mean or median of the data points on the missing places. On data representation involved on how we can represent categorical features so as they can be used in the algorithm, the method learnt here is One-Hot Encoding technique and its implementation in python using both Pandas and Sklearn Libraries.

Model evaluation and improvement. In this section I grasped the concept of how you can evaluate your model if its performing good or bad and the ways you could improve it. As the train_test_split technique  seems to be imbalance hence the cross-validation technique which included the K-fold , Stratified K-fold and other strategies such LeaveOneOut which will help on the improvement of your model by splitting data in a convenience  manner  to help in training of model, thus making it generalize well on unseen data. I learnt also on the GridSearch technique which included the best method in which one can choose the best parameters for the model to improve the performance such as the simple grid search and the GridSearch with cross-validation technique, all this I was able to implement them in code using the Sklearn library in python.

Lastly the week challenge task given to us was tremendous since I got to apply what I learned in theory to solve a real problem.It was good to apply the workflow of a machine learning task starting from understanding the problem, getting to know the data, data preprocessing , visualising the data to get more insights, model selection, training the model  to applying the model to make prediction.

## Conclusion
In general it was a tuff week basing on the modules but all in all i was able to grasp and learn much in this week from basic foundation of Machine Learning to the implementations of the algorithms in code. The great achievement so far is the intuition behind the algorithm especially supervised ones. Though yet is much to be covered but the accomplishment I have attained so far its a good start to say to this journey on Machine learning. My expectation on the coming week is on having a solid foundation on deep learning.
#


# WEEK 2
In this week I covered the concept of Deep learning, from the Multi Layer Perceptron(MLP), Convolution Neural Network(CNN) to Sequence models.The following is the summary of what I have managed to learn in this week.

Deep Neural Network. As the formal definition of Deep Learning which is the sub class of machine learning which unlike ML, deep learning  learns underlying features in data using multiple processing layers cascaded together. In this section I learned the building blocks of neural networks which include the layers(input, hidden and output), input data, activation functions, loss function and optimizers. The deep in deep learning is just the depth of layers.The loss function here depend on the problem I.e regression MSELoss (mean square error), Binary cross entropy for two-class classification and cross entropy for multi-class classification. The training of deep neural networks involves finding the best values of parameters θ (w and b) to effectively minimise the loss function through a technique known as back propagation. This is achieved by optimisation functions such the Gradient Descent, Stochastic Gradient Descent(SGD), Mini-batch SGD. The problem in training is setting the learning rate, this can be solved by using the adaptive learning rates algorithms such as Adam, Momentum, Adagrad, RSMProp.Also learnt the various regularisation methods such as reducing network size, adding weight regularisation(can be L1 or L2) and dropout .Then I learned the code implementation of deep neural network in python using the pytorch framework.

Convolutional neural network, In this section I learned the intuition behind the CNN which is specialised in visual data(images) that employ the convolution operation rather than the normal matrix operation. The task in computer vision involve image classification, image classification with localisation, object detection and image segmentation. Also learned the composition of a CNN which include the convo layers consisting of filters, feature maps and a pooling layer then after the convo there comes the full connected layers of a neural network.Hence its implementation in python using the pytorch framework. Got an idea of  transfer learning in which rather than building your own model from scratch you can use the trained model which is developed already and tune it to your context such pre-trained models include AlexNet,ResNet

Sequence models. Here the main concept learnt is the limitation that faced MLP and CNN which mainly is, they accept fixed sized input and generate a fixed sized output. The other problem is that they do not provide best way to handle long-term dependencies, hence the Recurrent Neural Network(RNN) which is the family of the neural network which handles sequential data. There many variants can be one-to-one , one-to-many, many-to-one and many-to-many architectures basing on the input and output. The training in RNN is through back propagation through time, where here the loss is the sum of loses over time. The main two challenges of RNN is the vanishing and exploding gradient. The problem of vanishing gradient is controlled by using gates through Long Short-term Memory and Gated Recurrent Unit(GRU).The application of sequential modelling such as speech recognition, machine translation, image captioning and many others.

The concepts pertaining deep learning in this week were clear and had great time going through them, even though I had to grasp a lot of concepts in a short period of time.Challenges faced is on the last part of the week module which I had no ample time to go deep into the concept of sequential modelling. The achievement for this week is the intuition behind the deep learning concepts in mathematical terms. I was able to understand the mathematical background of concepts like back propagation algorithm.


# WEEK 3
Through this week I have manage to cover the following concepts.
Recommender systems, in this concept I managed to grasp some basic intuition of recommender system, which they have an advantage of increasing sales as result of very personalised offers and an enhanced customer experience as seen in amazon, youtube, AliExpress and others.

Types of recommender systems
	1. Content based
	2. Collaborative filtering

Methods of achieving collaborative filtering
	1. Memory-based
	identifies clusters of users and utilizes the interactions of one specific user to predict the interactions of 
  other similar users.                   
	Dis-advantage: encounter major problems with large sparse matrices.
	2. Model-based
	based on machine learning and data mining techniques
	Advantage: can work with large sparse matrices.
  
Apart from recommender systems I went back and learnt more on sequential modelling which I had a partial intro to the subject last week. The new concepts grasped this week pertaining the subject include the mathematical aspect of gated recurrent unit (GRU) and Long Short-term Memory(LSTM) which tends to avoid vanishing gradient using so called cells/gates. Basing on this gates the GRU and LSTM works best in long term dependencies environment since they know when to or not to update the values which is much of mathematical approach based on equations. The GRU is simple hence can be used to build a large dense network but the LSTM is more complex and still more preferable in most cases.

Also I went through my week assignment which is no image captioning as one of the application of sequence modelling.The goal of image captioning is to convert a given input image into a natural language description. The encoder-decoder framework is widely used for this task. The image encoder is a convolutional neural network (CNN). In this assignment, we used resnet 152 pretrained model. The decoder is a long short-term memory (LSTM) network.For the encoder part, the pretrained CNN extracts the feature vector from a given input image. The feature vector is linearly transformed to have the same dimension as the input dimension of the LSTM network. For the decoder part, source and target texts are predefined.Using these source and target sequences and the feature vector, the LSTM decoder is trained as a language model conditioned on the feature vector. The implementation of this assignment was through PyTorch.


Lastly on the end of the week learnt how to plan and execute machine learning experiment which involve the following key issues
1.Understanding the problem
2.Understanding the data: 
3.Writing readable and reusable code
4.The use of arg parse
5.Choosing the appropriate metric for the problem

This week marks the end of theory part, the coming weeks will be more into practical part, thus helping us to apply all the underlying concepts of machine learning we have learnt so far. The achievement for this week was the knowledge I got on methods to follow when conducting machine learning experiment and the implementation of image caption in python using pytorch was joyful.
