# Week 1 Challenge

## Problem

The objective of this challenge is to accurately predict whether a student will drop out or not. 

## Metric
The performance measure that will be used to judge will be the F1 measure.

The evaluation metric for this challenge is Mean F1-Score. The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The F1 metric weights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously. Thus, moderately good performance on both will be favoured over extremely good performance on one and poor performance on the other.. The formula for the F1 score is:

F1 = 2 * (precision * recall) / (precision + recall)

## Data description
The data has been split into two groups:

- training set (train_data_week_1_challenge.csv)
- test set (test_data_week_1_challenge.csv)

The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each student. Your model will be based on “features” like gender, caste   mathematics marks etc. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each student. It is your job to predict these outcomes. For each student in the test set, use the model you trained to predict whether or not they drop.

** NOTE: You need to encode the target variable with the following codeing:
- continue => 1
- drop     => 0

You will only need to submit the prediction label from the test set in .npy format.
