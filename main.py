# import the models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


wine = load_wine()

data = wine.data
target = wine.target

# print all the attributes of the dataset
print(wine.keys())

# print all the names of all the classifications (target)
print(wine.target_names)

print("\n now we print the numeric data from the zip file \n")

# For each attribute, if it is numeric, print its minimum value, maximum value,
# and the average value. If it is categorical, print the list of possible values it can take.
for feature_name, feature_data in zip(wine.feature_names, wine.data.T):
    # the function is instance checks if the value is an instance of the class int or float
    # and that is hoe we check for all the values if they are numeric or not
    if all(isinstance(value, (int, float)) for value in feature_data):
        print(feature_name, "min:", min(feature_data), "max:", max(feature_data), "avg:", sum(feature_data) / len(feature_data))
    else:
        print(feature_name, "possible values:", set(feature_data))

# Separate the dataset into 4 separate files: 2 files for the features: training and testing (80-20)
# And in the same way 2 files for the classifications (training and testing).
# The files should be in CSV format, with the first line containing the names of the columns.
# The files should be named: wine_features_train.csv, wine_features_test.csv, wine_target_train.csv, wine_target_test.csv

# open the first file
featuresTrainFile = open("wine_features_train.csv", "w")
# open the second file
featuresTestFile = open("wine_features_test.csv", "w")
# open the third file
targetTrainFile = open("wine_target_train.csv", "w")
# open the fourth file
targetTestFile = open("wine_target_test.csv", "w")

# now we separate the data into 4 files as requested
for i in range(len(data)):
    if i > len(data) * 0.8:
        featuresTrainFile.write(str(data[i]) + "\n")
    else:
        featuresTestFile.write(str(data[i]) + "\n")

for i in range(len(target)):
    if i > len(target) * 0.8:
        targetTrainFile.write(str(target[i]) + "\n")
    else:
        targetTestFile.write(str(target[i]) + "\n")


# Perform training and classification using learn-Scikit according to the following models:
# Decision Trees (a)
# Logistic Regression (b)
# K-Nearest Neighbors (KNN) (c)
# For KNN run the model for 3=k to 7=k and do the continuation for each run.
# For each model:
# A. Print the name of the model followed by: score1-f, accuracy, recall, precision.
# B. Calculate and print a confusion matrix including a suitable title. For the presentation, use
# .ConfusionMatrixDisplay

print("\nDecision Trees \n")
# Decision Trees
# create the model
decisionTreeModel = DecisionTreeClassifier()
# train the model
decisionTreeModel.fit(data, target)
# predict the model
decisionTreePredict = decisionTreeModel.predict(data)
# print the score
print("score1-f:", decisionTreeModel.score(data, target))
# print the confusion matrix
print("confusion matrix:")
print(confusion_matrix(target, decisionTreePredict))
# print the confusion matrix display
print("confusion matrix display:")
ConfusionMatrixDisplay(confusion_matrix(target, decisionTreePredict)).plot()
# print the accuracy
accuracy = accuracy_score(target, decisionTreePredict)
print("accuracy:", accuracy)
# print the recall
recall = recall_score(target, decisionTreePredict, average='weighted')
print("recall:", recall)
# print the precision
precision = precision_score(target, decisionTreePredict, average='weighted')
print("precision:", precision)


print("\nLogistic Regression \n")

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
# Create the logistic regression model with increased max_iter
logisticRegressionModel = LogisticRegression(max_iter=10000)
# Train the model
logisticRegressionModel.fit(scaled_data, target)
# Predict using the model
logisticRegressionPredict = logisticRegressionModel.predict(scaled_data)
# Print the score
print("score1-f:", logisticRegressionModel.score(scaled_data, target))
# Print the confusion matrix
print("confusion matrix:")
print(confusion_matrix(target, logisticRegressionPredict))
# Print the confusion matrix display
print("confusion matrix display:")
ConfusionMatrixDisplay(confusion_matrix(target, logisticRegressionPredict)).plot()
# Print the accuracy
accuracy = accuracy_score(target, logisticRegressionPredict)
print("accuracy:", accuracy)
# Print the recall
recall = recall_score(target, logisticRegressionPredict, average='weighted')
print("recall:", recall)
# Print the precision
precision = precision_score(target, logisticRegressionPredict, average='weighted')
print("precision:", precision)

print("\nK-Nearest Neighbors")
# K-Nearest Neighbors
# create 5 models with k=3,4,5,6,7 and print the results
for i in range(3, 8):
    print("\nk=", i)
    # create the model
    kNeighborsModel = KNeighborsClassifier(n_neighbors=i)
    # train the model
    kNeighborsModel.fit(data, target)
    # predict the model
    kNeighborsPredict = kNeighborsModel.predict(data)
    # print the score
    print("score1-f:", kNeighborsModel.score(data, target))
    # print the confusion matrix
    print("confusion matrix:")
    print(confusion_matrix(target, kNeighborsPredict))
    # print the confusion matrix display
    print("confusion matrix display:")
    ConfusionMatrixDisplay(confusion_matrix(target, kNeighborsPredict)).plot()
    # print the accuracy
    accuracy = accuracy_score(target, kNeighborsPredict)
    print("accuracy:", accuracy)
    # print the recall
    recall = recall_score(target, kNeighborsPredict, average='weighted')
    print("recall:", recall)
    # print the precision
    precision = precision_score(target, kNeighborsPredict, average='weighted')
    print("precision:", precision)

# close all the files
featuresTrainFile.close()
featuresTestFile.close()
targetTrainFile.close()
targetTestFile.close()