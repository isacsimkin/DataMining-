#This code was written without consulting code written by anyone else. 
# I used the documentation provided by the sklearn and the matplotlib library
# - Isac Simkin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import ShuffleSplit

data = pd.read_csv("grades.csv", sep = ",")
data = data.fillna(0)



def label_grade(row):
	if (row ['Total Score'] >= 900):
		return "A"
	elif (row ['Total Score'] < 900 and row ['Total Score'] >= 800):
		return "B"
	elif (row ['Total Score'] < 800 and row ['Total Score'] >= 700):
		return "C"
	elif (row ['Total Score'] < 700 and row ['Total Score'] >= 600):
		return "D"
	elif (row ['Total Score'] < 600):
		return "F"

#creates a new column with the values obtained in the 'label_grade' function above
data["Letter"] = data.apply(lambda row: label_grade(row), axis = 1)
plt.figure()
	
def plotter (num, title, xaxis, line1,line2, line3, line4):	
	plt.title(title)
	plt.xlabel("Quizzes")
	plt.ylabel("Accuracy")
	plt.subplot(num)
	plt.plot(xaxis, line1, 'bs', xaxis, line2,'g^', xaxis,line3, 'mh', xaxis, line4, 'rv')
	plt.grid(True)
	return plt

plt.figure()
def singlePlotter(title, xaxis, line1, line2, line3, line4):
	plt.title(title)
	plt.xlabel("")
	plt.ylabel("")
	plt.plot(xaxis, line1, 'bs', xaxis, line2,'g^', xaxis,line3, 'mh', xaxis, line4, 'rv')
	plt.grid(True)
	return plt

for i in range(12, 24): #loop that processes 12 classifiers, each time adding another quiz. 
	X = data.values[:, 11 : i]
	Y = data.values[:, 29]

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.334, random_state = 100)
	# each of the algorithms with their respective fittings and predictions. 
	clf = DecisionTreeClassifier()
	clf.fit(X_train, y_train)
	clf_pred = clf.predict(X_test)

	gnb = GaussianNB()
	gnb.fit(X_train, y_train)
	gnb_pred = gnb.predict(X_test)

	mlp = svm.SVC(gamma = 'scale')
	mlp.fit(X_train,y_train)
	mlp_pred = mlp.predict(X_test)

	vcl = VotingClassifier(estimators = [('clf', clf), ('gnb', gnb), ('mlp', mlp)], voting = 'hard')
	vcl.fit(X_train, y_train)
	vcl_pred = vcl.predict(X_test)

	#Cross Validation scores for all of the algorithms used above 

	for clsr, label in zip([clf, gnb, mlp, vcl], ['DecisionTreeClassifier', 'GaussianNB', 'SVC', 'Ensemble']):
		scores = cross_val_score(clsr, X, Y, cv = 5, scoring = 'accuracy')
		print("cross_val_score: %0.2f [%s]" % (scores.mean(), label))


	# obtains each score for each model and calls for plotter function for plotting the scores obtained

	line1 = accuracy_score(y_test,clf_pred)*100
	line2 = precision_score(y_test, clf_pred, average = 'weighted')*100
	line3 = recall_score(y_test, clf_pred, average = 'weighted')*100
	line4 = f1_score(y_test, clf_pred, average = 'weighted')*100

	plotter(221, "Decision Tree", i-11, line1, line2, line3, line4)

	line1 = accuracy_score(y_test,gnb_pred)*100
	line2 = precision_score(y_test, gnb_pred, average = 'weighted')*100
	line3 = recall_score(y_test, gnb_pred, average = 'weighted')*100
	line4 = f1_score(y_test, gnb_pred, average = 'weighted')*100

	plotter(222,"GaussianNB", i-11, line1, line2, line3, line4)

	line1 = accuracy_score(y_test,mlp_pred)*100
	line2 = precision_score(y_test, mlp_pred, average = 'weighted')*100
	line3 = recall_score(y_test, mlp_pred, average = 'weighted')*100
	line4 = f1_score(y_test, mlp_pred, average = 'weighted')*100

	plotter (223, "SVC", i-11, line1, line2, line3, line4)

	line1 = accuracy_score(y_test,vcl_pred)*100
	line2 = precision_score(y_test, vcl_pred, average = 'weighted')*100
	line3 = recall_score(y_test, vcl_pred, average = 'weighted')*100
	line4 = f1_score(y_test, vcl_pred, average = 'weighted')*100

	plotter(224, "Voting Classifier", i-11, line1, line2, line3, line4)
	
	
plt.show()
	
