import math
import numpy
import sklearn.metrics.pairwise
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model


train_files = []
test_files = []

for i in range(1, 6):
	train_files.append("u" + str(i) + ".base")

for i in range(1, 6):
	test_files.append("u" + str(i) + ".test")


"""
	data[i] -> user id | item id | rating | timestamp
"""

"""
	-> 943 users
	-> 1682 movies
"""

users = 943
movies = 1682

errorSet = []


for k in range(1):

	for testCase in range(0, 5):

		print "testCase : ", testCase

		with open("../ml-100k/" + train_files[testCase]) as fp:
			data = fp.readlines()


		""" Prepare data """
		for i in range(len(data)):
			data[i] = data[i].strip('\r\n')
			data[i] = data[i].split('\t')[:3]

		ratingMatrix = numpy.array([numpy.zeros(movies) for i in range(users)])

		for point in data:

			currUser = int(point[0]) - 1
			currMovie = int(point[1]) - 1
			currRating = float(point[2])

			ratingMatrix[currUser][currMovie] = currRating


		with open("../ml-100k/" + test_files[testCase]) as fp:
			testData = fp.readlines()

		for i in range(len(testData)):
			testData[i] = testData[i].strip('\r\n')
			testData[i] = testData[i].split('\t')[:3]

		mse = 0.0
		mae = 0.0

		ALL = len(testData)
		num = 1
		unable = 0

		allErrors = []

		for point in testData:

			num += 1

			print "testCase : ", testCase, "num : ", num

			currUser = int(point[0]) - 1
			currMovie = int(point[1]) - 1
			actualRating = float(point[2])


			y = np.append(ratingMatrix[currUser][:currMovie], ratingMatrix[currUser][currMovie + 1:])
			X = np.concatenate((ratingMatrix.T[:, :currUser], ratingMatrix.T[:, currUser + 1:]), axis = 1)
			X = np.concatenate((X[:currMovie], X[currMovie + 1:]))

			validLabels = np.nonzero(y)
			y = y[validLabels]
			X = X[validLabels]

			clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
			clf.fit(X, y)

			predictVector = np.concatenate((ratingMatrix.T[currMovie, :currUser], \
							ratingMatrix.T[currMovie, currUser + 1:]))[np.newaxis, :]

			try:
				predicted = clf.predict(predictVector)
				mae += math.fabs(actualRating - predicted)

			except:
				unable += 1


		""" Compute error """
		mae /= (len(testData) - unable)
		nmae = mae/4.

		print str("MAE" + str(mae) + " " + "NMAE" + str(nmae) + ' ' + str(unable) + '\n')
		



