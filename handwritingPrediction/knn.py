import argparse
from collections import Counter, defaultdict

import random
import numpy
from numpy import median
from sklearn.neighbors import BallTree
import sys
import csv
csv.field_size_limit(sys.maxsize)

class Numbers_old:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        #print self.train_x,valid_set
        f.close()

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.


        # Load the dataset
        f = open(location, 'rb')
        self.train_x = []
        self.test_x = []
        spamreader = csv.reader(f, delimiter=',', quotechar='|') 
        count = 0
        from random import *
        shuffle(spamreader)
        for row in spamreader:
                #print row
                if count == 0 :
                    count += 1
                    continue
                if count <= 200*4:
                    self.train_x.append([ float(x) for x in row[4:]])
                else:
                    self.test_x.append([ float(x) for x in row[4:]])
                    if count == 250*4 :
                        break
               
                count += 1
                
        
        f.close()
        
        #print len(self.test_x[0])
        f = open("train_answers.csv", 'rb')
        self.train_y = []
        self.test_y = []
        spamreader = csv.reader(f, delimiter=',', quotechar='|') 
        count = 0
        for row in spamreader:
                #print row
                if count == 0 :
                    count += 1
                    continue
                if count <= 200:
                    for i in range(4):
                        self.train_y.append(int(row[1]))
                else:
                    for i in range(4):
                        self.test_y.append(int(row[1]))
                    if count == 250 :
                        break
               
                count += 1
        #print self.test_y[196:]
        #self.train_x, self.train_y = train_set
        #self.test_x, self.test_y = valid_set
        #print self.train_x,valid_set
        f.close()



class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k=5):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        # You can modify the constructor, but you shouldn't need to.
        # Do not use another data structure from anywhere else to
        # complete the assignment.

        self._kdtree = BallTree(x)
        self._y = y
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median of the majority labels (as implemented 
        in numpy).

        :param item_indices: The indices of the k nearest neighbors
        """
        assert len(item_indices) == self._k, "Did not get k inputs"

        # Finish this function to return the most common y label for
        # the given indices.  The current return value is a placeholder 
        # and definitely needs to be changed. 
        #
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html
        
        counter = Counter(self._y[item_indices])
        mostCommon = counter.most_common()
        mostCommonList = []
        tuple1 = mostCommon[0]
        prev = tuple1[1]
        for i in mostCommon:
            if (i[1] == prev):
                mostCommonList.append(i[0])
            else:
                break
            prev = i[1]        
 
        return numpy.ceil(numpy.median(mostCommonList))

    def classify(self, example):
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """

        # Finish this function to find the k closest points, query the
        # majority function, and return the predicted label.
        # Again, the current return value is a placeholder 
        # and definitely needs to be changed. 

        indice = self._kdtree.query(numpy.reshape(example,(1,-1)), self._k, False)

        return self.majority(indice[0])

    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrix for the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param test_x: Test data representation
        :param test_y: Test data answers
        """

        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the classify
        # function for each example.

        d = defaultdict(dict)
        data_index = 0
        for ii in range(2):
            for jj in range(2):
                d[ii][jj] = 0

        for xx, yy in zip(test_x, test_y):
            data_index += 1
            retxx = self.classify(xx)
            d[yy][retxx] += 1

            if data_index % 100 == 0:
                print("%i/%i for confusion matrix" % (data_index, len(test_x)))
        return d

    @staticmethod
    def accuracy(confusion_matrix):
        """
        Given a confusion matrix, compute the accuracy of the underlying classifier.
        """

        # You do not need to modify this function

        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        if total:
            return float(correct) / float(total)
        else:
            return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("train.csv")

    # You should not have to modify any of this code

    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")

    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print("\t" + "\t".join(str(x) for x in xrange(2)))
    print("".join(["-"] * 90))
    for ii in xrange(2):
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in xrange(2)))
    print("Accuracy: %f" % knn.accuracy(confusion))
