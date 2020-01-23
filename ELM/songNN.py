#Serena Khanna 10145609
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import time

start_time = time.process_time()

class ELM(BaseEstimator, ClassifierMixin):

    """
    3 step model ELM Tool
    """

    def __init__(self,
                 hid_num,
                 a=1):
        """
        Args:
        hid_num (int): number of hidden neurons
        a (int) : const value of sigmoid funcion
        """
        self.hid_num = hid_num
        self.a = a

    def _sigmoid(self, x):
        """
        sigmoid function

        """
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1 / (1 + np.exp(-self.a * x))

    def _add_bias(self, X):
        """add bias to list

        """

        return np.c_[X, np.ones(X.shape[0])]

    def _ltov(self, n, label):
        """
        trasform label scalar to vector

        """
        return [-1 if i != label else 1 for i in range(1, n + 1)]

    def fit(self, X, y):
        """
        learning

        """
        # number of class, number of output neuron
        self.out_num = max(y)

        if self.out_num != 1:
            y = np.array([self._ltov(self.out_num, _y) for _y in y])

        # add bias to feature vectors
        X = self._add_bias(X)

        # generate weights between input layer and hidden layer
        np.random.seed()
        self.W = np.random.uniform(-1., 1.,
                                   (self.hid_num, X.shape[1]))

        # find inverse weight matrix
        _H = np.linalg.pinv(self._sigmoid(np.dot(self.W, X.T)))

        self.beta = np.dot(_H.T, y)

        return self

    def predict(self, X):
        """
        predict classify result

        """
        _H = self._sigmoid(np.dot(self.W, self._add_bias(X).T))
        y = np.dot(_H.T, self.beta)

        if self.out_num == 1:
            return np.sign(y)
        else:
            return np.argmax(y, 1) + np.ones(y.shape[0])


def main():

    """
    Here the code to train and test the ELM for each genre
    """
    
    import pandas as pd
    import numpy
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import normalize
    
    #Import the dataset
    data = pd.read_csv('data.csv', header=None)
    
    #Separate data into genres
    popX = data.iloc[2:102,1:-1].values
    popY = data.iloc[2:102,-1].values
    rockX = data.iloc[103:203,1:-1].values
    rockY = data.iloc[103:203,-1].values
    countryX = data.iloc[204:304,1:-1].values
    countryY = data.iloc[204:304,-1].values
    electronicX = data.iloc[305:405,1:-1].values
    electronicY = data.iloc[305:405,-1].values
    indieX = data.iloc[406:509,1:-1].values
    indieY = data.iloc[406:509,-1].values

    
    #set hidden number of neurons
    hn = 100
    
    #obtain accuracy value
    def accuracy(gold, predicted):
        true_pos = sum(1 for p,g in zip(predicted, gold) if p==1 and g==1)
        true_neg = sum(1 for p,g in zip(predicted, gold) if p==0 and g==0)
        try:
            accuracy = (true_pos + true_neg) / float(len(gold))
        except:
            accuracy = 0
        return accuracy
    
    #obtain popular precision value
    def pprecision(gold, predicted):
        true_pos = sum(1 for p,g in zip(predicted, gold) if p==1 and g==1)
        false_pos = sum(1 for p,g in zip(predicted, gold) if p==1 and g==0)
        try:
            precision = true_pos / float(true_pos + false_pos)
        except:
            precision = 0
        return precision
    
    #obtain unpopular precision value
    def uprecision(gold, predicted):
        true_neg = sum(1 for p,g in zip(predicted, gold) if p==0 and g==0)
        false_neg = sum(1 for p,g in zip(predicted, gold) if p==0 and g==1)
        try:
            precision = true_neg / float(true_neg + false_neg)
        except:
            precision = 0
        return precision

    
    """
    POP
    """
    #convert targets to binary, 1 means popular, 0 means unpopular
    for i in range(len(popY)):
        if int(popY[i]) >= 75:
            popY[i] = 1
        else:
            popY[i] = 0
    
    #splitting testing and training        
    X_train, X_test, y_train, y_test = train_test_split(popX, popY, test_size=0.2, random_state=3)
    
    #preprocessing
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    
    #training
    elm = ELM(hid_num=hn).fit(X_train, y_train)
    
    #testing
    predict = elm.predict(X_test)
    for i in range(len(predict)):
        if int(predict[i]) == 1:
            predict[i] = 0
        else:
            predict[i] = 1
    
    
    #print results
    print("Pop")
    print("Actual:")
    print(y_test)
    print("Predicted:")
    print(predict)
    pa = accuracy(y_test, predict)
    print("Accuracy:")
    print(pa)
    print("Unpopular Precision (0):")
    print(uprecision(y_test, predict))
    print("Popular Precision (1):")
    print(pprecision(y_test, predict))
    print("_______________________________________________")    

    """
    ROCK
    """
    #convert targets to binary, 1 means popular, 0 means unpopular
    for i in range(len(rockY)):
        if int(rockY[i]) >= 75:
            rockY[i] = 1
        else:
            rockY[i] = 0
    
    #splitting testing and training        
    X_train, X_test, y_train, y_test = train_test_split(rockX, rockY, test_size=0.2, random_state=3)
    
    #preprocessing
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    
    #training
    elm = ELM(hid_num=hn).fit(X_train, y_train)
    
    #testing
    predict = elm.predict(X_test)
    for i in range(len(predict)):
        if int(predict[i]) == 1:
            predict[i] = 0
        else:
            predict[i] = 1
    
    
    #print results
    print("Rock")
    print("Actual:")
    print(y_test)
    print("Predicted:")
    print(predict)
    ra = accuracy(y_test, predict)
    print("Accuracy:")
    print(ra)
    print("Unpopular Precision (0):")
    print(uprecision(y_test, predict))
    print("Popular Precision (1):")
    print(pprecision(y_test, predict))
    print("_______________________________________________")  

    """
    COUNTRY
    """
    #convert targets to binary, 1 means popular, 0 means unpopular
    for i in range(len(countryY)):
        if int(countryY[i]) >= 75:
            countryY[i] = 1
        else:
            countryY[i] = 0
    
    #splitting testing and training        
    X_train, X_test, y_train, y_test = train_test_split(countryX, countryY, test_size=0.2, random_state=3)
    
    #preprocessing
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    
    #training
    elm = ELM(hid_num=hn).fit(X_train, y_train)
    
    #testing
    predict = elm.predict(X_test)
    for i in range(len(predict)):
        if int(predict[i]) == 1:
            predict[i] = 0
        else:
            predict[i] = 1
    
    
    #print results
    print("Country")
    print("Actual:")
    print(y_test)
    print("Predicted:")
    print(predict)
    ca = accuracy(y_test, predict)
    print("Accuracy:")
    print(ca)
    print("Unpopular Precision (0):")
    print(uprecision(y_test, predict))
    print("Popular Precision (1):")
    print(pprecision(y_test, predict))
    print("_______________________________________________")   
    """
    ELECTRONIC
    """
    #clear error row
    electronicY = numpy.delete(electronicY, 86, 0)
    electronicX = numpy.delete(electronicX, 86, 0)

    #convert targets to binary, 1 means popular, 0 means unpopular
    for i in range(len(electronicY)):
        #if pd.isnull(electronicY[i]):
            #print(i)
        if int(electronicY[i]) >= 75:
            electronicY[i] = 1
        else:
            electronicY[i] = 0
    
    #splitting testing and training        
    X_train, X_test, y_train, y_test = train_test_split(electronicX, electronicY, test_size=0.2, random_state=3)
    
    #preprocessing
    X_train = normalize(X_train)
    X_test = normalize(X_test)
        
    #training
    elm = ELM(hid_num=hn).fit(X_train, y_train)
    
    #testing
    predict = elm.predict(X_test)
    for i in range(len(predict)):
        if int(predict[i]) == 1:
            predict[i] = 0
        else:
            predict[i] = 1
    
    
    #print results
    print("Electronic")
    print("Actual:")
    print(y_test)
    print("Predicted:")
    print(predict)
    ea = accuracy(y_test, predict)
    print("Accuracy:")
    print(ea)
    print("Unpopular Precision (0):")
    print(uprecision(y_test, predict))
    print("Popular Precision (1):")
    print(pprecision(y_test, predict))
    print("_______________________________________________")   
    """
    INDIE
    """
    #clear error row
    indieY = numpy.delete(indieY, 54, 0)
    indieX = numpy.delete(indieX, 54, 0)
    
    #convert targets to binary, 1 means popular, 0 means unpopular
    for i in range(len(indieY)):
        #if pd.isnull(indieY[i]):
            #print(i)
        if int(indieY[i]) >= 75:
            indieY[i] = 1
        else:
            indieY[i] = 0
    
    #splitting testing and training        
    X_train, X_test, y_train, y_test = train_test_split(indieX, indieY, test_size=0.2, random_state=3)
    
    #preprocessing
    X_train = normalize(X_train)
    X_test = normalize(X_test)
        
    #training
    elm = ELM(hid_num=hn).fit(X_train, y_train)
    
    #testing
    predict = elm.predict(X_test)
    for i in range(len(predict)):
        if int(predict[i]) == 1:
            predict[i] = 0
        else:
            predict[i] = 1
    
    #print results
    print("Indie")
    print("Actual:")
    print(y_test)
    print("Predicted:")
    print(predict)
    ia = accuracy(y_test, predict)
    print("Accuracy:")
    print(ia)
    print("Unpopular Precision (0):")
    print(uprecision(y_test, predict))
    print("Popular Precision (1):")
    print(pprecision(y_test, predict))
    print("_______________________________________________")  
    
    """
    OVERALL ACCURACY
    """
    print("Overall Accuracy:")
    oa = (pa+ra+ca+ea+ia)/5
    print(oa)
    print("Execution Time:")
    print(time.process_time() - start_time, "seconds")
    
    
     
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()