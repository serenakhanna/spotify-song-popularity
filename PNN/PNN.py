# -*- coding: utf-8 -*-

# 0. import the needed packages
import numpy as np
import pandas as pd
from numpy import genfromtxt

# 1. Read the data points
#can change .csv to indie.csv, pop.csv, rock.csv, electronic.csv or country.csv
data = pd.read_csv('pop.csv', sep=',')
my_data = data[['Danceability','Energy','Key','Loudness','Mode','Speechiness','Instrumentalness','Liveness','Valence','Tempo','Duration(ms)','Time Signature','Class']].apply(lambda x: (x - x.min()) / (x.max() - x.min())) 
cleaned_data = np.array(my_data)

#separate training and testing data
train_test_per = 80/100
my_data['train'] = np.random.rand(len(my_data)) < train_test_per
train = my_data[my_data.train == 1]
train = train.drop('train', axis=1).sample(frac=1)
training =  np.array(train)

test = my_data[my_data.train == 0]
test = test.drop('train', axis=1)
testArray = np.array(test)

#array of success status
desired = list(testArray[:,12])
print("desired")
print(desired)
estimated = []

testArray = testArray[:,0:12]

# 2. Declare the needed variable 
groups = train.groupby('Class')
number_of_classes = len(groups)  # Here we have 2 different classes
dictionary_of_sum = {}
number_of_features  = 12  
sigma = 1
increment_current_row_in_matrix = 0


# 3. Define the point that we wish to classifiy - Clearly it is Red 
point_want_to_classify = testArray

# **INPUT LAYER OF THE PNN **
# 4. Loop inputs
for j in range(0,(len(testArray))):
    
    for k in range(1,number_of_classes+1):
    
    	# 4.1 Initiate the sume to zero 
        dictionary_of_sum[k] = 0
        number_of_data_point_from_class_k = len(groups.get_group(k-1))
    
    	# ** PATTERN LAYER OF PNN **
    	# 5. Loop via the number of training example in class i 
    	# 5.1 - Declare a temporary variable to hold the sum of gaussian distribution sum
        temp_summnation = 0.0
    
    	# 6. Loop via number of points in the class - NUMBER OF POINTS IN THE CLASS!
        #for i in range(1,number_of_data_point_from_class_k+1):
        temptot = 0
		# 6.1 - Implementation of getting Gaussians 
        
        for z in range(0,11):
            temptot =+ (point_want_to_classify[j][z] - training[increment_current_row_in_matrix][z]) * (point_want_to_classify[j][z] - training[increment_current_row_in_matrix][z]) 
        

        temp_sum = -1 * (temptot)
        temp_sum = temp_sum/( 2 * np.power(sigma,2) )

		# 6.2 - Implementation of Sum of Gaussians
        temp_summnation = temp_summnation + temp_sum

		# 6.3 - Increamenting the row of the matrix to get the next data point
        increment_current_row_in_matrix  = increment_current_row_in_matrix + 1

    	# 7. Finally - For K class - the Probability of current data point belonging to that class
        dictionary_of_sum[k]  = temp_summnation 
    
    # 8. Get the classified class 
    classified_class = (max(dictionary_of_sum, key=dictionary_of_sum.get)-1)
    #print("classified as: ")
    estimated.append(classified_class)
print("estimated")
print(estimated)

#predictions and evaluation
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(desired, estimated))
print(classification_report(desired, estimated))

