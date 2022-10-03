# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def transform(data):
  #importing libraries
  from sklearn.preprocessing import OneHotEncoder

  # Converting Taxable Income to float type
  data['Taxable Income']  = pd.to_numeric([x[:-1] for x in data['Taxable Income'].values]).astype(float)

  # Converting type of columns to category
  data['Cheat'] = data['Cheat'].astype('category')
  data['Refund'] = data['Refund'].astype('category')

  #Assigning numerical values and storing it in another columns
  data['Cheat'] = data['Cheat'].cat.codes
  data['Refund'] = data['Refund'].cat.codes

  # One hot encoding Marital Status
  one_hot_encoded_data = pd.get_dummies(data, columns = ['Marital Status'])
  one_hot_encoded_data.rename(columns = {'Marital Status_Divorced':'Divorced','Marital Status_Married':'Married','Marital Status_Single':'Single'}, inplace = True)

  cols = ['Tid','Refund','Single','Divorced','Married', 'Taxable Income','Cheat']

  return one_hot_encoded_data[cols]

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    new_df = transform(df)
    data_training = np.array(new_df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.
    X = data_training[:,:-1]

    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    Y = data_training[:,5]

    #loop your training and test tasks 10 times here
    for i in range (10):

       #fitting the decision tree to the data by using Gini index and no max_depth
       clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
       clf = clf.fit(X, Y)

       #plotting the decision tree
       tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
       plt.show()

       #read the test data and add this data to data_test NumPy
       #--> add your Python code here
       data_test = transform(pd.read_csv('cheat_test.csv'))
       
       from sklearn.model_selection import train_test_split
       from sklearn.metrics import accuracy_score
       from sklearn import metrics
       
       predictions = []
       for data in data_test:
           #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
           #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
           class_predicted = clf.predict(np.array(data_test.values)[:,1:-1])
           

           #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
           #--> add your Python code here
           accuracy = accuracy_score(np.array(data_test.values)[:,6], class_predicted)
           predictions.append(accuracy)
           print(predictions)

       #find the average accuracy of this model during the 10 runs (training and test set)
       #--> add your Python code here
      #  print("Average Accuracy:",metrics.accuracy_score(data_test.iloc[:,6][i], class_predicted))
       np.average(accuracy_score(data_test.iloc[:,6][i], class_predicted))


    #print the accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    #--> add your Python code here
    print("Training Accuracy on Cheat_Training_1: ",metrics.accuracy_score(data_training[i], class_predicted))
    print("Training Accuracy on Cheat_Training_2: ",metrics.accuracy_score(data_training[i], class_predicted))
