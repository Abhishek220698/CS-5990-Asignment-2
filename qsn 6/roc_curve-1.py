# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: ROC Curve
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

# Function to transform the data format
def data_transform(data):
  #importing libraries
  import pandas as pd
  import numpy as np
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

  cols = ['Refund','Single','Divorced','Married', 'Taxable Income','Cheat']

  return one_hot_encoded_data[cols]


# read the dataset cheat_data.csv and prepare the data_training numpy array
# --> add your Python code here
data_training = pd.read_csv('cheat_data.csv')

# transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
# Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [0, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
# be converted to a float.
# --> add your Python code here
df = data_transform(data_training)
new_df = np.array(df.values)
X = new_df[:,:-1]

# transform the original training classes to numbers and add them to the vector y. For instance Yes = 1, No = 0, so Y = [1, 1, 0, 0, ...]
# --> add your Python code here
y = new_df[:,5]

# split into train/test sets using 30% for test
# --> add your Python code here
trainX, testX, trainy, testy = train_test_split(X, y, test_size = .3)

# generate a no skill prediction (random classifier - scores should be all zero)
# --> add your Python code here
ns_probs = [0 for _ in range(len(testy))]

# fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=2)
clf = clf.fit(trainX, trainy)

# predict probabilities for all test samples (scores)
dt_probs = clf.predict_proba(testX)

# keep probabilities for the positive outcome only
# --> add your Python code here
dt_probs = dt_probs[:, 1]

# calculate scores by using both classifeirs (no skilled and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()
