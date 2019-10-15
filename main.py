import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

tr_dataset = pd.read_csv('training data.csv')
tr_dataset = tr_dataset.drop(['Instance', 'Wears Glasses', 'Hair Color', 'Body Height [cm]'], axis = 1)
#tr_dataset.info()

#finding the missing data from the training dataset
sb.heatmap(tr_dataset.isnull(), yticklabels=False)
plt.show()

# =============================================================================
# #----------------------------------------------------------------------------
# #finding outliers
# sb.boxplot(x='Income in EUR',data=tr_dataset)
# plt.show()
#
# sb.boxplot(x='Income in EUR',y = 'University Degree', data=tr_dataset)
# plt.show()
#
# sb.boxplot(x='Income in EUR',y = 'Gender', data=tr_dataset)
# plt.show()
#
# figure, ax = plt.subplots(figsize=(10,6))
# ax.scatter(tr_dataset['Income in EUR'], tr_dataset['Age'])
# plt.show()
#
# figure, ax = plt.subplots(figsize=(10,6))
# ax.scatter(tr_dataset['Income in EUR'], tr_dataset['Size of City'])
# plt.show()
# #----------------------------------------------------------------------------
# =============================================================================

# =============================================================================
# #removing outliers
# Q1 = tr_dataset.quantile(0.25)
# Q3 = tr_dataset.quantile(0.75)
# IQR = Q3 - Q1
#
# tr_dataset_o = tr_dataset[~((tr_dataset < (Q1 - 1.5 * IQR)) |(tr_dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
# =============================================================================

# =============================================================================
# #----------------------------------------------------------------------------
# #after removing outliers
# sb.boxplot(x='Income in EUR',data=tr_dataset_o)
# plt.show()
#
# sb.boxplot(x='Income in EUR',y = 'University Degree', data=tr_dataset_o)
# plt.show()
#
# sb.boxplot(x='Income in EUR',y = 'Gender', data=tr_dataset_o)
# plt.show()
#
# figure, ax = plt.subplots(figsize=(10,6))
# ax.scatter(tr_dataset_o['Income in EUR'], tr_dataset_o['Age'])
# plt.show()
#
# figure, ax = plt.subplots(figsize=(10,6))
# ax.scatter(tr_dataset_o['Income in EUR'], tr_dataset_o['Size of City'])
# plt.show()
# #----------------------------------------------------------------------------
# =============================================================================
def addNoise(dataframe, noise_level):
    return dataframe * (1 + noise_level * np.random.rand(len(dataframe)))

#encoding the dataset using target encoding
def targetEncoding(training_set, target_variable, cat_cols, min_sample_leaf, alpha, noise_level):
    tr_target = training_set.copy()
    globalmean = training_set[target_variable].mean()
    cat_mapping = dict()
    default_mapping = dict()

    for column in cat_cols:
        cat_count = training_set.groupby(column).size()
        target_cat_mean = training_set.groupby(column)[target_variable].mean()
        reg_smooth_val = ((target_cat_mean * cat_count) + (globalmean * alpha))/(cat_count + alpha)

        tr_target.loc[:, column] = tr_target[column].map(reg_smooth_val)
        tr_target[column].fillna(globalmean, inplace =True)
        #tr_target[column] = addNoise(tr_target[column], noise_level)

        cat_mapping[column] = reg_smooth_val
        default_mapping[column] = globalmean
    return tr_target, cat_mapping, default_mapping

categorical_columns = ['Gender', 'Country', 'Profession', 'University Degree']
tr_targetX, target_mapping, default_mapping = targetEncoding(tr_dataset, 'Income in EUR', categorical_columns,100, 10, 0.05)

#----------------------------------------------------------------------------
#finding the missing data from the encoded dataset
sb.heatmap(tr_targetX.isnull(), yticklabels=False)
plt.show()
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
#filling the missing values for numerical columns
def fillNA(dataframe, num_cols):
    data = {}
    for column in num_cols:
        data[column] = dataframe[column].mean()
    return dataframe.fillna(value = data)

numerical_columns = ['Year of Record', 'Age']
tr_targetX = fillNA(tr_targetX, numerical_columns)
#----------------------------------------------------------------------------

tr_X = tr_targetX.iloc[:, :-1]
tr_y = tr_targetX.iloc[:, -1]

#splitting the data into training set & testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tr_X, tr_y, test_size = 0.25, random_state = 7)

regressor = RandomForestRegressor(n_estimators=250, criterion='mse', max_depth=30)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
from math import sqrt
print('The accuracy of the regression model is: ',regressor.score(tr_X,tr_y))
print ('RMSE is: ', sqrt(mean_squared_error(y_test, y_pred)))
#----------------------------------------------------------------------------


ts_dataset = pd.read_csv('test data.csv')
ts_dataset = ts_dataset.drop(['Instance', 'Wears Glasses', 'Hair Color', 'Body Height [cm]'], axis = 1)
#ts_dataset.info()

#finding the missing data from the training dataset
sb.heatmap(ts_dataset.isnull(), yticklabels=False)
plt.show()

ts_targetX = ts_dataset.copy()
for column in categorical_columns:
    ts_targetX.loc[:, column] = ts_targetX[column].map(target_mapping[column])
    ts_targetX[column].fillna(default_mapping[column], inplace =True)

#----------------------------------------------------------------------------
#finding the missing data from the encoded dataset
sb.heatmap(ts_targetX.isnull(), yticklabels=False)
plt.show()
#----------------------------------------------------------------------------

ts_targetX = fillNA(ts_targetX, numerical_columns)

ts_X = ts_targetX.iloc[:, :-1]
ts_y = ts_targetX.iloc[:, -1]

ts_y_pred = regressor.predict(ts_X)