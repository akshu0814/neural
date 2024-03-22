# import pandas as pd
# import re
# from DecisionTree import *
# from AdaBoost import *
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Loading the titanic dataset
# titanic_data = pd.read_csv('train.csv')

# # Feature that tells whether a passenger had a cabin on the Titanic
# titanic_data['Has_Cabin'] = titanic_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# # Create new feature FamilySize as a combination of SibSp and Parch
# titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1

# # Create new feature IsAlone from FamilySize
# titanic_data['IsAlone'] = 0
# titanic_data.loc[titanic_data['FamilySize'] == 1, 'IsAlone'] = 1

# # Remove all NULLS in the Embarked column
# titanic_data['Embarked'] = titanic_data['Embarked'].fillna('S')

# # Remove all NULLS in the Fare column
# titanic_data['Fare'] = titanic_data['Fare'].fillna(titanic_data['Fare'].median())

# # Remove all NULLS in the Age column
# age_avg = titanic_data['Age'].mean()
# age_std = titanic_data['Age'].std()
# age_null_count = titanic_data['Age'].isnull().sum()
# age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
# titanic_data.loc[np.isnan(titanic_data['Age']), 'Age'] = age_null_random_list
# titanic_data['Age'] = titanic_data['Age'].astype(int)

# # Define function to extract titles from passenger names
# def get_title(name):
#     title_search = re.search(' ([A-Za-z]+)\.', name)
#     if title_search:
#         return title_search.group(1)
#     return ""

# titanic_data['Title'] = titanic_data['Name'].apply(get_title)

# # Group all non-common titles into one single grouping "Rare"
# titanic_data['Title'] = titanic_data['Title'].replace(
#     ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

# titanic_data['Title'] = titanic_data['Title'].replace('Mlle', 'Miss')
# titanic_data['Title'] = titanic_data['Title'].replace('Ms', 'Miss')
# titanic_data['Title'] = titanic_data['Title'].replace('Mme', 'Mrs')

# # Mapping Sex
# titanic_data['Sex'] = titanic_data['Sex'].map({'female': 0, 'male': 1}).astype(int)

# # Mapping titles
# title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
# titanic_data['Title'] = titanic_data['Title'].map(title_mapping)
# titanic_data['Title'] = titanic_data['Title'].fillna(0)

# # Mapping Embarked
# titanic_data['Embarked'] = titanic_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# # Mapping Fare
# titanic_data.loc[titanic_data['Fare'] <= 7.91, 'Fare'] = 0
# titanic_data.loc[(titanic_data['Fare'] > 7.91) & (titanic_data['Fare'] <= 14.454), 'Fare'] = 1
# titanic_data.loc[(titanic_data['Fare'] > 14.454) & (titanic_data['Fare'] <= 31), 'Fare'] = 2
# titanic_data.loc[titanic_data['Fare'] > 31, 'Fare'] = 3
# titanic_data['Fare'] = titanic_data['Fare'].astype(int)

# # Mapping Age
# titanic_data.loc[titanic_data['Age'] <= 16, 'Age'] = 0
# titanic_data.loc[(titanic_data['Age'] > 16) & (titanic_data['Age'] <= 32), 'Age'] = 1
# titanic_data.loc[(titanic_data['Age'] > 32) & (titanic_data['Age'] <= 48), 'Age'] = 2
# titanic_data.loc[(titanic_data['Age'] > 48) & (titanic_data['Age'] <= 64), 'Age'] = 3
# titanic_data.loc[titanic_data['Age'] > 64, 'Age'] = 4

# drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp']
# titanic_data = titanic_data.drop(drop_elements, axis=1)

# # Split data into training (80%), validation (20%), and testing (remaining)
# train, remaining = train_test_split(titanic_data, test_size=0.2, random_state=42)
# val, test = train_test_split(remaining, test_size=0.5, random_state=42)

# X_train = train.drop(['Survived'], axis=1)
# y_train = train["Survived"]
# X_val = val.drop(['Survived'], axis=1)
# y_val = val["Survived"]
# X_test = test.drop(['Survived'], axis=1)
# y_test = test["Survived"]

# print('AdaBoost Model')

# # Taking input from the user for model parameters
# criterion = input("Enter the criterion (misclassification, gini, entropy): ").strip().lower()
# max_depth = int(input("Enter the maximum depth of the decision tree: "))
# min_sample_split = int(input("Enter the minimum number of samples required to split: "))
# min_samples_leaf = int(input("Enter the minimum number of samples required for a leaf node: "))
# n_estimators = int(input("Enter the number of estimators: "))
# learning_rate = float(input("Enter the learning rate: "))

# # Training the model and making predictions
# base_learner = DecisionTree(criterion, max_depth, min_sample_split, min_samples_leaf)
# adaboost_model = Adaboost(base_learner, n_estimators, learning_rate)
# adaboost_model.fit(X_train, y_train)
# y_pred_val = adaboost_model.predict(X_val)
# accuracy_val = accuracy_score(y_val, y_pred_val)
# print("Validation Accuracy: {:.2f}%".format(accuracy_val * 100))
# y_pred_test = adaboost_model.predict(X_test)

# # Accuracy of the model on the test set
# accuracy_test = accuracy_score(y_test, y_pred_test)
# print("Test Accuracy: {:.2f}%".format(accuracy_test * 100))


import pandas as pd
import re
import numpy as np
from DecisionTree import *
from AdaBoost import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Loading the titanic dataset
titanic_data = pd.read_csv('train.csv')

# Feature that tells whether a passenger had a cabin on the Titanic
titanic_data['Has_Cabin'] = titanic_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Create new feature FamilySize as a combination of SibSp and Parch
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1

# Create new feature IsAlone from FamilySize
titanic_data['IsAlone'] = 0
titanic_data.loc[titanic_data['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column
titanic_data['Embarked'] = titanic_data['Embarked'].fillna('S')

# Remove all NULLS in the Fare column
titanic_data['Fare'] = titanic_data['Fare'].fillna(titanic_data['Fare'].median())

# Remove all NULLS in the Age column
age_avg = titanic_data['Age'].mean()
age_std = titanic_data['Age'].std()
age_null_count = titanic_data['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
titanic_data.loc[np.isnan(titanic_data['Age']), 'Age'] = age_null_random_list
titanic_data['Age'] = titanic_data['Age'].astype(int)

# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

titanic_data['Title'] = titanic_data['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"
titanic_data['Title'] = titanic_data['Title'].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

titanic_data['Title'] = titanic_data['Title'].replace('Mlle', 'Miss')
titanic_data['Title'] = titanic_data['Title'].replace('Ms', 'Miss')
titanic_data['Title'] = titanic_data['Title'].replace('Mme', 'Mrs')

# Mapping Sex
titanic_data['Sex'] = titanic_data['Sex'].map({'female': 0, 'male': 1}).astype(int)

# Mapping titles
title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
titanic_data['Title'] = titanic_data['Title'].map(title_mapping)
titanic_data['Title'] = titanic_data['Title'].fillna(0)

# Mapping Embarked
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# Mapping Fare
titanic_data.loc[titanic_data['Fare'] <= 7.91, 'Fare'] = 0
titanic_data.loc[(titanic_data['Fare'] > 7.91) & (titanic_data['Fare'] <= 14.454), 'Fare'] = 1
titanic_data.loc[(titanic_data['Fare'] > 14.454) & (titanic_data['Fare'] <= 31), 'Fare'] = 2
titanic_data.loc[titanic_data['Fare'] > 31, 'Fare'] = 3
titanic_data['Fare'] = titanic_data['Fare'].astype(int)

# Mapping Age
titanic_data.loc[titanic_data['Age'] <= 16, 'Age'] = 0
titanic_data.loc[(titanic_data['Age'] > 16) & (titanic_data['Age'] <= 32), 'Age'] = 1
titanic_data.loc[(titanic_data['Age'] > 32) & (titanic_data['Age'] <= 48), 'Age'] = 2
titanic_data.loc[(titanic_data['Age'] > 48) & (titanic_data['Age'] <= 64), 'Age'] = 3
titanic_data.loc[titanic_data['Age'] > 64, 'Age'] = 4

drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp']
titanic_data = titanic_data.drop(drop_elements, axis=1)

# Split data into training (80%), validation (20%), and testing (remaining) with a random seed
train, remaining = train_test_split(titanic_data, test_size=0.2, random_state=42)
val, test = train_test_split(remaining, test_size=0.5, random_state=43)  # Change the random seed for a different split

X_train = train.drop(['Survived'], axis=1)
y_train = train["Survived"]
X_val = val.drop(['Survived'], axis=1)
y_val = val["Survived"]
X_test = test.drop(['Survived'], axis=1)
y_test = test["Survived"]

print('AdaBoost Model')

# Taking input from the user for model parameters
criterion = input("Enter the criterion (misclassification, gini, entropy): ").strip().lower()
max_depth = int(input("Enter the maximum depth of the decision tree: "))
min_sample_split = int(input("Enter the minimum number of samples required to split: "))
min_samples_leaf = int(input("Enter the minimum number of samples required for a leaf node: "))
n_estimators = int(input("Enter the number of estimators: "))
learning_rate = float(input("Enter the learning rate: "))

# Training the model and making predictions
base_learner = DecisionTree(criterion, max_depth, min_sample_split, min_samples_leaf)
adaboost_model = Adaboost(base_learner, n_estimators, learning_rate)
adaboost_model.fit(X_train, y_train)
y_pred_val = adaboost_model.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy: {:.2f}%".format(accuracy_val * 100))
y_pred_test = adaboost_model.predict(X_test)

# Accuracy of the model on the test set
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Test Accuracy: {:.2f}%".format(accuracy_test * 100))
