import pandas as pd
import re
import numpy as np
from DecisionTree import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Loading the titanic dataset
train = pd.read_csv('train.csv')
full_data = [train]

# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Create new feature FamilySize as a combination of SibSp and Parch
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
# Create new feature IsAlone from FamilySize
train['IsAlone'] = 0
train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column
train['Embarked'] = train['Embarked'].fillna('S')
# Remove all NULLS in the Fare column
train['Fare'] = train['Fare'].fillna(train['Fare'].median())

# Remove all NULLS in the Age column
age_avg = train['Age'].mean()
age_std = train['Age'].std()
age_null_count = train['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
train.loc[np.isnan(train['Age']), 'Age'] = age_null_random_list
train['Age'] = train['Age'].astype(int)

# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

train['Title'] = train['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"
train['Title'] = train['Title'].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')

# Mapping Sex
train['Sex'] = train['Sex'].map({'female': 0, 'male': 1}).astype(int)

# Mapping titles
title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
train['Title'] = train['Title'].map(title_mapping)
train['Title'] = train['Title'].fillna(0)

# Mapping Embarked
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# Mapping Fare
train.loc[train['Fare'] <= 7.91, 'Fare'] = 0
train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare'] = 2
train.loc[train['Fare'] > 31, 'Fare'] = 3
train['Fare'] = train['Fare'].astype(int)

# Mapping Age
train.loc[train['Age'] <= 16, 'Age'] = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[train['Age'] > 64, 'Age'] = 4

# Drop unnecessary columns
drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis=1)

# Splitting the data into training, validation, and testing sets
X = train.drop(['Survived'], axis=1)
y = train["Survived"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print('Decision Tree Model')

# Taking input from the user for model parameters
criterion = input("Enter the criterion (misclassification, gini, entropy): ").strip().lower()
max_depth = int(input("Enter the maximum depth of the decision tree: "))
min_sample_split = int(input("Enter the minimum number of samples required to split: "))
min_samples_leaf = int(input("Enter the minimum number of samples required for a leaf node: "))

# Training the model and making predictions
decision_tree = DecisionTree(criterion, max_depth, min_sample_split, min_samples_leaf)
decision_tree.fit(X_train, y_train)
y_pred_val = decision_tree.predict(X_val)
validation_accuracy = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy: {:.2f}%".format(validation_accuracy * 100))

y_pred_test = decision_tree.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

# Visualization of the decision tree
show_decision_tree = input("Do you want to see the decision tree? (yes/no): ").strip().lower()
if show_decision_tree == 'yes':
    decision_tree.print_tree()
