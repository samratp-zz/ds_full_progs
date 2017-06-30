# Importing the required libraries
import pandas as pd
import numpy as np
import os

# Read the CSV Files for both Test and Train
train_data_orig = pd.read_csv('train.csv')
test_data_orig = pd.read_csv('test.csv')

# Drop the Survived column from the Test data and combine the Test and Train data to get better insights
combined_train_test_orig = train_data_orig.drop('Survived',1).append(test_data_orig)

# Analyze the Train Data
train_data_orig.shape
train_data_orig.info()


# Identify the missing values for 'Age' feature
def find_age_missing_values(all_data, age, algo_type):
    if algo_type == 0:
        #Mean of all data - Train + Test
        age_df = pd.DataFrame(age)
        age_updated = age_df
        if (age_df.isnull().values.any()):
            age_updated.Age[age_updated['Age'].isnull()] = all_data.Age.mean()
            #print(age_updated.info())
            return age_updated
        else:
            return age_df
    elif algo_type == 1:
        #Grouped Mean of all data - Train + Test based on the 'Name' column's Title (Mr, Mrs, Miss, Dr, Master)
        #title_mapping_tup = ('Mr', 'Mrs', 'Miss', 'Dr', 'Master')
        age_df = pd.DataFrame(age)
        age_updated = age_df
        if(age_df.isnull().values.any()):
            #age_updated.Age[age_updated['Age'].isnull()] = all_data['Name'].str.contains("Mr|Mrs|Miss|Dr|Master", case = False).groupby(all_data['Name'].str.contains("Mr|Mrs|Miss|Dr|Master", case = False)).mean()
            age_updated.Age[age_updated['Age'].isnull()] = all_data.Age.groupby(all_data['Name'].str.contains("Mr|Mrs|Miss|Dr|Master", case = False)).mean()
            

# Identify the missing values for 'Embarked' feature
def find_emb_missing_values(all_data, emb):
    emb_df = pd.DataFrame(emb)
    emb_updated = emb_df
    if(emb_df.isnull().values.any()):
        emb_updated.Embarked[emb_updated['Embarked'].isnull()] = all_data.Embarked.mode()[0]
        #emb_updated.ix[emb_updated.Embarked.isnull(), 'Embarked'] = all_data.Embarked.mode()[0]
        #emb_updated.update([emb_updated.Embarked.isnull(),all_data.Embarked.mode()])[0]
        #print(emb_updated.info())
        return emb_updated
    else:
        return emb_df

# Identify the missing values for 'Age' feature
def find_fare_missing_values(all_data, fare):
    fare_df = pd.DataFrame(fare)
    fare_updated = fare_df
    if (fare_df.isnull().values.any()):
        fare_updated.Fare[fare_updated['Fare'].isnull()] = all_data.Fare.mean()
        #print(fare_updated.info())
        return fare_updated
    else:
        return fare_df

# Build the Train data using the required columns
train_data = train_data_orig['Survived']
train_data = pd.concat([train_data, train_data_orig['Pclass']], axis=1)
train_data = pd.concat([train_data, train_data_orig['Name']], axis=1)
train_data = pd.concat([train_data, train_data_orig['Sex']], axis=1)

# Before appending Age, call "find_age_missing_values" to fill the NaN values
age_updated = find_age_missing_values(combined_train_test_orig, train_data_orig['Age'], 1)
train_data = pd.concat([train_data, age_updated['Age']], axis=1)

train_data = pd.concat([train_data, train_data_orig['SibSp']], axis=1)
train_data = pd.concat([train_data, train_data_orig['Parch']], axis=1)
#train_data = pd.concat([train_data, train_data_orig['Ticket']], axis=1)
train_data = pd.concat([train_data, train_data_orig['Fare']], axis=1)

# Before appending Embarked, call "find_emb_missing_values" to fill the NaN values
#combined_train_test_orig.replace(['S','Q','C'], [1,2,3], inplace='True')
#train_data_orig.replace(['S','Q','C'], [1,2,3],  inplace='True')
embarked_updated = find_emb_missing_values(combined_train_test_orig, train_data_orig['Embarked'])
train_data = pd.concat([train_data, embarked_updated['Embarked']], axis=1)

train_data.shape
train_data.info()

# Build the Test data using the required columns
test_data = test_data_orig['Pclass']
test_data = pd.concat([test_data, test_data_orig['Name']], axis=1)
test_data = pd.concat([test_data, test_data_orig['Sex']], axis=1)

# Before appending Age, call "find_age_missing_values" to fill the NaN values
age_updated = find_age_missing_values(combined_train_test_orig, test_data_orig['Age'], 1)
test_data = pd.concat([test_data, age_updated['Age']], axis=1)

test_data = pd.concat([test_data, test_data_orig['SibSp']], axis=1)
test_data = pd.concat([test_data, test_data_orig['Parch']], axis=1)
#test_data = pd.concat([test_data, test_data_orig['Ticket']], axis=1)

# Before appending Fare, call "find_emb_missing_values" to fill the NaN values
fare_updated = find_fare_missing_values(combined_train_test_orig, test_data_orig['Fare'])
test_data = pd.concat([test_data, fare_updated['Fare']], axis=1)

test_data = pd.concat([test_data, test_data_orig['Embarked']], axis=1)

test_data.shape
test_data.info()

combined_train_test = train_data.drop('Survived',1).append(test_data)

combined_train_test.shape
combined_train_test.info()
