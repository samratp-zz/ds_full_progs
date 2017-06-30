import os
import pandas as pd
import seaborn as sns
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import model_selection

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("c:/__backup/kaggle/Titanic/")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

#explore missing data
titanic_train.apply(lambda x : sum(x.isnull()))

#pre-process Embarked
titanic_train.Embarked[titanic_train['Embarked'].isnull()] = 'S'

#pre-process Age
sns.factorplot(x="Age", row="Survived", data=titanic_train, kind="box", size=6)

imputer = preprocessing.Imputer()
titanic_train[['Age']] = imputer.fit_transform(titanic_train[['Age']])
sns.factorplot(x="Age", row="Survived", data=titanic_train, kind="box", size=6)
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.distplot, "Age").add_legend()

#create family size feature
def size_to_type(x):
    if(x == 1): 
        return 'Single'
    elif(x >= 2 and x <= 4): 
        return 'Small'
    else: 
        return 'Large'
    
titanic_train['FamilySize'] = titanic_train.SibSp + titanic_train.Parch + 1
titanic_train['FamilyType'] = titanic_train['FamilySize'].map(size_to_type)
sns.distplot(titanic_train['FamilySize'])
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.distplot, "FamilySize").add_legend()
sns.factorplot(x="Survived", hue="FamilyType", data=titanic_train, kind="count", size=6)

#process names of passengers
title_Dictionary = {
                        "Capt":       "Officer", "Col":        "Officer",
                        "Major":      "Officer", "Jonkheer":   "Royalty",
                        "Don":        "Royalty", "Sir" :       "Royalty",
                        "Dr":         "Officer", "Rev":        "Officer",
                        "the Countess":"Royalty","Dona":       "Royalty",
                        "Mme":        "Mrs", "Mlle":       "Miss",
                        "Ms":         "Mrs", "Mr" :        "Mr",
                        "Mrs" :       "Mrs", "Miss" :      "Miss",
                        "Master" :    "Master", "Lady" :      "Royalty"
}

def extract_title(name):
    return name.split(',')[1].split('.')[0].strip()

titanic_train['Title'] = titanic_train['Name'].map(extract_title)
titanic_train['Title'] = titanic_train['Title'].map(title_Dictionary)
    
pd.crosstab(index=titanic_train["Title"], columns="count")   
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Title'])
sns.factorplot(x="Survived", hue="Title", data=titanic_train, kind="count", size=6)

#process ticket feature
def extract_id(ticket):        
        id = ticket.replace('.','').replace('/','').split()[0]
        if not id.isdigit() and len(id) > 0:
            return id.upper()
        else: 
            return 'X'

titanic_train['TicketId'] = titanic_train['Ticket'].map(extract_id)

pd.crosstab(index=titanic_train["TicketId"], columns="count")   
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['TicketId'])
sns.factorplot(x="Survived", hue="TicketId", data=titanic_train, kind="count", size=6)

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'FamilyType', 'Embarked', 'Sex','Title','TicketId'])
titanic_train1.drop(['PassengerId','Name','Ticket','Cabin','Survived'], axis=1, inplace=True)
# Hack ???
titanic_train1.drop(['TicketId_AS',
 'TicketId_CASOTON',
 'TicketId_FA',
 'TicketId_LINE',
 'TicketId_PPP',
 'TicketId_SCOW',
 'TicketId_SOP',
 'TicketId_SP',
 'TicketId_SWPP'], axis = 1, inplace = True)

titanic_train1.info()

X_train = titanic_train1
y_train = titanic_train['Survived']

gbm_est = ensemble.GradientBoostingClassifier()

param_grid = {'n_estimators':[500], 'learning_rate':[0.1], 'max_depth':[3]}
gbm_grid = model_selection.GridSearchCV(gbm_est, param_grid, n_jobs = 15, cv = 10, verbose = 1)
gbm_grid.fit(X_train, y_train)
gbm_grid.grid_scores_
gbm_grid.best_score_
gbm_grid.score(X_train, y_train)

feature_imp_df = pd.DataFrame({'features': list(titanic_train1), 'importance': gbm_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending = False)
sns.barplot(x='importance', y = 'features', data = feature_imp_df)


titanic_test = pd.read_csv("test.csv")
titanic_test.shape
titanic_test.info()
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

#pre-process Age
imputer = preprocessing.Imputer()
titanic_test[['Age']] = imputer.fit_transform(titanic_test[['Age']])

titanic_test['FamilySize'] = titanic_test.SibSp + titanic_train.Parch + 1
titanic_test['FamilyType'] = titanic_test['FamilySize'].map(size_to_type)

titanic_test['Title'] = titanic_test['Name'].map(extract_title)
titanic_test['Title'] = titanic_test['Title'].map(title_Dictionary)

titanic_test['TicketId'] = titanic_test['Ticket'].map(extract_id)

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'FamilyType', 'Embarked', 'Sex','Title','TicketId'])
titanic_test1.shape
titanic_test1.info()

titanic_test1.apply(lambda x : sum(x.isnull()))
titanic_test1.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
# Hack ???
titanic_test1.drop(['TicketId_A',
 'TicketId_AQ3',
 'TicketId_AQ4',
 'TicketId_LP',
 'TicketId_SCA3',
 'TicketId_STONOQ'], axis = 1, inplace = True)

titanic_test1.info()

X_test = titanic_test1
titanic_test['Survived'] = gbm_grid.predict(X_test)
titanic_test.to_csv("submission_feature_engg_4.csv", columns=['PassengerId','Survived'], index=False)
