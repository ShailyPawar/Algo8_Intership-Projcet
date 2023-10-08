import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('bank.csv')  # reading the data file
print(df)
print(df.head())
print(df.shape)


#Find Missing Values
#Find missing values
features_na = [features for features in df.columns if df[features].isnull().sum() > 0]
for feature in features_na:
    print(feature, np.round(df[feature].isnull().mean(), 4), '% missing values')
else:
    print("No missing values found")

# Find features with one value
for column in df.columns:
    print(column, df[column].nunique)

# explore the categorial features
categorical_features = [feature for feature in df.columns if
                        ((df[feature].dtypes == '0') & (feature not in ['deposit']))]
categorical_features

for feature in categorical_features:
    print('The feature is () and number of categories are {}'.format(feature, len(df[feature].unique())))

# find categorical feature distribution
plt.figure(figsize=(15, 80), facecolor='white')
plotnumber = 1
for categorical_feature in categorical_features:
    ax - plt.subplot(12, 3, plotnumber)
    sns.countplot(y=categorical_feature, data=df)
    plt.xlabel(categorical_feature)
    plt.title(categorical_feature)
    plotnumber += 1
plt.show()

#Relationship between categorical features and label
#check target label split over categorical features
#find out the relationship between categorical varibale and dependent variable
for categorical_feature in categorical_features:
    sns.catplot(x='deposit', col=categorical_feature, kind='count',data= df)
plt.show()
#check target label split over categorical features and find the count
for categorical_features in categorical_features:
    print(df.groupby(['deposit',categorical_feature]).size())

#Explore the numerical features
#list of numerical variable
numerical_features = [feature for feature in df.columns if((df[feature].dtypes !='0') & (feature not in ['deposit']))]
print('Number of numerical variables: ',len(numerical_features))
#visualise the numerical variable
df[numerical_features].head()

#find discrete numerical features
discrete_feature = [feature for feature in numerical_features if len(df[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))

#Relation between Discrete numerical features and labels
#Find Continuous Numerical Features
continuous_features=[feature for feature in numerical_features if feature not in discrete_feature+['deposit']]
print("Continuous feature Count {}".format(len(continuous_features)))

#Distribution of Continuous Numerical Features
#plot a univariable distribution of contionuous observations
plt.figure(figsize=(20,60),facecolor='white')
plotnumber =1
for continuous_feature in continuous_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.distplot(df[continuous_feature])
    plt.xlabel(continuous_feature)
    plotnumber+=1
plt.show()

#Relationship between continuous numerical features and labels
#boxplot to show target distribtuion with respect numerical features
plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for feature in continuous_features:
    ax = plt.subplot(12, 3, plotnumber)
    sns.boxplot(x="deposit",y=df[feature],data=df)
    plt.xlabel(feature)
    plotnumber += 1
plt.show()

#find outliers in numerical features
#boxplot on numerical features to find outliers
plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for numerical_feature in continuous_features:
    ax = plt.subplot(12, 3, plotnumber)
    sns.boxplot(df[numerical_feature])
    plt.xlabel(numerical_feature)
    plotnumber += 1
plt.show()

#Explore the correlation between numerical features
## Checkin for correlation
cor_mat=df.corr()
fig = plt.figure(figsize=(15,7))
sns.heatmap(cor_mat,annot=True)

#Check the Data set is balanced or not based on target values in classifiaction
#total patient count based on cardio_results
sns.countplot(x='deposit',data=df)
plt.show()
df['deposit'].groupby(df['deposit']).count()
