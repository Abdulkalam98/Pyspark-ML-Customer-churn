#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize': [9,6]}, font_scale=1.3)


# In[2]:


#loading data
df = pd.read_csv('C:\\Users\\Downloads\\BDA601_Assessment 2_Telco-Customer-Churn_downloaded 05082020.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.columns.values


# In[7]:


df.dtypes


# # Removing the attributes

# In[8]:


df=df.drop(['MonthlyCharges','OnlineSecurity','StreamingTV','InternetService','Partner'],axis=1)


# Saving the dataset as csv file

# In[9]:


df.to_csv('C:\\Users\\abdulki\\Downloads\\modified dataset.csv',header=True,index=False)


# # Data Manipulation

# In[10]:


df = df.drop(['customerID'], axis = 1)
#dropping the Customer ID. Since these attributes are unique and it will not help impact the churn analysis
df.head()


# In[11]:


df["SeniorCitizen"]= df["SeniorCitizen"].map({0: "No", 1: "Yes"})


# # Handling Missing Values

# In[12]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[13]:


df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df.isnull().sum()


# There are 11 missing values in the dataset. For missing values the mean value of totalcharges are filled for emptyrows

# In[14]:


df.TotalCharges.fillna(round(df.TotalCharges.median(),2),inplace=True)


# In[15]:


df.isnull().sum()


# The Data is cleansed and ready to process

# # Data Visualization

# In[16]:


sns.countplot(x=df["gender"],hue=df["Churn"],palette='magma');


# By Gender we cant get able to find churn.

# In[17]:


sns.countplot(x=df["SeniorCitizen"],hue=df["Churn"],palette='magma');


# Here we can see proportionality between senior if he stay or leave. Most Senior Citizen want to churn

# In[18]:


sns.countplot(x=df["Contract"],hue=df["Churn"],palette='magma');


# Customers using month to month are more likely to churn. About 75% of customer with Month-to-Month Contract opted to move out as compared to 13% of customers with One Year Contract and 3% with Two Year Contract

# In[19]:


sns.countplot(x=df['TechSupport'],hue=df['Churn'],palette='magma')


# Customers dont have Techsupport are more likely to churn. Need to focus on provide Techsupport to all the customers

# In[20]:


sns.countplot(x=df['PaperlessBilling'],hue=df['Churn'],palette='magma')


# Customers using Paperlessbilling are more likely to churn

# In[21]:


labels = df['PaymentMethod'].unique()
values = df['PaymentMethod'].value_counts()
explode = (0.2, 0, 0, 0) 
fig1, ax1 = plt.subplots()
ax1.pie(values, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.show()


# Electronic Check has the highest payment option of 33%. It has high impact on Churn.

# In[22]:


sns.countplot(x=df['PaymentMethod'],hue=df['Churn'],palette='magma')


# Major customers who moved out were having Electronic Check as Payment Method.
# 
# Customers who opted for Credit-Card automatic transfer or Bank Automatic Transfer and Mailed Check as Payment Method were less likely to move out.

# In[23]:


sns.kdeplot(df[df["Churn"]=='No']["tenure"],color='blue',label='Churn: No')
sns.kdeplot(df[df["Churn"]=='Yes']["tenure"],color='red',label='Churn: Yes')
plt.legend()
plt.show()


# In[24]:


sns.boxplot(x='Churn',y='tenure',data=df)


# New customers are more likely to churn.Most customers churn in 3 to 10 months. 
# 
# Old Customers with more than 40 months want to stay with company 

# In[25]:


df['tenure_bin'] = pd.cut(df['tenure'],[-1,12,24,36,48,60,100])
df["Churn"]=df['Churn'].map({"No": 0, "Yes": 1})
df['tenure_bin'].value_counts(sort = False)


# In[26]:


plt.figure(figsize=(12,4))

ax = sns.barplot(x = "tenure_bin", y = "Churn", data = df, palette = 'rocket', ci = None)

plt.ylabel("% of Churn", fontsize= 12)
plt.ylim(0,0.6)
plt.xticks([0,1,2,3,4,5], ['12 or less', '13 to 24', '25 to 36', '37 to 48', '49 to 60', 'more than 60'], fontsize = 12)
plt.xlabel("Tenure Group (in months)", fontsize= 12)



for p in ax.patches:
    ax.annotate("%.2f" %(p.get_height()), (p.get_x()+0.25, p.get_height()+0.03),fontsize=14)

plt.show()


# Almost 50 percent of those who became a customer for a year or less ended up leaving the company. A churn rate this high in the first year indicates that the quality of the service provided fails to hold up to their new customers’ expectation.

# In[27]:


df["SeniorCitizen"]= df["SeniorCitizen"].map({"No": 0, "Yes": 1})
df = df.drop(["tenure_bin"], axis = 1)


# In[28]:


sns.heatmap(df.corr(),annot=True);


# Total Charges affects 83% on Churn Prediction. It is main attribute which is directly proportional to Churn Rate.

# # Machine Learning Model Evaluations and Predictions

# In[29]:


import findspark
findspark.init("C:\spark-3.0.2-bin-hadoop2.7")
import pyspark
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName("local").getOrCreate()


# Converting pandas dataframe to Spark dataframe

# In[31]:


df_sp = spark.createDataFrame(df)


# In[32]:


df_sp.show(5)


# In[34]:


# Import VectorAssembler and Vectors
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml import Pipeline


# # Feature Selection and Pre Processing

# Using StringIndexer changing the categorical valriables to binary values for processing of data

# In[33]:


categorical_columns=['gender', 'SeniorCitizen', 'Dependents', 'PhoneService',
       'MultipleLines', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'Churn']


# In[35]:


numerical_columns=['tenure','TotalCharges']


# In[36]:


# Categorical Features (which are all strings)
stringindexers = [StringIndexer(inputCol=column, 
                 outputCol=column+"_Index") for column in categorical_columns]


# Converting the binary values to binary vectors

# In[37]:


onehotencoder_categorical = OneHotEncoder(
    inputCols = [column + "_Index" for column in categorical_columns],
    outputCols = [column + "_Vec" for column in categorical_columns])


# Selecting the Binary Vectors

# In[38]:


categorical_columns_class_vector = [col + "_Vec" for col in categorical_columns]
categorical_columns_class_vector.remove("Churn_Vec") #Dropping Churn_Vec since it is need to be predicted
categorical_numerical_inputs = categorical_columns_class_vector + numerical_columns


# Combining all the numerical Values and Binary vectors to a Single Feature Vector

# In[39]:


# Assembler for all columns
assembler = VectorAssembler(inputCols = categorical_numerical_inputs, 
                            outputCol="features")


# Standardizing the features before applying an ML algorithm in order to avert the risk of an algorithm being insensitive to a certain features

# In[40]:


pipeline = Pipeline(
    stages=[*stringindexers,
            onehotencoder_categorical,
            assembler,
            StandardScaler(
                withStd=True,
                withMean=False,
                inputCol="features",
                outputCol="scaledFeatures")
           ])


# Applying Each Stage on the dataframe using Pipeline

# In[41]:


output_fixed=pipeline.fit(df_sp).transform(df_sp)


# Selecting only two columns. scaledFeatures is the feature column to predict the churn. churn_index is attribute need to predict

# In[42]:


final_data = output_fixed.select("scaledFeatures",'Churn_Index')


# Split the 70% of records to train data and 30% to test data

# In[43]:


train_data,test_data = final_data.randomSplit([0.7,0.3])


# ### The Classifiers

# In[44]:


from pyspark.ml.classification import DecisionTreeClassifier,GBTClassifier,RandomForestClassifier
from pyspark.ml import Pipeline


# In[45]:


dtc = DecisionTreeClassifier(labelCol='Churn_Index',featuresCol='scaledFeatures')#Decision Tree Classifier
rfc = RandomForestClassifier(labelCol='Churn_Index',featuresCol='scaledFeatures')#Random Forest Classifier
gbt = GBTClassifier(labelCol='Churn_Index',featuresCol='scaledFeatures')#GBT Classifier


# Training the Model

# In[46]:


dtc_model = dtc.fit(train_data)
rfc_model = rfc.fit(train_data)
gbt_model = gbt.fit(train_data)


# ## Model Comparison
# 
# Let's compare each of these models!

# In[47]:


dtc_predictions = dtc_model.transform(test_data)
rfc_predictions = rfc_model.transform(test_data)
gbt_predictions = gbt_model.transform(test_data)


# **Evaluation Metrics:**

# In[48]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[49]:


# Select (prediction, true label) and compute test error
acc_evaluator = MulticlassClassificationEvaluator(labelCol="Churn_Index", predictionCol="prediction", metricName="accuracy")


# In[50]:


dtc_acc = acc_evaluator.evaluate(dtc_predictions)
rfc_acc = acc_evaluator.evaluate(rfc_predictions)
gbt_acc = acc_evaluator.evaluate(gbt_predictions)


# In[51]:


print("Here are the results!")
print('-'*80)
print('A single decision tree had an accuracy of: {0:2.2f}%'.format(dtc_acc*100))
print('-'*80)
print('A random forest ensemble had an accuracy of: {0:2.2f}%'.format(rfc_acc*100))
print('-'*80)
print('A ensemble using GBT had an accuracy of: {0:2.2f}%'.format(gbt_acc*100))


# GBT Classifer gives more accuracy compared to Random Forest and Decision Tree
