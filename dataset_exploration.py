#!/usr/bin/env python
# coding: utf-8

# # DATASET EXPLORATION

# In[1]:


import numpy as np
from sklearn import model_selection
from sklearn import svm
import pandas as pd
import zipfile


# ## List all files in Dataset directory using the subprocess library

# In[2]:


from subprocess import check_output 
print(check_output(["ls" , "/home/beltus/image/Data Science/datasets"]).decode("utf8")) #List all files in dataset folder


# ## Unzip Dataset if it requires nzipping

# In[4]:


#specify file name of the file you want to unzip

Dataset = "airline_safety"
with zipfile.ZipFile("/home/beltus/image/Data Science/datasets/" + Dataset + ".zip", "r") as z: 
    z.extractall("/home/beltus/image/Data Science/datasets/") #specify directory to extract the files to 


# In[5]:


#List all file in a folder.
from subprocess import check_output
print(check_output(['ls', "/home/beltus/image/Data Science/datasets/"]).decode("utf8"))


# # Read Datasets using Pandas

# In[108]:


#load in the dataset using pandas
iris_dataset = pd.read_csv("./datasets/iris.data")

#data = pandas.read_csv("./datasets/digits.csv")
#url = "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/""#download

#dataset = pd.read_csv("./datasets/airline-safety.csv")
#dataset = pd.read_csv("iris.data" , header = None)

#dataset = pd.read_csv("./datasets/covid_19_global/train.csv")
covid_dataset = pd.read_csv('./datasets/corona-virus-report/covid_19.csv')


# ## Print the number of Observations(Samples) and the number of Features(Attributes) In your Dataset

# In[76]:


#check number of rows and colums in the dataset

rows, columns = covid_dataset.shape
print("Number of Samples or observations: " , rows)
print("Number of Attributes or Features" , columns)

#Prints the number of datapoints(rows)
print('Number of Samples: ' , len(dataset))


# #  Take a Quick Peak at the contents of your Dataset

# In[77]:


## Take a quick peak at the data of the first 10 enteries

covid_dataset.head()


# ## Display all the columns in Dataset

# In[79]:


#Just for illustration
digits_dataset = pd.read_csv("./datasets/digits.csv")

pd.set_option("display.max_columns" , None) #permits all the attributes(columns) to be displayed
digits_dataset.head(5)


# ## Display all the columns of Dataset and their Data types

# In[81]:


covid_dataset.info


# ## Describe method summarizes the Data

# In[113]:



covid_dataset.describe()


# ## Correlation

# In[83]:


dataset.corr()


# ## Delete Irrelevant Columns

# In[38]:


newdataset = dataset.drop(['Lat' , 'Long'], axis = 1)
newdataset.head()


# ## Change Column Names

# In[114]:


#list of column names 
new_col_names = ['ID' , 'Region' , 'Country' , 'Latitude', 'Longitude' , 'Dates' , 'Confirmed_Cases' , 'Fatal']

# assign the names to your dataset
covid_dataset.columns = new_col_names

#dataset.rename(columns = {'ID': 'IDENTITY'}, inplace = True)
covid_dataset.head()


# In[ ]:





# ## How to Know the Number of Samples per unique category in the dataset

# In[86]:


covid_dataset.groupby('Country').size()


# ## Plot Histogram

# In[88]:




iris_dataset.head()


# In[90]:


#histogram plot from iris dataset
dataset = pd.read_csv("./datasets/iris.data")

iris_dataset['3.5'].plot.hist()


# ## Multiple histogram plot

# In[92]:


#multiple plots
iris_dataset.plot.hist(subplots=True , layout = (2,2) , figsize=(10,10) , bins = 40)


# ## Bar Chart Plot

# In[93]:


#vertical barchart 
iris_dataset['Iris-setosa'].value_counts().sort_index().plot.bar()


# In[94]:



#horizontal bar chart
iris_dataset['Iris-setosa'].value_counts().sort_index().plot.barh()


# In[124]:


dataset = pd.read_csv('./datasets/corona-virus-report/covid_19.csv')

#wine_reviews.groupby("country").price.mean().sort_values(ascending=False)[:5].plot.bar()
dataset.rename(columns = {'Country/Region' : 'country'} , inplace=True)


#Group data by country and calculated the mean of confirm cases and sorted in ascending other. .finally plotting a bar chart
dataset.groupby('country').Confirmed.mean().sort_values(ascending=False)[:10].plot.bar()


# In[118]:


covid_dataset.head()


# ## Scatter Plot

# In[126]:


## Scatter Plot
#iris.plot.scatter(x='sepal_length', y='sepal_width', title='Iris Dataset')
#plot from iris dataset
dataset = pd.read_csv("./datasets/iris.data")  

dataset.rename(columns = {'0.2' : 'petal width' , '1.4':'petal length'} , inplace = True )

dataset.plot.scatter(x='petal width', y='petal length', title='Iris Dataset')


# 

# In[16]:





# In[ ]:





# In[ ]:




