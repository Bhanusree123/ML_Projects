#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


netflix_data = pd.read_csv('C:\\Users\\Hp\\Desktop\\Python Assignments\\movies.csv', header = None, names =['Cust_Id', 'Rating'], usecols =[0,1])
netflix_data.head()


# In[7]:


netflix_data


# In[8]:


netflix_data.dtypes


# In[9]:


netflix_data.shape


# In[11]:


movie_count1 = netflix_data.isnull().sum()
movie_count1


# In[12]:


# To calculate how many customers we are having in dataset
customer_count1 = netflix_data['Cust_Id'].nunique()


# In[ ]:


customer_count1


# In[16]:


customer_count2 = netflix_data['Cust_Id'].nunique() - movie_count1
customer_count2


# In[17]:


# get the total number of ratings
rating_count = netflix_data['Cust_Id'].count() - movie_count1
rating_count


# In[14]:


# to find out how many peoples have given 1,2,3,4,5 ratings to the movies
stars = netflix_data.groupby('Rating')['Rating'].agg(['count'])
stars


# In[ ]:


ax = stars.plot(kind = 'barh', legend=False, figsize=(15,10))
plt.title(f'Total pool: {movie_count} Movies, {customer_count} Customers, {rating_count} given ratings', fontsize=15)
plt.grid(True)


# In[ ]:


# add another column that will have movie_id
# first of all we will be calculating how many null values we are having in rating columns
df_nan = pd.DataFrame(pd.isnull(netflix_data.Rating))


# In[ ]:


df_nan


# In[ ]:


df_nan = df_nan[df_nan['Rating'] == True]


# In[ ]:


df_nan


# In[ ]:


df_nan.shape


# In[ ]:


df_nan.head()


# In[ ]:


df_nan.tail()


# In[ ]:


df_nan = df_nan.reset_index()


# In[ ]:


df_nan


# In[ ]:


df_nan.iloc[-1,0]


# In[ ]:


24057834-24053764-1


# In[ ]:


#now we will create a numpy array that will contain 1 from values 0 to 547, 2 from 549 to 693 and so on
movie_np = []
movie_id = 1

for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
  temp = np.full((1, i-j-1), movie_id)
  movie_np = np.append(movie_np, temp)
  movie_id += 1

#account for last record and corresponding length
#numpy approach
last_record = np.full((1, len(netflix_data)- df_nan.iloc[-1, 0-1]), movie_id)
movie_np=np.append(movie_np, last_record)


# In[ ]:


len(netflix_data)


# In[ ]:


netflix_data


# In[ ]:


df_nan['index'][1:]


# In[ ]:


df_nan['index'][:-1]


# In[ ]:


x = zip(df_nan['index'][1:], df_nan['index'][:-1])


# In[ ]:


tuple(x)


# In[ ]:


temp = np.full((1, 547), 1)


# In[ ]:


print(temp)


# In[ ]:


netflix_dataset = netflix_data[pd.notnull(netflix_data['Rating'])]
netflix_dataset['Movie_Id']= pd.Series(movie_np.astype(int))
netflix_dataset['Cust_Id']=netflix_dataset['Cust_Id'].astype(int)
print(netflix_dataset)


# In[ ]:


netflix_dataset.head()


# In[ ]:


netflix_dataset.tail()


# In[ ]:


#now we will remove all the users that have rated less movies and
#also all those movies that has been rated less in numbers
f = ['count', 'mean']


# In[ ]:


dataset_movie_summary = netflix_dataset.groupby('Movie_Id').agg(f)


# In[ ]:


dataset_movie_summary


# In[ ]:


dataset_movie_summary = netflix_dataset.groupby('Movie_Id')['Rating'].agg(f)


# In[ ]:


dataset_movie_summary


# In[ ]:


#now we will store all the movie_id indexes in a variable dataset_movie_summary.index and convert the datatype to int
# dataset_movie_summary.index=dataset_movie_summary.index.map(int)


# In[ ]:


movie_benchmark = round(dataset_movie_summary['count'].quantile(0.7), 0)


# In[ ]:


movie_benchmark


# In[ ]:


dataset_movie_summary['count']


# In[ ]:


drop_movie_list = dataset_movie_summary[dataset_movie_summary['count']<movie_benchmark].index


# In[ ]:


drop_movie_list


# In[ ]:


dataset_cust_summary = netflix_dataset.groupby('Cust_Id')['Rating'].agg(f)
dataset_cust_summary


# In[ ]:


cust_benchmark = round(dataset_cust_summary['count'].quantile(0.7), 0)


# In[ ]:


cust_benchmark


# In[ ]:


drop_cust_list = dataset_cust_summary[dataset_cust_summary['count']<cust_benchmark].index


# In[ ]:


drop_cust_list


# In[ ]:


print("The original dataset has:", netflix_dataset.shape)


# In[ ]:


netflix_dataset = netflix_dataset[~netflix_dataset['Movie_Id'].isin(drop_movie_list)]
netflix_dataset = netflix_dataset[~netflix_dataset['Cust_Id'].isin(drop_cust_list)]
print('After the triming, the shape is: {}'.format(netflix_dataset.shape))


# In[ ]:


#now we will prepare the dataset for SVD and it takes the matrix as the input
# so for input, we will convert the dataset into sparse matrix
#4499 movies
df_p = pd.pivot_table(netflix_dataset, values='Rating', index = 'Cust_Id', columns = 'Movie_Id')
print(df_p.shape)


# In[ ]:


df_p


# In[ ]:


import pandas as pd


# In[ ]:


df_title = pd.read_csv("C:\\Users\\Hp\\Desktop\\Python Assignments\\movies.csv", encoding='ISO-8859-1', header=None, usecols=[0,1,2], names =['Movie_Id', 'Year', 'Name'])

df_title.set_index('Movie_Id', inplace=True)


# In[ ]:


df_title.head()


# In[ ]:


get_ipython().system('pip install scikit-surprise')


# In[ ]:


# model building
import math
import seaborn as sns
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate


# In[ ]:


reader = Reader()


# In[ ]:


data = Dataset.load_from_df(netflix_dataset[['Cust_Id', 'Movie_Id', 'Rating']][:100000], reader)


# In[ ]:


svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3)


# In[ ]:


netflix_dataset.head()


# In[ ]:


#so first we take user 1283204 and we try to recommend some movies based on the past data
#He rated so many movies with 3 *
dataset_1283204 = netflix_dataset[(netflix_dataset['Cust_Id'] == 1283204) & (netflix_dataset['Rating'] == 3)]
dataset_1283204


# In[ ]:


dataset_712664=netflix_dataset[(netflix_dataset['Cust_Id'] ==712664)& (netflix_dataset['Rating']==5)]
dataset_712664


# In[ ]:


df_title


# In[ ]:


#now we will build the recommendation algorithm
#first we will make a shallow copy of the movie_titles.csv file so that we can change
#the values in the copied dataset, not in the actual dataset

user_712664 = df_title.copy()
user_712664


# In[ ]:


user_712664 = user_712664.reset_index()
user_712664


# In[ ]:


user_712664 = user_712664[~user_712664['Movie_Id'].isin(drop_movie_list)]
user_712664


# In[ ]:


user_712664['Estimate Score'] = user_712664['Movie_Id'].apply(lambda x: svd.predict(712664, x).est)
user_712664 = user_712664.drop('Movie_Id', axis=1)


# In[ ]:


user_712664=user_712664.sort_values('Estimate Score', ascending=False)
print(user_712664.head(10))

