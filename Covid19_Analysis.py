#!/usr/bin/env python
# coding: utf-8

# # Analyzing the trends of COVID-19 with Python

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("C:\\Users\\Hp\\Desktop\\Python Assignments\\covid_19_analysis_data.csv")


# In[3]:


df.head()


# In[4]:


df.rename(columns = {
    "Province/State" : "state",
    "Country/Region" : "country",
    "Lat" : "lat",
    "Long" : "long",
    "Confirmed" : "confirmed",
    "Deaths" : "deaths",
    "Recovered" : "recovered",
    "Date" : "date"
}, inplace = True)


# In[5]:


df.head()


# In[8]:


df["active"] = df["confirmed"] - df["deaths"] - df["recovered"]
df


# In[9]:


top = df[df['date'] == df['date'].max()]
top


# In[10]:


world = top.groupby("country")["active", "confirmed", "deaths"].sum().reset_index()
world


# In[11]:


top["country"].value_counts()


# In[12]:


world["active"].max()


# In[13]:


figure = px.choropleth(world, locations="country",locationmode="country names", 
                       color="active", color_continuous_scale="reds", hover_name="country",
                       range_color=[1,100000],title="Contries with all active cases")
figure.show()


# In[14]:


# Trend of how covid spreaded
plt.figure(figsize=(15, 10))
plt.xlabel("Dates", fontsize=10)
plt.ylabel("Total Cases", fontsize=10)
plt.xticks(rotation=90,fontsize=5)
plt.yticks(fontsize=10, rotation=90)
plt.title("worldwide Confirmed Cases Over Time", fontsize=25)
total_cases = df.groupby("date")["date", "confirmed"].sum().reset_index()
total_cases["date"] = pd.to_datetime(total_cases["date"])

ax = sns.pointplot(x = total_cases.date, y = total_cases.confirmed, color="b")
ax.set(xlabel="Dates", ylabel="Total Cases")


# In[30]:


top


# In[31]:


top_actives = top.groupby("country")["active"].sum().sort_values(ascending=False).head(20).reset_index()
top_actives


# In[32]:


plt.figure(figsize=(15, 10))
plt.xlabel("Total Active Case", fontsize=10)
plt.ylabel("Country", fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("Top 20 Total Active Cases", fontsize=25)

ax = sns.barplot(x=top_actives.active, y=top_actives.country)

for i, (value, name) in enumerate(zip(top_actives.active, top_actives.country)):
    ax.text(value, i-.05, f'{value:,.0f}', size = 20, ha="left", va="center")
    
ax.set(xlabel="Total Active Cases", ylabel="Country")


# In[33]:


# Top Deaths Cases
top_deaths = top.groupby("country")["deaths"].sum().sort_values(ascending=False).head(20).reset_index()
top_deaths


# In[34]:


plt.figure(figsize=(15, 10))
plt.xlabel("Total Death Case", fontsize=10)
plt.ylabel("Country", fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("Top 20 Total Death Cases", fontsize=25)

ax = sns.barplot(x=top_deaths.deaths, y=top_deaths.country)

for i, (value, name) in enumerate(zip(top_deaths.deaths, top_deaths.country)):
    ax.text(value, i-.05, f'{value:,.0f}', size = 20, ha="left", va="center")
    
ax.set(xlabel="Total Death Cases", ylabel="Country")


# In[35]:


# Recoverd Cases
top_recovered = top.groupby("country")["recovered"].sum().sort_values(ascending=False).head(20).reset_index()
top_recovered


# In[36]:


plt.figure(figsize=(15, 10))
plt.xlabel("Total Recovered Case", fontsize=10)
plt.ylabel("Country", fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("Top 20 Total Recovered Cases", fontsize=25)

ax = sns.barplot(x=top_recovered.recovered, y=top_recovered.country)

for i, (value, name) in enumerate(zip(top_recovered.recovered, top_recovered.country)):
    ax.text(value, i-.05, f'{value:,.0f}', size = 20, ha="left", va="center")
    
ax.set(xlabel="Total Recovered Cases", ylabel="Country")


# In[38]:


brazil = df[df.country == "Brazil"]
brazil = brazil.groupby("date")["recovered", "deaths", "confirmed", "active"].sum().reset_index()
brazil


# In[40]:


us = df[df.country == "US"]
us = us.groupby("date")["recovered", "deaths", "confirmed", "active"].sum().reset_index()
us


# In[41]:


uk = df[df.country == "United Kingdom"]
uk = uk.groupby("date")["recovered", "deaths", "confirmed", "active"].sum().reset_index()
uk


# In[42]:


india = df[df.country == "India"]
india = india.groupby("date")["recovered", "deaths", "confirmed", "active"].sum().reset_index()
india


# In[43]:


russia = df[df.country == "Russia"]
russia = russia.groupby("date")["recovered", "deaths", "confirmed", "active"].sum().reset_index()
russia


# In[45]:


plt.figure(figsize=(15, 10))
sns.pointplot(brazil.index, brazil.confirmed, color="Blue")
sns.pointplot(us.index, us.confirmed, color="Pink")
sns.pointplot(uk.index, uk.confirmed, color="Green")
sns.pointplot(india.index, india.confirmed, color="Black")
sns.pointplot(russia.index, russia.confirmed, color="Red")
plt.xlabel("No. Of Days", fontsize=2)
plt.ylabel("Confirmed Cases", fontsize=10)
plt.title("Confirmed Cases Over Time", fontsize=25)

plt.show()


# In[46]:


plt.figure(figsize=(15, 10))
sns.pointplot(brazil.index, brazil.deaths, color="Blue")
sns.pointplot(us.index, us.deaths, color="Pink")
sns.pointplot(uk.index, uk.deaths, color="Green")
sns.pointplot(india.index, india.deaths, color="Black")
sns.pointplot(russia.index, russia.deaths, color="Red")
plt.xlabel("No. Of Days", fontsize=2)
plt.ylabel("Death Cases", fontsize=10)
plt.title("Death Cases Over Time", fontsize=25)

plt.show()


# In[47]:


plt.figure(figsize=(15, 10))
sns.pointplot(brazil.index, brazil.recovered, color="Blue")
sns.pointplot(us.index, us.recovered, color="Pink")
sns.pointplot(uk.index, uk.recovered, color="Green")
sns.pointplot(india.index, india.recovered, color="Black")
sns.pointplot(russia.index, russia.recovered, color="Red")
plt.xlabel("No. Of Days", fontsize=2)
plt.ylabel("Recovered Cases", fontsize=10)
plt.title("Recovered Cases Over Time", fontsize=25)

plt.show()


# In[48]:


get_ipython().system(' pip install prophet')


# In[50]:


from prophet import Prophet


# In[51]:


import os
os


# In[52]:


data = pd.read_csv("C:\\Users\\Hp\\Desktop\\Python Assignments\\covid_19_analysis_data.csv", parse_dates=["Date"])


# In[53]:


data.head()


# In[54]:


total_active = data["Active"].sum()


# In[55]:


total_active


# In[56]:


data["Active"] = data["Confirmed"] - data["Deaths"] - data["Recovered"]


# In[57]:


total_active = data["Active"].sum()
total_active


# In[61]:


confirmed = data.groupby("Date").sum()["Confirmed"].reset_index()
deaths=data.groupby('Date').sum()['Deaths'].reset_index()
recovered=data.groupby('Date').sum()['Recovered'].reset_index()
confirmed.head()


# In[62]:


deaths.head()


# In[63]:


# For building a forecasting model using fbProphet library,
# there should be only 2 columns passed
# The column names should always be --> 'ds','y'

confirmed.columns = ["ds", "y"]
confirmed["ds"] = pd.to_datetime(confirmed["ds"])


# In[64]:


confirmed


# In[65]:


m = Prophet(interval_width=0.95)
m.fit(confirmed)

future = m.make_future_dataframe(periods=7)
future.tail(7)


# In[66]:


forecast = m.predict(future)
forecast.tail(7)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7)


# In[67]:


confirmed_forecast_plot = m.plot(forecast)


# In[68]:


confirmed_forecast_plot1 = m.plot_components(forecast)


# In[69]:


recovered=data.groupby('Date').sum()['Recovered'].reset_index()
recovered.head()


# In[70]:


recovered.columns = ["ds", "y"]
recovered["ds"] = pd.to_datetime(recovered["ds"])


# In[71]:


m = Prophet(interval_width=0.95)
m.fit(recovered)

future = m.make_future_dataframe(periods=7)
future.tail(7)


# In[72]:


forecast = m.predict(future)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7)


# In[73]:


recovered_forecast_plot = m.plot(forecast)


# In[74]:


recovered_forecast_plot1 = m.plot_components(forecast)


# In[75]:


deaths=data.groupby('Date').sum()['Deaths'].reset_index()
deaths


# In[76]:


deaths.columns = ["ds", "y"]
deaths["ds"] = pd.to_datetime(deaths["ds"])


# In[77]:


deaths


# In[78]:


m = Prophet(interval_width=0.95)

m.fit(deaths)

future = m.make_future_dataframe(periods=7)
future.tail(7)


# In[79]:


forecast = m.predict(future)


# In[80]:


forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7)


# In[81]:


deaths_forecast_plot = m.plot(forecast)


# In[82]:


deaths_forecast_plot1 = m.plot_components(forecast)

