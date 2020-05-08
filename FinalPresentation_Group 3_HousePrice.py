#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# Online property companies offer valuations of houses using machine learning techniques. This dataset tells us about the house sales in King County, Washington State, USA. The dataset consists of historic data of houses sold between May 2014 to May 2015. This data was published/released under CC0*: Public Domain. The data was downloaded from https://www.kaggle.com/shivachandel/kc-house-data.

# ## About the Project

# The aim of this project is to predict the price of houses based on various factors like number of bedrooms, bathrooms, conditions, area and age of the houses. These features greatly contribute to the variation in prices of the house. Online property companies offer valuations of houses using machine learning techniques. We have included the analysis of maximum price of house based on number of bedrooms, highest prices of ten houses with the oldest build year, average price of houses with each condition ranging from one to five in a range of 25 years. We have used visualizations to better demonstrate the data to detect patterns, trends, and outliers in groups of data. The outer part of the project provides you with a series of exploratory analysis to keep you aware of the variables and factors predisposing to decisions on results. A few models are executed to train our data to estimate the dependence of different characteristics on house prices. Jupyter Notebooks is a perfect way to comment and evaluate code thoroughly. For analysing and modelling, we have used PySpark.

# In[12]:


from pyspark.sql import SQLContext,Row
from pyspark import SparkConf, SparkContext


# In[83]:


sc=SparkContext()


# In[13]:


sc


# Preprocessing--Data Upload in RDD form.

# ## Data Cleaning

# In[14]:


rdd=sc.textFile(r'C:\Everything\Data\2020Spring\BigData\kc_house_data.csv')


# In[15]:


rdd=rdd.map(lambda l:l.split(","))
rdd.take(2)


# Remvoing the header so that further analysis on RDD could be performed

# In[16]:


header=rdd.first()
rdd=rdd.filter(lambda line:line != header)
rdd.take(2)


# In[17]:


from pyspark.sql import SparkSession
from pyspark import SparkContext


# In[18]:


spark = SparkSession(sc)


# Converting Rdd to dataframe

# In[19]:


from pyspark.sql import Row

df=rdd.map(lambda r1:Row(id=r1[0],
                         date=r1[1],
                         price=r1[2],
                         bedrooms=r1[3],
                         bathrooms=r1[4],
                         livingsqft=r1[5],
                         lotsqft=r1[6],
                         floor=r1[7],
                         waterfront=r1[8],
                         view=r1[9],
                         condition=r1[10],
                         grade=r1[11],
                         sqftabove=r1[12],
                         sqftbase=r1[13],
                         yr_built=r1[14],
                         yr_renovated=r1[15],
                         zipcode=r1[16],
                         lat=r1[17],
                         long=r1[18],
                         sqftliving1=r1[19],
                         sqftlot15=r1[20])).toDF()
  


# In[20]:


df.show(3)


# In this data set, we will convert the data types of columns to the appropriate one and calculate the age of the house to include the age of the house in the analysis

# In[21]:


df.toPandas().head(3)


# In[22]:


df.count()


# In[23]:


## extracting year
from pyspark.sql.functions import substring
df1=df.withColumn("year",substring(df["date"],1,4))


# In[24]:


df1.show(2)


# We will be predicting the price of the house in this analysis. To accomplish this we are including the the house price, bedrooms, bathrooms, and age of the house.

# In[25]:


### choosing relevant variables for price analysis
df3=df1.select('price','bedrooms','bathrooms','condition','yr_built', 'livingsqft','year')


# In[26]:


df3.printSchema()


# In[27]:


### converting data types for analysis
from pyspark.sql.types import IntegerType
from pyspark.sql.types import *
df3=df3.withColumn("price",df3["price"].cast(IntegerType()))
df3=df3.withColumn("bedrooms",df3["bedrooms"].cast(IntegerType()))
df3=df3.withColumn("bathrooms",df3["bathrooms"].cast(FloatType()))
df3=df3.withColumn("condition",df3["condition"].cast(IntegerType()))
df3=df3.withColumn("yr_built",df3["yr_built"].cast(IntegerType()))
df3=df3.withColumn("livingsqft",df3["livingsqft"].cast(IntegerType()))
df3=df3.withColumn("year",df3["year"].cast(IntegerType()))





# In[28]:


df3.printSchema()


# In[29]:


df3.show(3)


# In[30]:


df3=df3.withColumn("age",df3["year"]-df3["yr_built"])


# In[31]:


df_final=df3.select(["price","bedrooms","bathrooms","condition","livingsqft","age"])


# In[32]:


df_final.toPandas().head()


# ## Data Analysis

# In[33]:


#Creat table
temp_table_name = "house"

df.createOrReplaceTempView(temp_table_name)


# Analysis 1: Identify the topic that has the maximum number of price at each bedroom level.
# 
# 

# In[34]:


price=spark.sql("select bedrooms,max(price) from house group by bedrooms order by max(price) desc")
price.show()


# Analysis 2: Identify the topic that the top 10 highest price with the oldest build year.

# 

# In[35]:


old = spark.sql("select  price, min(yr_built) from house group by price order by min(yr_built) asc,price desc limit 10")
old.show()


# In[36]:


#The sale year only have two number 2014 and 2015, so we can treat the build year like age.
avgprice=spark.sql("(select '1900-1925' as YearRange,condition, avg(price) as AveragePrice from house where yr_built >=1900 and yr_built <= 1925 group by condition,'1900-1925'order by condition)union all(select '1926-1950'as YearRange,condition, avg(price) as AveragePrice from house where yr_built >=1926 and yr_built <= 1950 group by condition,'1926-1950'order by condition)union all(select '1951-1975'as YearRange,condition, avg(price)  as AveragePrice from house where yr_built >=1951 and yr_built <= 1975 group by condition,'1951-1975'order by condition)union all(select '1976-2000'as YearRange,condition, avg(price) as AveragePrice from house where yr_built >=1976 and yr_built <= 2000 group by condition,'1976-2000'order by condition)union all(select '2001-2015'as YearRange,condition, avg(price) as AveragePrice from house where yr_built >=2001 and yr_built <= 2015 group by condition,'2001-2015'order by condition)")
avgprice.show()


# ## Visualization

# In[37]:


import folium
import pandas as pd


# In[38]:


# define the world map
world_map = folium.Map()


# In[39]:


# San Francisco latitude and longitude values
latitude = 47.608013
longitude = -122.335167


# In[40]:


# Create map and display it
san_map = folium.Map(location=[latitude, longitude], zoom_start=12)


# In[41]:


# Read Dataset 
cdata = pd.read_csv('https://raw.githubusercontent.com/superfishmaster/BigDataHouse/master/kc_house_data.csv')
cdata.head()


# In[42]:


# get the first 200 crimes in the cdata
limit = 500
data = cdata.iloc[0:limit, :]


# In[43]:


# Instantiate a feature group for the incidents in the dataframe
incidents = folium.map.FeatureGroup()


# In[44]:


# Loop through the 200 crimes and add each to the incidents feature group
for lat, lng, in zip(cdata.lat, data.long):
    incidents.add_child(
        folium.CircleMarker(
            [lat, lng],
            radius=7, # define how big you want the circle markers to be
            color='yellow',
            fill=True,
            fill_color='red',
            fill_opacity=0.4
        )
    )


# In[45]:


# Add incidents to map
# We are creating a mmp of house locations in Washington area. The following graph below just show the point in the map.
#The second map show the lable with price.
san_map = folium.Map(location=[latitude, longitude], zoom_start=12)
san_map.add_child(incidents)


# In[46]:



# add pop-up text to each marker on the map
latitudes = list(data.lat)
longitudes = list(data.long)
labels = list(data.price)

for lat, lng, label in zip(latitudes, longitudes, labels):
    folium.Marker([lat, lng], popup=label).add_to(san_map)    
    
# add incidents to map
san_map.add_child(incidents)
san_map.add_child(incidents)


# In[47]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf


# In[48]:


cdata.iplot(kind='bar',y='price',x='yr_built',title='Price over Time')


# In[49]:



cdata=cdata.groupby(by='yr_built').sum()


# In[50]:


import plotly.graph_objs as go
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf


# In[51]:


cdata.iplot(kind='line',y='price',title='Price group by Build Year',
         xTitle='Build Year', yTitle='Price',colors='red')


# In[52]:


cdata.iplot(kind='line',y='sqft_living',title='Livingsqft group by Build Year',
         xTitle='yr_built', yTitle='livingsqft',colors='blue')


# ## Data Modeling--Machine Learning

# In[53]:


df_final.dropna()


# In[54]:


df_final.dropDuplicates()


# In[55]:


from pyspark.ml.feature import VectorAssembler


# In[56]:


df_final.columns


# In[57]:


# Vector assembler is used to create a vector of input features
assembler = VectorAssembler(inputCols=['bedrooms', 'bathrooms', 'condition', 'livingsqft', 'age'],
                            outputCol="features")


# In[58]:


final_data=assembler.transform(df_final)


# In[59]:


final_data.show()


# In[60]:


Reg_data=final_data.select(["features","price"])


# ### Chaisquare Test

# Chaisquare test is done in order to check the statistical significance of the different variables

# In[61]:


####Chaisquare test
from pyspark.ml.stat import ChiSquareTest


# In[62]:


result_hypothesis = ChiSquareTest.test(Reg_data,"features", "price").head()


# In[63]:


print("pValues: " + str(result_hypothesis.pValues))


# Interpretation: we can observe from the above output that all the p-values are 0 or 1 which explains that all the slected variables has some identifiable linear relationship.

# In[64]:


print("statistics: " + str(result_hypothesis.statistics))


# In[ ]:





# In[65]:


# Preparing train and test data set
splits = Reg_data.randomSplit([0.8, 0.2],seed='2020')
train_df = splits[0]
test_df = splits[1]


# In[66]:


train_df.describe().show()


# In these outputs we can observe the number of data points in the training and testing data test

# In[67]:


test_df.describe().show()


# ### Linear Regression

# In[68]:



# Create a Linear Regression Model object
from pyspark.ml.regression import LinearRegression
linearReg = LinearRegression(featuresCol = 'features', labelCol='price', maxIter=10, regParam=0.3, elasticNetParam=0.8)


# In[69]:


# Fit the model to the data and call this model lrModel
lr_model = linearReg.fit(train_df)


# In[70]:



print("Coefficients: " + str(lr_model.coefficients))


# In[71]:


'bedrooms', 'bathrooms', 'condition', 'livingsqft', 'age'


# From the model we have betacoefficients that explains the following:<br />
# For age of the house:-76780.67, this explains the decrese of price of house by 76780.67 for 1 year increase in house.<br />
# For bathrooms of the house:90570.58, this explains the increase of price of house by 90570.58 for 1 bathrooms increase in house.<br />
# For bedrooms of the house:10692.11, this explains the increase of price of house by 10692.11 for 1 bathrooms increase in house.<br />
# For condition of the house:298.97, this explains the increase of price of house by 298.97 for 1 increase in condition rating of house.<br />
# For livingsqft of the house:3055.73, this explains the increase of price of house by 3055.73 for increase in 1 square ft of living area in house.
# 

# In[72]:


print("Intercept: " + str(lr_model.intercept))


# In[73]:


test_results = lr_model.evaluate(test_df)


# In[74]:


print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))


# In[75]:


print("R-squared value: {}".format(test_results.r2))


# R-squared value of 0.54 tells us that this model is able to explaing approx 54 percent of variance in data with the applied input variables.

# In[76]:


lr_predictions=lr_model.transform(test_df)
lr_predictions.select(['prediction','Features','price']).toPandas()


# Interpretation: We have created the regression model which is able to explaing 54 percent of change in price if we use the 
# variables which are included in the above regression model. that means there is still 46 percent of variation unexplainded
# there can be more factors affecting the price such as inflation, Localicty(Area is urban,suburban ou rural) etc.
# 

# ### Random Forest Regressor Model

# In[77]:


from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator


# In[78]:


# Create a Linear Regression Model object
RandomReg = RandomForestRegressor(featuresCol = 'features', labelCol='price')


# In[79]:


# Fit the model to the data and call this model lrModel
rf_model = RandomReg.fit(train_df)
import warnings
warnings.filterwarnings("ignore")


# In[80]:


rf_predictions=rf_model.transform(test_df)
rf_predictions.select(['prediction','Features','price']).toPandas()


# In[81]:


# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(rf_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# In[82]:


import sklearn.metrics
y_true = rf_predictions.select("price").toPandas()
y_pred = rf_predictions.select("prediction").toPandas()
r2_score = sklearn.metrics.r2_score(y_true, y_pred)
print('r2_score: {0}'.format(r2_score))


# Interpretation: we have r2 value from Linear rgression model = 0.54, whereas 0.52 from random forest regressor. so we have better explaination of the variance in data from linear regression model.
# Although our regression model is able to explain 54 percent of variation, it is better model than Random forest regressor.
# here we find out there are other factors affecting the price which are not avilable in the data set such as inflation, locality and development in that region. So to develop the model with prediction we need to include those parameters which are not present in current data set
# 

# ### Model comparison

# r2 value: we have r2 value from Linear rgression model = 0.54, whereas 0.52 from random forest regressor. so we have better explaination of the variance in data from linear regression model. Although our regression model is able to explain 54 percent of variation, it is better model than Random forest regressor. here we find out there are other factors affecting the price which are not avilable in the data set such as inflation, locality and development in that region. So to develop the model with prediction we need to include those parameters which are not present in current data set.
# 
# Root Mean Squared Error: RMSE from regression model: 247017.49. (RMSE) from random forest regressor: 252755. We have lower value of RMSE for regression model than random forest.
# 
# Therefore, Regression model appear to be a better model fot this data set.

# In[ ]:




