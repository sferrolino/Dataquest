#Dataquest Week 1 Commands - Intermediate Stream

#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, 
#along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
#Pandas is a software library written for the Python programming language for data manipulation and analysis
#We can perform basic operations on rows/columns like selecting, deleting, adding, and renaming.
import pandas as pd

##### MISSION 1: Intro to K-Nearest Neighbors #####
#reading csv file
dc_listings = pd.read_csv ('dc_airbnb.csv')
#print first 4 columns (1-4), OMIT index 0 b/c it's the header
print(dc_listings[1:5])
#print first 10 entries of price column
print(dc_listings[:10]['price'])

#df.iloc[r][c] - allows us to select a particular cell in the data set
first_living_space_value = dc_listings.iloc[0]['accommodates']
#np.abs(v2 - v1) - Calculating the absolute value of a number
first_distance = np.abs(first_living_space_value - our_value)

#df[column].apply(lambda x: np.abs(x)) - Apply a function along an axis of the DataFrame, here we apply np.abs() val to each value.
dc_listings['distance'] = dc_listings['accommodates'].apply(lambda x: np.abs(x - new_listing))
#df[column].value_counts() - display unique values from column
print(dc_listings['distance'].value_counts())

#initalize shuffled_index
shuffled_index = np.random.permutation(len(dc_listings["distance"]))

#use loc[row_indexer, column_indexer] - Access a group of rows and columns by label(s) - in this case reading it based on shuffled index
dc_listings = dc_listings.loc[shuffled_index]

#df.sort_values(by=[column]) - sorting rows by column
dc_listings = dc_listings.sort_values(by=['distance'])

#df[column].str.replace(char to replace, replacement) and list.str.replace(char to replace, replacement) - remove commas and dollar sign char
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')

#list.astype(datatype) - change datatype
dc_listings['price'] = stripped_dollars.astype(float)

#calculate mean value (average of numbers)
mean_price = dc_listings[0:5]['price'].mean()

#create a local copy of a data frame (dc_listings)
temp_df = dc_listings.copy()

##### MISSION 2: Evaluating Model Performance #####
#using what we've learned in mission 1 to create a predict_price(new_listing) function
#this function is meant to represent a machine learning model, outputting the prediction based on input
def predict_price(new_listing):
    temp_df = train_df.copy() #copy dataframe
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing)) #apply abs() to all values in column
    temp_df = temp_df.sort_values('distance') # sort columns
    nearest_neighbor_prices = temp_df.iloc[0:5]['price'] #retrtieve the first 4 entries of 'price'
    predicted_price = nearest_neighbor_prices.mean() #find the mean of the 4 entries
    return(predicted_price) #return predicted price
  
#how to calculate MAE (mean absolute error)
test_df['error'] = np.absolute(test_df['predicted_price'] - test_df['price'])
mae = test_df['error'].mean()

#how to calculate MSE (mean squared error)
test_df['squared_error'] = np.square(test_df['predicted_price'] - test_df['price'])
mse = test_df['squared_error'].mean()

#how to calculate RMSE (root mean squared error)
test_df['predicted_price'] = test_df['bathrooms'].apply(lambda x: predict_price(x))
test_df['squared_error'] = (test_df['predicted_price'] - test_df['price'])**(2)
mse = test_df['squared_error'].mean()
#2 OPTIONS:
rmse = np.sqrt(mse) 
rmse = mse**(1/2)

##### MISSION 3: Multivariate K-Nearest Neighbors #####

#return the number of non-null values in each column
dc_listings.info()

#DROPPING COLUMNS: DataFrame.drop(columns, axis=1) 
#NOTE: axis=1 (drop columns), axis=0 (drop rows)
drop_columns1 = ['room_type', 'city', 'state']
dc_listings = dc_listings.drop(drop_columns1, axis=1)

#isnull() - checks for missing values
#isnull().sum() - return number of missing values
dc_listings.isnull().sum()

#normalize feature columns in dc_listings and assign the new Dataframe containing just the normalized feature columns to normalized_listings
normalized_listings = (dc_listings - dc_listings.mean()) / (dc_listings.std())

#find euclidean distance between 2 listings
first_listing = normalized_listings.iloc[0][['accommodates', 'bathrooms']]
fifth_listing = normalized_listings.iloc[4][['accommodates', 'bathrooms']]
first_fifth_distance = distance.euclidean(first_listing, fifth_listing)

#KNN regression is to calculate the average of the numerical target of the K nearest neighbors.
#Scikit-learn uses a similar object-oriented style to Matplotlib and you need to instantiate an empty model first by calling the constructor:
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
#KNeighborsRegressor(n_neighbors=, algorithm=)
#if n_neighbors blank then default=5, OPTIONS: int
#if algorithm is blank, default='auto' OPTIONS: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
knn = KNeighborsRegressor(algorithm='brute')
knr = KNeighborsRegressor(n_neighbors=5, algorithm='brute')

#fit knn model - use fit method to specify the data we want the k-nearest neighbor model to use.
#parameters: training data (featured columns) and training data (target column)
feature_columns = ['accommodates', 'bathrooms']
knr.fit(train_df[feature_columns], train_df['price'])

#knn.predict(df[[c]]) - make predictions on the columns from df and assign it to predictions
predictions = knr.predict(test_df[['accommodates', 'bathrooms']])

#WHOLE KNN REGRESSOR PROCESS (6 STEPS)
#featured columns
train_columns = ['accommodates', 'bathrooms']
#instintate KNN
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute', metric='euclidean')
#fit model
knn.fit(train_df[train_columns], train_df['price'])
#make predictions
predictions = knn.predict(test_df[train_columns])
#find the mse
two_features_mse = mean_squared_error(test_df['price'], predictions)
#find the rmse
two_features_rmse = two_features_mse**(1/2)

#retrieve all columns in dataframe to list
features = train_df.columns.tolist()
#remove element from list
features.remove('price')           

#### END OF FILE ####
