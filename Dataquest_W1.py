#Dataquest Week 1 Commands - Intermediate Stream

#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, 
#along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
#Pandas is a software library written for the Python programming language for data manipulation and analysis
#We can perform basic operations on rows/columns like selecting, deleting, adding, and renaming.
import pandas as pd

#reading csv file
dc_listings = pd.read_csv ('dc_airbnb.csv')
#print first row of .csv file we read in dc_listings
print(dc_listings[:1])

#initialize
our_value = 3
#df.iloc[r][c] - allows us to select a particular cell in the data set
first_living_space_value = dc_listings.iloc[0]['accommodates']
#np.abs(v2 - v1) - Calculating the absolute value of a number
first_distance = np.abs(first_living_space_value - our_value)

#get euclidean distance
new_listing = 3
#df[column].apply(lambda x: np.abs(x)) - 
dc_listings['distance'] = dc_listings['accommodates'].apply(lambda x: np.abs(x - new_listing))
#display unique values from distance column
print(dc_listings['distance'].value_counts())


#
#initalize shuffled_index
shuffled_index = np.random.permutation(len(dc_listings["distance"]))

#use loc[row_indexer, column_indexer] on dc_listings
dc_listings = dc_listings.loc[shuffled_index]

#sort by distance column
dc_listings = dc_listings.sort_values(by=['distance'])

#print first 10
print(dc_listings[:10]['price'])

#
#remove commas and dollar sign char
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')

#change to float datatype
dc_listings['price'] = stripped_dollars.astype(float)

#calculate mean value (average of numbers)
mean_price = dc_listings[0:5]['price'].mean()
print(mean_price)
