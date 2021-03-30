#Dataquest Week 5 Commands - Intermediate Stream

################################################ Mission 4: Overfitting ################################################
#importing our external functions
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
#this function is meant to calculate variance and MSE of our columns 
def train_and_test(cols):
    # Split into features & target.
    features = filtered_cars[cols]
    target = filtered_cars["mpg"]
    # Fit model.
    lr = LinearRegression()
    lr.fit(features, target)
    # Make predictions on training set.
    predictions = lr.predict(features)
    # Compute MSE and Variance.
    mse = mean_squared_error(filtered_cars["mpg"], predictions)
    variance = np.var(predictions)
    return(mse, variance)

#Calculating the MSE and Variance of specific column
cyl_mse, cyl_var = train_and_test(["cylinders"])
weight_mse, weight_var = train_and_test(["weight"])

#Here we're calculating the MSE and Variance of multiple columns - so we're getting the MSE and Variance of both column together as one value
two_mse, two_var = train_and_test(["cylinders", "displacement"])
#Calculating the MSE and Variance of 3 columns
three_mse, three_var = train_and_test(["cylinders", "displacement", "horsepower"])
#Calculating the MSE and Variance of 4 columns
four_mse, four_var = train_and_test(["cylinders", "displacement", "horsepower", "weight"])

#This function uses KFold and LinearRegression to performs 10-fold validation 
#Recap: KFold cross validation is used to test the model's ability to predict new data that was not used in estimating it by dividing them up into "k" sections
def train_and_cross_val(cols):
    features = filtered_cars[cols]
    target = filtered_cars["mpg"]
    
    variance_values = []
    mse_values = []
    
    # KFold instance.
    kf = KFold(n_splits=10, shuffle=True, random_state=3)
    
    # Iterate through over each fold.
    for train_index, test_index in kf.split(features):
        # Training and test sets.
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        
        # Fit the model and make predictions.
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        predictions = lr.predict(X_test)
        
        # Calculate mse and variance values for this fold.
        mse = mean_squared_error(y_test, predictions)
        var = np.var(predictions)

        # Append to arrays to do calculate overall average mse and variance values.
        variance_values.append(var)
        mse_values.append(mse)
   
    # Compute average mse and variance values.
    avg_mse = np.mean(mse_values)
    avg_var = np.mean(variance_values)
    return(avg_mse, avg_var)
  
#Scatter a number of different featues at once
#Here we can also assign a color to each set using "c='color'"
plt.scatter([2,3,4,5,6,7], [two_mse, three_mse, four_mse, five_mse, six_mse, seven_mse], c='red')
plt.scatter([2,3,4,5,6,7], [two_var, three_var, four_var, five_var, six_var, seven_var], c='blue')
#plt.show() helps us show the scatter plot in our command output
plt.show()

################################################ Mission 5: Clustering Basics ################################################
import pandas as pd
#reading csv file and assigning it to variable votes
votes = pd.read_csv("114_congress.csv")

#df[col].value_counts() - Return a Series containing counts of unique values
print(votes["party"].value_counts())
#calculates average "votes" in the dataframe - the mean 
print(votes.mean())

#calculates euclidean distance of the senators (starting from column 3 and onward because 0-2 contain name, party, state)
#-1 in reshape function is used when you dont know or want to explicitly tell the dimension of that axis
#the 1 will reshape in such a way that the resulting array only has 1 column
distance = euclidean_distances(votes.iloc[0,3:].values.reshape(1, -1), votes.iloc[2,3:].values.reshape(1, -1))

#initalizing KMeans method that initalizes n clusters and random_state
#n_clusters is assigned to how many clusters you want 
#random_state=1 to allow for the same results to be reproduced whenever the algorithm runs
kmeans_model = KMeans(n_clusters=2, random_state=1)
#model.fit_transform - allows us to compare the euclidean distances of each row to the n clusters
senator_distances = kmeans_model.fit_transform(votes.iloc[:, 3:])

#Use the labels_ attribute to extract the labels from kmeans_model
labels = kmeans_model.labels_
#Use the crosstab() method to print out a table comparing labels to votes["party"]
#crosstab() Compute a simple cross tabulation of two (or more) factors
print(pd.crosstab(labels, votes["party"]))
#Selects all Senators who were assigned to the second cluster(1) that were Democrats.
#Here we can include conditions in our variable
democratic_outliers = votes[(labels == 1) & (votes["party"] == "D")]

#Compute an extremism rating by cubing every value in senator_distances, then finding the sum across each row
extremism = (senator_distances ** 3).sum(axis=1)
#Sort votes on the extremism column, in descending order
votes.sort_values("extremism", inplace=True, ascending=False)
#Here we print out the first 10 values of votes
print(votes.head(10))

################################################ Mission 6: K-means Clustering  ################################################
#NOTE: Most of this missions will deal with experimenting with values with minimal coding. Feel free to experiment with different values in this mission!

#We create a new dataframe that only contain the pointguards(PG) in the dataset using a conditional statement
#position = Point Guard
point_guards = nba[nba['pos'] == 'PG']

#ATR = Assist Turnover Ration = Assists/Turnovers
#Calculating the ATR row by dividing assets row by turnover row
point_guards['atr'] = point_guards['ast'] / point_guards['tov']

#REMEMBER: For the K-means clustering algorithm, there are 2 steps
#(1) recalculating the centroid of each cluster 
#(2) calculating the number of items in each cluster
#We assign a row to whichever cluster it is most similar to using "Points Per Game" and "Assist Turnover Ratio" 
def assign_to_cluster(row):
    lowest_distance = -1
    closest_cluster = -1
    #extracting the row elements we need and calculating the euclidean distance of the centroid and row
    for cluster_id, centroid in centroids_dict.items():
        df_row = [row['ppg'], row['atr']]
        euclidean_distance = calculate_distance(centroid, df_row)
        #update thhe lowest_distance and closest_cluster accordingly
        if lowest_distance == -1:
            lowest_distance = euclidean_distance
            closest_cluster = cluster_id 
        elif euclidean_distance < lowest_distance:
            lowest_distance = euclidean_distance
            closest_cluster = cluster_id
    return closest_cluster
#calculate euclidean distance to clusters for each row
point_guards['cluster'] = point_guards.apply(lambda row: assign_to_cluster(row), axis=1)

#Function to recalculate centroids after assigning to new cluster
def recalculate_centroids(df):
    new_centroids_dict = dict()
    
    for cluster_id in range(0, num_clusters):
        values_in_cluster = df[df['cluster'] == cluster_id]
        # Calculate new centroid using mean of values in the cluster
        new_centroid = [np.average(values_in_cluster['ppg']), np.average(values_in_cluster['atr'])]
        new_centroids_dict[cluster_id] = new_centroid
    return new_centroids_dict
