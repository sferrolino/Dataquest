#Dataquest Week 2 Commands - Intermediate Stream

#### MISSION 4: Hyperparameter Optimization ####
#reading csv files recap
train_df = pd.read_csv('dc_airbnb_train.csv')

#initalizing your hyper_params and using them in a for loop for hyperparameter optimization
hyper_params = [1,2,3,4,5]
for x in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=x, algorithm="brute")
    knn.fit(train_df[features], train_df['price'])
    predictions = knn.predict(test_df[features])
    mse = mean_squared_error(test_df['price'], predictions)
    mse_values.append(mse)

#plotting your scatter plot .scatter(x-axis, y-axis)   
plt.scatter(hyper_params, mse_values)
#show scatter plot
plt.show()

#convert all columns in dataset to list
features = train_df.columns.tolist()
#remove specific element in list
features.remove('price')

#find lowest mse
for k,mse in enumerate(mse_values): #for index and mse, iterate through list
    if mse < two_lowest_mse:
        two_lowest_mse = mse 
        two_lowest_k = k + 1 #need to update this reference for key value
    
#creating dictionary with key and pair
#dictionary[key] = value
two_hyp_mse[two_lowest_k] = two_lowest_mse
    
#### MISSION 5: Cross Validation ####
#replacing rows in column - replacing special char with whitespace for easier readability
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')

#using numpy.random.permutation() function to shuffle the ordering of the rows in dc_listings
#create shuffled_index object
shuffled_index = np.random.permutation(dc_listings.index)
#shuffling indexes of dc_listings using our shuffled object
dc_listings = dc_listings.reindex(shuffled_index)

#finding average RMSE of 2 RMSE values
avg_rmse = np.mean([iteration_two_rmse, iteration_one_rmse])

#Creating fold rows
#REFERENCE dc_listings.loc[] - access a group of rows or columns
dc_listings.loc[dc_listings.index[0:745], "fold"] = 1       #referenced as fold 1
dc_listings.loc[dc_listings.index[745:1490], "fold"] = 2    #referenced as fold 2...etc.
dc_listings.loc[dc_listings.index[1490:2234], "fold"] = 3
dc_listings.loc[dc_listings.index[2234:2978], "fold"] = 4
dc_listings.loc[dc_listings.index[2978:3723], "fold"] = 5

#Display the unique value counts for the fold column to confirm that each fold has roughly the same number of elements.
print(dc_listings['fold'].value_counts())
#Display the number of missing values in the fold column to confirm we didn't miss any rows
print("missing values: ", dc_listings['fold'].isnull().sum())

#Performing K-fold cross validation in a method
fold_ids = [1,2,3,4,5]
#START CODE
def train_and_validate(df, folds):
    #initialize result list
    fold_rmses = []
    #want to go through all fold_ids, so we make a loop
    for x in folds:
        knn = KNeighborsRegressor()
        train = dc_listings[dc_listings["fold"] != x]
        test = dc_listings[dc_listings["fold"] == x].copy()
        knn.fit(train[['accommodates']], train['price'])
        
        prediction = knn.predict(test[['accommodates']])
        test['predicted_price'] = prediction
        mse = mean_squared_error(test['price'], test['predicted_price'])
        rmse = mse**(1/2)
        fold_rmses.append(rmse)
    return(fold_rmses)
#calling the method we created
rmse_list = train_and_validate(dc_listings, fold_ids)
#calculating the average RMSE
avg_rmse = np.mean(rmse_list)

#IMPLEMENTING cross_val_score and KFold
from sklearn.model_selection import cross_val_score, KFold
#kf = KFold(n_splits, shuffle=False, random_state=None)
kf = KFold(5, shuffle=True, random_state=1)
knn = KNeighborsRegressor()
#cross_val_score(estimator, training column, target column, scoring=, cv=) - This function returns array of MSE values f
mses = cross_val_score(knn, dc_listings[['accommodates']], dc_listings['price'], scoring='neg_mean_squared_error', cv=kf)
#rmse = root mean square error, remember we need abs val
rmses = np.sqrt(np.absolute(mses))
#calculate avg
avg_rmse = rmses.mean()

#Implementing leave-one-out cross validation - using KFold and cross_val_score (adding on to our prior knowledge of test/train validation)
num_folds = [3, 5, 7, 9, 10, 11, 13, 15, 17, 19, 21, 23]
for fold in num_folds:
    #initalize KFold
    kf = KFold(fold, shuffle=True, random_state=1)
    #instintate knn model
    model = KNeighborsRegressor()
    #find the MSE
    mses = cross_val_score(model, dc_listings[["accommodates"]], dc_listings["price"], scoring="neg_mean_squared_error", cv=kf)
    #find the RMSE
    rmses = np.sqrt(np.absolute(mses))
    #find average
    avg_rmse = np.mean(rmses)

## END OF CODE ##
