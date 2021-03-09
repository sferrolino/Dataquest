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
