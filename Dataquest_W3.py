#Dataquest Week 3 Commands - Intermediate Stream

### Mission 1: Linear Regression Model ###

#plotting a figure - plt.figure(figsize=(x,y))
fig = plt.figure(figsize=(7,15))
#plotting areas
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
#name plots on scatter plot
train.plot(x="Garage Area", y="SalePrice", ax=ax1, kind="scatter")
train.plot(x="Gr Liv Area", y="SalePrice", ax=ax2, kind="scatter")
train.plot(x="Overall Cond", y="SalePrice", ax=ax3, kind="scatter")
#display scatter plot
plt.show()

#instintate linear regression model
lr = LinearRegression()
#fit linear regression model - lr.fit(featured col, target col)
lr.fit(train[['Gr Liv Area']], train['SalePrice'])
#display coefficent and intercept of fitted model
print(lr.coef_)
print(lr.intercept_)
#assign values accordingly - values a0 = intercept of Linear Regressor and a1 = coefficent of Linear Regressor
a0 = lr.intercept_
a1 = lr.coef_

#make predictions for Linear Regressor (lr)
train_predictions = lr.predict(train[['Gr Liv Area']])
test_predictions = lr.predict(test[['Gr Liv Area']])
#calculating RMSE from predictions for Linear Regressor (lr)
train_mse = mean_squared_error(train_predictions, train['SalePrice'])
test_mse = mean_squared_error(test_predictions, test['SalePrice'])
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

### Mission 2: Feature Selection ###

#Select the integer and float columns from train and assign them to the variable numerical_train
numerical_train = train.select_dtypes(include=['int', 'float'])
#dropping columns from dataset numerical train (axis=1 is columns, axis=0 is rows)
numerical_train = numerical_train.drop(['PID', 'Year Built', 'Year Remod/Add', 'Garage Yr Blt', 'Mo Sold', 'Yr Sold'], axis=1)

#df.corr() - compute correlation coefficients between all columnbs in train_subset
corrmat = train_subset.corr()
#sort correlation matrix for better readability on heatmap df[featured col].abs().sort_values()
sorted_corrs = corrmat['SalePrice'].abs().sort_values()

#filter to strong correlations (0.3 or higher correlation)
strong_corrs = sorted_corrs[sorted_corrs > 0.3]
#recalculate correlation matrix with strong correlations
corrmat = train_subset[strong_corrs.index].corr()
#create and print heatmap - sns.heatmap(correlation matrix)
sns.heatmap(corrmat)

#dropping columns that have high correlation rates (this is to minimize risk of duplicate value) 
#Context: 'Garage Cars' and 'Garage Area' were very similar so we dropped Garage Cars. Same for 'TotRms AbvGrd' and 'Gr Liv Area' - so we dropped 'TotRms AbvGrd'
final_corr_cols = strong_corrs.drop(['Garage Cars', 'TotRms AbvGrd'])
#cleaning the test data set by dropping null values in columns
clean_test = test[final_corr_cols.index].dropna()

#Formula for feature rescaling
unit_train = (train[features] - train[features].min())/(train[features].max() - train[features].min())

#FULL CLEANING OF LINEAR REGRESSION MODEL
#initalize model
lr = LinearRegression()
#fit model - lr.fit(featured column, target column)
lr.fit(train[features], train['SalePrice'])
#make prediction on training and test data set
train_predictions = lr.predict(train[features])
test_predictions = lr.predict(clean_test[features])
#calculate MSE for both sets
train_mse = mean_squared_error(train_predictions, train[target])
test_mse = mean_squared_error(test_predictions, clean_test[target])
#calculate RMSE for both sets
train_rmse_2 = np.sqrt(train_mse)
test_rmse_2 = np.sqrt(test_mse)

### Mission 3: Gradient Descent ###

#function of derivative
# MSE(a0, a1) = (2/n)(n∑i=1)(x1^i)(a0 + a1(x1^i) - y^i)
def derivative(a1, xi_list, yi_list):
    len_data = len(xi_list)
    error = 0
    for i in range(0, len_data):
        error += xi_list[i]*(a1*xi_list[i] - yi_list[i])
    deriv = 2*error/len_data
    return deriv
 
#function for Gradient Descent
def gradient_descent(xi_list, yi_list, max_iterations, alpha, a1_initial):
    #we assume a1 is already a list of MSE values
    a1_list = [a1_initial]

    for i in range(0, max_iterations):
        a1 = a1_list[i]
        deriv = derivative(a1, xi_list, yi_list)
        a1_new = a1 - alpha*deriv
        a1_list.append(a1_new)
    return(a1_list)
  
#function for a1 derivative 
# MSE(a1) = (1/n)(n∑i=1)(a1(x1^i) - y^i)^2
def a1_derivative(a0, a1, xi_list, yi_list):
    len_data = len(xi_list)
    error = 0
    for i in range(0, len_data):
        error += xi_list[i]*(a0 + a1*xi_list[i] - yi_list[i])
    deriv = 2*error/len_data
    return deriv

#function for a0 derivative
# MSE(a1) = (1/n)(n∑i=1)(a0 + a1(x1^i) - y^i)^2
def a0_derivative(a0, a1, xi_list, yi_list):
    len_data = len(xi_list)
    error = 0
    for i in range(0, len_data):
        error += a0 + a1*xi_list[i] - yi_list[i]
    deriv = 2*error/len_data
    return deriv
  
### Mission 4: Ordinary Least Squares ###
(Ky Update)

### Mission 5: Processing and Transforming Features###
(Ky Update)
