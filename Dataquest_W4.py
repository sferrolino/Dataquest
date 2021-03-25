#Dataquest Week 4 Commands - Intermediate Stream

################################################ Mission 1: Logistic Regression ################################################
### plotting
#importing packages
# pandas allow us to interact with dataframes
# matplotlib.pylot allows us to plot out graphs
import pandas as pd
import matplotlib.pyplot as plt
#reading admission excel file using pandas
admissions = pd.read_csv("admissions.csv")
# plotting graph using plt
# having GPA as our X value and admit as our Y value
plt.scatter(admissions['gpa'], admissions['admit'])
# .show() shows the plt graph we just created
plt.show()
 
###Logistic function
import numpy as np
# define logistic as a function and accept a value
def logistic(x):
    # np.exp(x) raises x to the exponential power, ie e^x. e ~= 2.71828
    return np.exp(x)  / (1 + np.exp(x)) 
    
# Generate 50 real values, evenly spaced, between -6 and 6.
x = np.linspace(-6,6,50, dtype=float)

# Transform each number in t using the logistic function.
y = logistic(x)

# Plot the resulting data.
plt.plot(x, y)
plt.ylabel("Probability")
plt.show()

### training Linear Regression and Logistic models
# importing linear regression model 
from sklearn.linear_model import LinearRegression
# setting the model to linear_model
linear_model = LinearRegression()
# training the model using GPA as our X value and admit as our Y value
linear_model.fit(admissions[["gpa"]], admissions["admit"])
# importing Logistic regression model 
from sklearn.linear_model import LogisticRegression
# setting the model to logistic_model
logistic_model = LogisticRegression()
# training the model using GPA as our X value and admit as our Y value
logistic_model.fit(admissions[["gpa"]], admissions["admit"])

### predicting GPA using logistics
#setting the model 
logistic_model = LogisticRegression()
# training the model
logistic_model.fit(admissions[["gpa"]], admissions["admit"])
# using trained model with predict_proba method to predict GPA
# predict_proba returns an estimate for all classes 
pred_probs = logistic_model.predict_proba(admissions[["gpa"]])
# graph using a scatterplot with GPA as X and our predictions as Y
plt.scatter(admissions["gpa"], pred_probs[:,1])

###predicting GPA using Logistic 
#setting the model
logistic_model = LogisticRegression()
# training the model
logistic_model.fit(admissions[["gpa"]], admissions["admit"])
# using the trained model to predict GPA
fitted_labels = logistic_model.predict(admissions[["gpa"]])
# graph using a scatterplot with GPA as X and our predictions as Y
plt.scatter(admissions["gpa"], fitted_labels)

################################################ Mission 2: Introduction to evaluating binary classifiers ################################################
#reading csv file admisions
admissions = pd.read_csv("admissions.csv")
#initalize and fit LogisticRegression model
model = LogisticRegression()
model.fit(admissions[["gpa"]], admissions["admit"])
#predict labels of featured column
labels = model.predict(admissions[["gpa"]])
#assign labels predicted above to "predicted_label" column
admissions["predicted_label"] = labels

#assign admit values to actual_label column
admissions["actual_label"] = admissions["admit"]
#if predicted label equals actual label, then add it to list "matches"
matches = admissions["predicted_label"] == admissions["actual_label"]
#make a list of values from matches column in admission dataset
correct_predictions = admissions[matches]
#accuracy equation
accuracy = len(correct_predictions) / len(admissions)

#Calculating True Positives - We predicted they would be admitted & they were
true_positive_filter = (admissions["predicted_label"] == 1) & (admissions["actual_label"] == 1)
true_positives = len(admissions[true_positive_filter])
#Calculating True Negatives - We predicted they wouldn't be admitted & they weren't
true_negative_filter = (admissions["predicted_label"] == 0) & (admissions["actual_label"] == 0)
true_negatives = len(admissions[true_negative_filter])

#Calculating False Negatives - We predicted they weren't admitted & they actually were
false_negative_filter = (admissions["predicted_label"] == 0) & (admissions["actual_label"] == 1)
false_negatives = len(admissions[false_negative_filter])
#calculate sensitivity (TRUE POSITIVE RATE)
sensitivity = true_positives / (true_positives + false_negatives)

#Calculating False Positive - We predicted they were admitted & they actually weren't
false_positive_filter = (admissions["predicted_label"] == 1) & (admissions["actual_label"] == 0)
false_positives = len(admissions[false_positive_filter])
#calculate specificity (TRUE NEGATIVE RATE)
specificity = (true_negatives) / (false_positives + true_negatives)

################################################ Mission 3: Multiclass Classification ################################################
#read csv file and assign to cars
cars = pd.read_csv("auto.csv")
#find all unique values in column "origin" from cars dataset
unique_regions = cars["origin"].unique()

#create dummy data from cars['year']
dummy_years = pd.get_dummies(cars["year"], prefix="year")
#merge cars and dummy_years dataset together (columns)
cars = pd.concat([cars, dummy_years], axis=1)
#drop column 'year' & 'cylinders'
cars = cars.drop("year", axis=1)
cars = cars.drop("cylinders", axis=1)

#getting the index of the 70% mark of the dataset
highest_train_row = int(cars.shape[0] * .70)
#split into train and test datasets
#train is 70% of dataset
train = shuffled_cars.iloc[0:highest_train_row]
#test is 30% of dataset
test = shuffled_cars.iloc[highest_train_row:]

#START CODE
features = [c for c in train.columns if c.startswith("cyl") or c.startswith("year")]
#for every element in unique_origins
for origin in unique_origins:
    #initalize model
    model = LogisticRegression()
    #create features list
    X_train = train[features]
    #make a boolean list if current element we're iterating through equals the row values
    #e.g. origin = Italy (for this loop)
    # train["origin"]  |   y_train
    # Italy            |   True
    # Japan            |   False
    # USA              |   False
    y_train = train["origin"] == origin
    #fit model 
    model.fit(X_train, y_train)
    #update origin column in model
    models[origin] = model
    
#creating list of testing_prob from unique_origins column
testing_probs = pd.DataFrame(columns=unique_origins)  
#for each unique element in unique_origins
for origin in unique_origins:
    # Select testing features.
    X_test = test[features]   
    # Compute probability of observation being in the origin.
    testing_probs[origin] = models[origin].predict_proba(X_test)[:,1]
    
#Classify each observation in the test set using the testing_probs
predicted_origins = testing_probs.idxmax(axis=1)


