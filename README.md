# Bondora-P2P-lending

# Bondora-P2P-lending
# Analysing-Credit-Risk-on-European-Peer-to-Peer-lending-Platform
In this project we will be doing credit risk modelling of peer to peer lending Bondora systems.Data for the study has been retrieved from a publicly available data set of a leading European P2P lending platform (Bondora).The retrieved data is a pool of both defaulted and non-defaulted loans from the time period between 1st March 2009 and 27th January 2020. The data comprises of demographic and financial information of borrowers, and loan transactions.In P2P lending, loans are typically uncollateralized and lenders seek higher returns as a compensation for the financial risk they take. In addition, they need to make decisions under information asymmetry that works in favor of the borrowers. In order to make rational decisions, lenders want to minimize the risk of default of each lending decision, and realize the return that compensates for the risk.

## Understanding the Dataset
The dataset we are working on is a combination of defaulted and non-defaulted loans from **1st March 2009** to **27th January 2020**.

## Preprocessing and Sentiment Analysis

-First of all we remove all the feature that have more the 40% of null values.
For the categorical features, we filled the nan values with value "empty".
For the numerical features, we filled the NaN values using the back filling method.

-Then we checked the missed values in the Dataframe , it was complete.
Apart from missing value features, there are some features which will have no role in default prediction like **'ReportAsOfEOD', 'LoanId', 'LoanNumber', 'ListedOnUTC', 'DateOfBirth' (because age is already present), 'BiddingStartedOn','UserName','NextPaymentNr','NrOfScheduledPayments','IncomeFromPrincipalEmployer', 'IncomeFromPension', 'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare','IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther' (As Total income is already present which is total of all these income), 'LoanApplicationStartedDate','ApplicationSignedHour', 'ApplicationSignedWeekday','ActiveScheduleFirstPaymentReached', 'PlannedInterestTillDate', 'LastPaymentOn', 'ExpectedLoss', 'LossGivenDefault', 'ExpectedReturn', 'ProbabilityOfDefault', 'PrincipalOverdueBySchedule', 'StageActiveSince', 'ModelVersion','WorseLateCategory'**
So, all of the above mentioned features were actually dropped.

-Then we dealt with the mis-entered features, which are the features that users misentered their values, such as MaritalStatus, UseOfLoan, OccupationArea, and EmploymentStatus. So, those values are replaced with "empty". 
- Creating target variable 
Here, status is the variable which help us in creating target variable. The reason for not making status as target variable is that it has three unique values current, Late and repaid. There is no default feature but there is a feature default date which tells us when the borrower has defaulted means on which date the borrower defaulted. So, we will be combining Status and Default date features for creating target variable.The reason we cannot simply treat Late as default because it also has some records in which actual status is Late but the user has never defaulted i.e., default date is null. So we will first filter out all the current status records because they are not matured yet they are current loans.

- Checking datatype of all features
In this step we will see any data type mismatch, 
we start with Checking distribution of categorical variables and all numeric columns

> First we will delete all the features related to date as it is not a time series analysis so these features will not help in predicting target variable.
> As we can see in numeric column distribution there are many columns which are present as numeric but they are actually categorical as per data description such as Verification Type, Language Code, Gender, Use of Loan, Education, Marital Status,EmployementStatus, OccupationArea etc.
> So we will convert these features to categorical features by decoding them.


-----------------------------------------------------------------------------------------------------------------------------------------
## EDA 
**Introduction:**

- Dataset comprises of 134529 rows and 112 columns.
- Dataset comprises of float variables and categorical data type. 

**Information of Dataset:**

Using countplot on target variable **Status** we could see that "Current" loans have more than 60,000 values, while "Late" and "Repaid" loans have about 45,000 and 30,000 repectively. By this information we could conclude that there is no imbalanced in the data and hence balancing of data is not required.

**Univariate Analysis:**

Plotted histograms and countplots to see the distribution of data for each column.

**Descriptive Statistics:**

Using **describe()** we could get the following result for the numerical features:
-Mean
-Mode
-Median
-std
-min
-max
-25th percentile
-75th percentile

-Visualization of Variables:
-As illustrated, new customers resembles more than half the company's customers.
-As shown, the maturity date for most of the loans has been restructured and increased by more than 60 days.
-Only 5% of the borrowers didn't submit the first payment according to the schedule.
-The majority of customers are identified as "Men".
-Most of the customers have either secondary or Higher Education.
-The density of customers who has current loan status is higher than late and Repaid status loans in the age range of 20-70.
-Most of the loans has interest rate of 20-40.
-The majority of loans have an expected loss in the range of 0.05-0.15.
-The probability of loan's default in one year horizon in the range of 0.1-0.35 for most of the loans. 
-In the range of 0-500 loan's status is either late or current.While the range of 500-4000 , most of the loans the repaid.
-The probability of default for loans increases as the the interest increases.
-----------------------------------------------------------------------------------------------------------------------------------------


Before modelling and after splitting we scaled the data using standardization to shift the distribution to have a mean of zero and a standard deviation of one.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X1 = scaler.fit_transform(X1)
```
**fit_transform()** is used on the training data so that we can scale the training data and also learn the scaling parameters of that data. Here, the model built by us will learn the mean and variance of the features of the training set. These learned parameters are then used to scale our test data.

**transform()** uses the same mean and variance as it is calculated from our training data to transform our test data. Thus, the parameters learned by our model using the training data will help us to transform our test data. As we do not want to be biased with our model, but we want our test data to be a completely new and a surprise set for our model.

## Model Building

#### Metrics considered for Model Evaluation
**Accuracy , Precision , Recall and F1 Score**
- Accuracy: What proportion of actual positives and negatives is correctly classified?
- Precision: What proportion of predicted positives are truly positive ?
- Recall: What proportion of actual positives is correctly classified ?
- F1 Score : Harmonic mean of Precision and Recall

### Extra trees Classifier:
Extremely Randomized Trees Classifier(Extra Trees Classifier) is a type of ensemble learning technique which aggregates the results of multiple de-correlated decision trees collected in a “forest” to output it's classification result.
### Mutual Info Classifier:
Mutual information (MI) [1] between two random variables is a non-negative value, which measures the dependency between the variables.

~~~

### Choosing the features
From the above two methods, we got which features are highly correlated with the target variable.Those features were used to train two different models to check the accuracy.



#### 1. Applying SVM on MI columns
By applying on the MI columns, the accuracy got from the confusion matrix was 91.808%.

#### 2. Applying G Boost  on MI columns
By applying XGBoost Classifier on the  MI columns,the accuracy got from the confusion matrix was 92.36%.

#### 3. Applying SVM on Extra trees columns
By applying on the Extra treescolumns, the accuracy got from the confusion matrix remained was 91.6 %.

#### 4. Applying GBoost  on Extra trees columns
By applying GBoost Classifier on the Extra treescolumns,the accuracy we got from the confusion matrix was 92.26 %.


## Deployment
you can access our app by following this link (https://bondoraloans.herokuapp.com/)
### Streamlit
- It is a tool that lets you creating applications for your machine learning model by using simple python code.
- We write a python code for our app using Streamlit; the app asks the user to enter some information regarding the applied loan. 
- The output of our app will be whether the loan has been accepted or denied.
- The app runs on local host.
- To deploy it on the internt we have to deploy it to Heroku.

### Heroku
We deploy our Streamlit app to [ Heroku.com](https://www.heroku.com/). In this way, we can share our app on the internet with others. 
We prepared the needed files to deploy our app sucessfully:
- Procfile: contains run statements for app file and setup.sh.
- setup.sh: contains setup information.
- requirements.txt: contains the libraries must be downloaded by Heroku to run app file (webapp.py)  successfully 
- webapp.py: contains the python code of a Streamlit web app.
- Saved_Model.sav: contains our GBoost model that built by modeling part.


