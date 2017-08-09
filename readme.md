#### eCommerce Fraud Detection Case Study

Team members: Corinne Carlson, James Woodruff, Neil Aronson, Stephanie Chong, Yulia Yukina

#### Project Scope
For this case study, we were tasked with developing a model to help an online event ticketing company detect fraud.

Defining "fraud": Our dataset provided us with an "account type" category -- account types with the term "fraud" were classified as fraud, all other account types (including spam) were classified as non-fraud.

Deliverables:
* Flask app with API
* Web based front-end to enable quick triage of potential fraud

#### Data Cleaning
We did the following to clean the data:
* Converted time stamps into readable formats
* Imputed missing / whitespace data with null values
* Turned categorical variables into numerical values
* Dropped "account type" to prevent leakage
* Set aside text-based features due to time limitations and in consideration of computational/modeling speed

#### Feature Engineering
The raw dataset has 43 features. For the feature engineering stage, we created the following new variables:
* Number of previous payouts: Most instances of fraud had no previous payouts, so we thought this feature would have high signal in predicting fraud.
* Number of ticket types, total ticket cost, total ticket quantity, total revenue: Extracted from "ticket type" data
* Has payee name, has org name, has payout type: Systematically missing data seemed to have high correlation with fraud.

#### Modeling & Results
We selected the following models to run on our cleaned data:
* Logistic regression
* Random Forest
* Gradient Boosting
* SVM

We optimized the parameters for these models by using recall as our priority metric, because our goal is to minimize false negatives. This is because we assumed that it would be significantly more costly to misclassify true fraudulent cases than to mistake non-fraud cases as fraud.
Out of these models, gradient boosting provided the best results, with recall of 0.980 and precision of 0.978.
Our baseline model detects fraud by assigning probabilities to instances that were calculated from the training data. This returns a result of 0.299 recall and 0.581 accuracy.

#### Next Steps
* Conduct NLP on the text data
* Create profit curve / perform cost-benefit analysis
* Web-based dashboard to quickly predict fraud 

#### Files
* clean_data.py: Python script that cleans and engineers features on the raw data.
* make_models2.py: Python script that imports the data, develops the model, and stores the model.
* predict.py: Python script that reads in and vectorizes a single instance, and predicts the probability that the event is fraud.
* app.py: Python script that retrieves new data from server, calls predict.py, and displays predictions. 
