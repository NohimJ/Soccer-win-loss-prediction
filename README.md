⚽ Soccer Match Win/Loss Prediction
this project predicts the outcome of soccer matches using a Random Forest.
the Notebook goes through Exploratory Data Analysis, feature engineering,
model training and evaluation.

This project was followed from a tutorial on Youtube by "DataQuest"

some of the code consisted of

**#1 Loading the dataset**
```python
import pandas as pd
matches = pd.read_csv('/content/matches.csv', index_col=0)
```
**#2 Exploring the data**
```python
matches.head()
```
**#3 One-hot encoding the object columns**
```python
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
#rest of the columns...
```
**#4 Splitting the data into train and test data and creating the Random Forest**
```python
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']
```
**#5 Choosing what columns the Random Forest was to predict on and training the model**
```python
predictors = ["venue_code", "opponent", "hour", "day_code"]
rf.fit(train[predictors], train["target"])
preds = rf.predict(test[predictors])
from sklearn.metrics import accuracy_score
acc = accuracy_score(test["target"], preds)
#acc = 0.6123188405797102 approx 61%
```
The full code is included

