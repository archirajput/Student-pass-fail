
#======= USING LOGISTIC REGRESSION =======


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set_style('whitegrid')


# to load data

file_path = r"/Users/aishwaryabhatnagar/Desktop/minor_project/student_pass_fail/dataset.csv"
df = pd.read_csv(file_path)

print("\n---- Intial Data ----")
print(df.head())         # prints the first 5 rows

print("\n---- Data Info ----")
print(df.info())         # prints the coloumn name, data types and missing values

print("\n---- Statistical Summary ----")
print(df.describe())     # to viualize the mean, SD, etc. (the statistical part)


# to handle missing data

df = df.fillna(df.mean(numeric_only=True))
       # fills the missing values in the dataset with the mean of the column of the missing value



# split features and targets
x = df.drop(["study_hours","result"], axis = 1)
y = df["result"]

# train and test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)     # splits data into 80% training and 20% testing

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)      # calcultes mean and SD from training set
x_test_scaled = scaler.transform(x_test)      


# training logistic regression model

model = LogisticRegression()
model.fit(x_train_scaled, y_train)

# to predict the results
y_predict = model.predict(x_test_scaled)



# Model evaluation
accuracy = accuracy_score(y_test, y_predict)      # y_test for actual values and y_predict for predicted values
cm = confusion_matrix(y_test, y_predict)   # shows true positive, true negative, false positive, false negative by using the confusion matrix

print("\n==== Model Evaluation ====")
print(f"Accuracy :- {accuracy:.2f}")

print("\n==== Confusion Matrix ====")
print(cm)

print("\n==== Classification Report ====")
print(classification_report(y_test, y_predict))



# Confusion Matrix Visualization

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Student Pass/Fail")
plt.tight_layout()
plt.show()
