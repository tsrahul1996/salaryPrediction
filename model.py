from pandas import read_csv
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = read_csv("salary_data.csv")

X = df["YearsExperience"].values.reshape(30,1)

y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mind = LinearRegression()

mind.fit(X_train, y_train)

dump(mind,"salaryModel.pkl")
