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

# printing the model name and accuracy !!!!!
print("Model name: ", model)
print("Model score : " + str(model.score(X_test, y_test) * 100))
print("R2 score : ", r2_score(y_test, y_pred))
print("MSE : ", mean_squared_error(y_test, y_pred))
print("MAE : ", mean_absolute_error(y_test, y_pred))
print("-------------------------------------------------------------")

dump(mind,"salaryModel.pkl")

