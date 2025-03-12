from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,  mean_absolute_error, mean_squared_error, r2_score

def prepAndTrainCSVForLinearRegression(df):
    train_x, test_x, train_y, test_y = prepAndTrainRegression(df)

    reg = LinearRegression()
    reg.fit(train_x, train_y)

    pred_y = reg.predict(test_x)
    return train_x, test_x, train_y, test_y, pred_y

def prepAndTrainCSVForLogisticRegression(df):
    train_x, test_x, train_y, test_y = prepAndTrainRegression(df)
    model = LogisticRegression()
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    
    return test_y, pred_y

# def logisticF1Score(test_y, pred_y):
#     f1 = f1_score(test_y, pred_y)
#     print(f1)


def prepAndTrainRegression(df):
    # Prepare X and y
    X = df.iloc[:,1:-1]
    y = df.iloc[:,-1]

    # Do train test split
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    return train_x, test_x, train_y, test_y

def regressionErrorScores(test_y, pred_y):
    # Calculate regression error metrics
    mae = mean_absolute_error(test_y, pred_y)
    mse = mean_squared_error(test_y, pred_y)
    rmse = mse ** 0.5  # Square root of MSE
    r2 = r2_score(test_y, pred_y)  # R² score (goodness of fit)
    return mae, mse, rmse, r2