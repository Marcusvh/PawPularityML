from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc

def prepRegressionData(df, test_size=0.2, random_state=42):
    # Filter out rows where 'Human' column is 1
    df = df[df["Human"] == 0]
    
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return train_x, test_x, train_y, test_y

def trainAndPredictLinearRegression(train_x, train_y, test_x):
    reg = LinearRegression()
    reg.fit(train_x, train_y)
    pred_y = reg.predict(test_x)
    return pred_y, reg

def trainAndPredictLogisticRegression(train_x, train_y, test_x):
    model = LogisticRegression()
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    return pred_y, model

def regressionErrorScores(test_y, pred_y):
    # Calculate regression error metrics
    mae = mean_absolute_error(test_y, pred_y)
    mse = mean_squared_error(test_y, pred_y)
    rmse = mse ** 0.5  # Square root of MSE
    r2 = r2_score(test_y, pred_y)  # R² score (goodness of fit)
    return mae, mse, rmse, r2

def predictModelRegression(model, df):
    X = df.iloc[:, 1:] # filter out 'Id' column

    pred_y = model.predict(X)
    return pred_y

def crossValidationScores(model, X, y, cv=5):
    cv_scores = cross_val_score(model, X, y, cv=cv)
    return cv_scores

def calculateROCCurve(model, test_x, test_y):
    pred_prob = model.predict_proba(test_x)[:, 1]
    fpr, tpr, _ = roc_curve(test_y, pred_prob)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc