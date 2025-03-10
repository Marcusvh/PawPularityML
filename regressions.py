from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

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


def prepAndTrainRegression(df):
    # Prepare X and y
    X = df.iloc[:,1:-1]
    y = df.iloc[:,-1]

    # Do train test split
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    return train_x, test_x, train_y, test_y