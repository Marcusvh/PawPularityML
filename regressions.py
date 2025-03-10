from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def prepAndTrainCSVForLinearRegression(df):

    # Prepare X and y
    X = df.iloc[:,1:-1]
    y = df.iloc[:,-1]

    # Do train test split
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = LinearRegression()
    reg.fit(train_x, train_y)

    pred_y = reg.predict(test_x)
    return train_x, test_x, train_y, test_y, pred_y