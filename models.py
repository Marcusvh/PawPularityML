from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import pandas as pd

def prepRegressionData(df, test_size=0.2, random_state=42):
    # Use all features except the target for X, and the target for y
    X = df.iloc[:, :-1]  # filter out 'Id' column
    y = df["target"]
    
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return train_x, test_x, train_y, test_y

def PredictLinearRegression(train_x, train_y, test_x):
    reg = LinearRegression()
    reg.fit(train_x, train_y)
    pred_y = reg.predict(test_x)
    return pred_y, reg

def PredictLogisticRegression(df):
    """Train a LogisticRegression model using all rows of the dataset."""
    X = df.drop(columns=["target"])  # Features
    y = df["target"]  # Target

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Train the model
    model = LogisticRegression(max_iter=5000, solver='saga')  # Increased iterations and changed solver
    model.fit(X_scaled, y)

    pred_y = model.predict(X_scaled)
    accuracy = accuracy_score(y, pred_y)
    return model, pred_y, accuracy

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

def prepGaussianNBData(df, features, target="Pawpularity"):
    # Ensure the provided features exist in the DataFrame
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise KeyError(f"The following features are missing from the DataFrame: {missing_features}")
    
    X = df[features]
    y = df[target]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

    return train_x, test_x, train_y, test_y

def trainGaussianNBWithSelectedFeatures(train_x, train_y, test_x):
    model = GaussianNB()
    model.fit(train_x, train_y)    
    pred_y = model.predict(test_x)
    
    gaussianNBexploreParameters(model)

    return pred_y

def gaussianNBexploreParameters(model):
    # This array represents the prior probabilities of each class (unique values in train_y).
    # GaussianNB assumes that each class has a probability before seeing the data.
    # These values sum to 1 and indicate how frequent each class is in the training data.
    print("Class Prior Probabilities: (model.class_prior_)")
    print(model.class_prior_)

    # Each row represents a different class, and each column represents a feature (Eyes, Face, Occlusion).
    # eg. Pawpularity = 0 => Eyes = 0.5, Face = 0.3, Occlusion = 0.2. for the probability of that class.
    print("Means of each feature for each class: (model.theta_)")
    print(model.theta_)

    # Variance of each feature for each class. This is used to calculate the Gaussian probability density function.
    # The variance indicates how much the feature varies for each class.
    print("Variance of each feature for each class: (model.var_)")
    print(model.var_) # sigma_ i bogen

def trainLogisticRegressionFullDataset(df):
    """Train a LogisticRegression model using all rows of the dataset."""
    X = df.drop(columns=["target"])  # Features
    y = df["target"]  # Target (benign or malignant)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Predict on the same dataset (for demonstration purposes)
    pred_y = model.predict(X)

    # Calculate accuracy
    accuracy = (pred_y == y).mean()
    return model, accuracy
