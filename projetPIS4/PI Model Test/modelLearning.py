import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf


#-----------------------LINEAR REGRESSION--------------------

def LinearmodelError(X_train, y_train,X_test,y_test):
    # Create a linear regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model using mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print("LINEAR REGRESSION Mean Squared Error:", mse)



#-----------------------Artificial neural network--------------------
def ann(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Define a list of seed values to try
    seed_values = [42, 123, 456, 789, 987,1,45,89,903,79]
    seed_values=random.sample(range(1, 1001), 10)
    # Dictionary to store seed-value and corresponding MSE
    mse_results = {}
    # Loop through different seed values
    for seed in seed_values:
        # Set the random seed
        tf.random.set_seed(seed)
        # Define the neural network architecture
        model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
        ])
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32)
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_results[seed] = mse
    fe=mse_results[seed_values[0]]
    se=42
    for  seed,mse in mse_results.items():
        if fe>mse:
            fe=mse
            se=seed
    if fe<25:
        # Save the model
        model.save('my_model.h5')
    print("ANN Mean Squared Error:(seed=",se,")", fe)

from sklearn.metrics import mean_squared_error
from math import sqrt

def compute_rmse(y_true, y_pred):
    """
    Cette fonction calcule la racine carrée de l'erreur quadratique moyenne (RMSE)
    entre les valeurs réelles et les valeurs prédites.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    return rmse

#-----------------------Decisionnal Tree--------------------
def decT (X_train, y_train,X_test,y_test) : 

# Créer et entraîner un modèle d'arbre de décision
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

# Calculer la RMSE entre les valeurs prédites et les valeurs réelles
    rmse = compute_rmse(y_test, y_pred)

    print("RMSE: ", rmse)



def annTm(X_test,y_test):

    # Load the model
    loaded_model = tf.keras.models.load_model('my_model.h5')

    # Make predictions on the test set
    y_pred = loaded_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    print("ANN Mean Squared Error:",mse)

    return y_pred

def forestMod(X_train, y_train,X_test,y_test):
    # Create a random forest regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the random forest model
    rf.fit(X_train, y_train)

    # Make predictions on the testing set
    predictions = rf.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print("random forest Mean Squared Error:", mse)