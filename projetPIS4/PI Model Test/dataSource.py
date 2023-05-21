import pandas as pd


def loadData():

    # Specify the path to your CSV file
    csv_file_path = 'C:\\Users\\asus\\Downloads\\IAM.PA (1).csv'

    # Read the CSV file into a DataFrame
    dataframe = pd.read_csv(csv_file_path)

    # Split the data into input features (X) and target variable (y)
    X = dataframe.drop('Close', axis=1)  # Drop the 'close' column from input features
    y = dataframe['Close']  # Assign the 'close' column to the target variable


    X['Date'] = pd.to_datetime(X['Date'])

    X['Day'] = X['Date'].dt.day
    X['Month'] = X['Date'].dt.month
    X['Year'] = X['Date'].dt.year

    return X,y

