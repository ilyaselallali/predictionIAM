import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def CloseOverTime(X,y):
    #close price over time
    plt.plot(X['Date'], y, color='blue')
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.title('Close Price Over Time')
    plt.show()

def CloseValueFrequency(y):
    #frequency of close price values
    plt.hist(y, bins=20)
    plt.xlabel('Close')
    plt.ylabel('Frequency')
    plt.title('Distribution of Close Prices')
    plt.show()

def CloseVsOpen(X,y):
    # Scatter plot for 'Close' vs 'Open'
    plt.scatter(X['Open'], y)
    plt.xlabel('Open')
    plt.ylabel('Close')
    plt.title('Close vs Close')
    plt.show()



