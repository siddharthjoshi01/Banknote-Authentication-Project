############################################################################################
# The following program determines the correlations between distinct variables as obtained #
# from the csv data file. Based on which, optimal variables are suggested to maximize the  #
# prediction accuracy of the machine learning model.                                       #
############################################################################################

import numpy as np                      # needed for arrays and math
import pandas as pd                     # needed to read the data
import seaborn                          # data visualization
import matplotlib.pyplot as plt         # used for plotting


# function to calculate and determine correlations
def corr(datframe, num):
    matrix = datframe.corr()                                                                # determining the correlations
    matrix = matrix * np.tri(*matrix.values.shape, k=-1).T
    matrix = matrix.stack()
    matrix = matrix.reindex(matrix.abs().sort_values(ascending=False).index).reset_index()
    matrix.columns = ["Var 1", "Var 2", "correlation"]
    print("\n Correlations: \n")
    print(matrix.head(num))


# function to generate and design the correlation plot
def corrmat(data):
    plot1 = plt.figure(figsize=(10, 10), dpi=200)                                            # generating a figure with given dimensions (10*10 and with 200 dpi density
    sub1 = plot1.add_subplot()                                                               # adding a subplot to the plot1 generated above
    map = sub1.imshow(np.abs(data.corr()), interpolation='nearest')                          # mapped absolute of correlated data and plotted it in subplot created above using nearest interpolation

    size = len(data.columns)                                                                 # defining plot
    tick = np.arange(0, size, 1)                                                             # creating ticks as required
    sub1.set_xticks(tick)
    sub1.set_yticks(tick)
    sub1.set_xticklabels(data.columns)                                                       # setting X and Y labels
    sub1.set_yticklabels(data.columns)
    sub1.grid()                                                                              # enabling grid for the plot
    plt.title('Correlation Plot')                                                            # defining plot title

    plot1.colorbar(map, ticks=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])               # adding colorbar for correlation intensity intuition
    plt.show()


# function to generate pair plots between subsequent features
def corrplot(df):
    seaborn.pairplot(df, height=3)                                                           # generating correlation plots
    plt.show()


banknotes = pd.read_csv('..\\resources\\data_banknote_authentication.txt')                                  # reading the data file
banknotes.columns=['variance', 'skewness', 'curtosis', 'entropy', 'class']
print('Here are the first 5 observations: \n', banknotes.head(5))                            # printing the first 5 observations

corr(banknotes, 5)                                                                           # highest correlated list
corrmat(banknotes)                                                                           # generate the covariance heat plot
corrplot(banknotes)                                                                          # plotting pairs
print(banknotes.describe())
