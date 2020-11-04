import matplotlib.pyplot as plt
import pandas as pd


def read_file(file):
    dataframe = pd.read_csv(file, delimiter=',')
    return dataframe


def get_plot(dataframe):
    dataframe.groupby(['Sex', 'Survived']).size().unstack().plot(kind='bar', stacked=True)
    plt.savefig('plot.png')


df = read_file('titanic.csv')

get_plot(df)
