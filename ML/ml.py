"""

"""

__author__ = "Bruno Faria & Dylan Perdigão"
__date__ = "May 2022"


import matplotlib.pyplot as plt
import numpy as np
from os.path import exists
import pandas as pd
import psycopg2
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2


def connect_to_db():
    """ 
    Connect to the PostgreSQL database server 

    Returns: 
        connection: a connection object
    """

    return psycopg2.connect(
		user="bi2022",
		password="in2022$$",
		host="bi2022.postgres.database.azure.com",
		port="5432",
		database="postgres"
	)


def read_data():
    """
    Reads the data from pickle, if exists, or from the database

    Returns:
        df: a dataframe with the data
    """

    if exists("ML/data.pkl"):
        print("Reading data from pickle...")
        df = read_data_from_pickle()
    else:
        print("Reading data from database...")
        df = read_data_from_db()

        #saves to pickle
        print("Saving data to pickle...")
        df.to_pickle("ML/data.pkl")

    return df

def read_data_from_pickle():
    """
    Read the data from the pickle file

    Returns:
        data: a dataframe with the data
    """
    
    return pd.read_pickle("ML/data.pkl")

def read_data_from_db():
    """
    Read the data from the database

    Returns:
        data: a dataframe with the data
    """
    connection = connect_to_db()


    sql_query = pd.read_sql_query ('''
                                SELECT *
                                FROM fact, flight, person, satisfaction, ticket
                                    WHERE fact.flight_id = flight.id
                                        AND fact.person_id = person.id
                                        AND fact.satisfaction_id = satisfaction.id
                                        AND fact.ticket_id = ticket.id;
                               ''', connection)

    df = pd.DataFrame(sql_query)

    connection.close()

    return df

def treat_data(df):
    """
    Treats the data

    Parameters:
        df: a dataframe with the data

    Returns:
        df: a dataframe with the treated data
    """
    
    #print(df.columns)
    #print(df.shape)

    ## Remove columns with ids
    cols = [c for c in df.columns if c.lower()[-2:] != 'id']
    df=df[cols]

    #print(df.columns)
    #print(df.shape)


    ## Change object columns to int
    #print(df.dtypes)

    # Check conflicting columns and object values
    print('Non-int columns')
    for col in df.columns:
        if df[col].dtypes == 'object':
            print(col, df[col].unique())

    # Replace values with int
    df.loc[:,'overall_satisfaction'] = df['overall_satisfaction'].replace({'satisfied':1, 'neutral or dissatisfied':0})
    
    df.loc[:,'gender'] = df['gender'].replace({'Male':1, 'Female':0})
    df.loc[:,'loyalty'] = df['loyalty'].replace({'Loyal Customer':1, 'disloyal Customer':0})
    df.loc[:,'flight_class'] = df['flight_class'].replace({'Eco':2, 'Business':1, 'Eco Plus':0})
    df.loc[:,'type_travel'] = df['type_travel'].replace({'Personal Travel':1,'Business travel':0})
    
    #print(df.dtypes)

    return df

def visualize_data(df, img_path):
    """
    Visualizes the data
    
    Parameters:
        df: a dataframe with the data
        img_path: the path to the images folder
    """

    print("===== Data Head =====")
    print(df.head)
    print("===== Data Description =====")
    print(df.describe())

    #print("===== Data y =====")

    ## Overall satisfaction Counts
    plt.figure(figsize=(10,5))
    ax = sns.countplot(x="overall_satisfaction", data=df)
    for p in ax.patches:
        ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=18)
    
    plt.savefig(img_path+"overall_satisfaction_counts.png")
    #plt.show()

    ## Correlation matrix
    correlations = df.corr(method='pearson')
    
    # heatmap of correlations
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(df.columns),1)
    #ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    #ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.columns)
    plt.savefig(img_path+"correlation_matrix.png")
    #plt.show()

    ## Histograms
    df.hist()
    plt.savefig(img_path+"histgrams.png")
    #plt.show()

    ## Density Plots
    df.plot(kind='density', subplots=True, layout=(5,5), sharex=False, sharey=False)
    plt.savefig(img_path+"density.png")
    plt.show()


def feature_selection(df):
    """
    Performs feature selection

    Parameters:
        df: a dataframe with the data

    Returns:
        us:
    """

    Y=df['overall_satisfaction']
    X=df.drop(['overall_satisfaction'], axis=1)


    ## Univariate Statistical Tests (Chi-squared for classification)
    test= SelectKBest(score_func=chi2, k=4)
    fit = test.fit(X, Y)
    #print(fit)

    # sumarize scores
    #print(fit.scores_)
    us = fit.transform(X)

    # summarize selected features
    print(us[0:5,:])


    rfe = pca = fi = 0

    return [us, rfe, pca, fi]


def main():
    pd.set_option('precision', 3)
    plt.rcParams.update({'font.size': 8})
    IMG_PATH = "ML/img"

    data = read_data()
    #print(data.shape)

    data = treat_data(data)
    #print(data.shape)

    visualize_data(data,IMG_PATH)


    [us, rfe, pca, fi] = feature_selection(data)



if __name__ == "__main__":
    main()
