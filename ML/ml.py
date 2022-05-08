"""

"""

__author__ = "Bruno Faria & Dylan Perdigão"
__date__ = "May 2022"

from os.path import exists
import pandas as pd
import psycopg2

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
    print(df.dtypes)

    # Check conflicting columns and object values
    for col in df.columns:
        if df[col].dtypes == 'object':
            print(col, df[col].unique())

    # Replace values with int
    df.loc[:,'overall_satisfaction'] = df['overall_satisfaction'].replace({'satisfied':1, 'neutral or dissatisfied':0})
    
    df.loc[:,'gender'] = df['gender'].replace({'Male':1, 'Female':0})
    df.loc[:,'loyalty'] = df['loyalty'].replace({'Loyal Customer':1, 'disloyal Customer':0})
    df.loc[:,'flight_class'] = df['flight_class'].replace({'Eco':2, 'Business':1, 'Eco Plus':0})
    df.loc[:,'type_travel'] = df['type_travel'].replace({'Personal Travel':1,'Business travel':0})
    
    print(df.dtypes)

    return df



def main():

    data = read_data()
    print(data.shape)

    data = treat_data(data)
    print(data.shape)



if __name__ == "__main__":
    main()