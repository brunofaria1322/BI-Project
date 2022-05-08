"""

"""

__author__ = "Bruno Faria & Dylan Perdigão"
__date__ = "May 2022"

from os.path import exists
import pandas as pd
import psycopg2

import time

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

    if exists("data.pkl"):
        print("Reading data from pickle...")
        df = read_data_from_pickle()
    else:
        print("Reading data from database...")
        df = read_data_from_db()

        #saves to pickle
        print("Saving data to pickle...")
        df.to_pickle("data.pkl")

    return df

def read_data_from_pickle():
    """
    Read the data from the database

    Returns:
        data: a dataframe with the data
    """
    
    return pd.read_pickle("data.pkl")

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


def main():

    data = read_data()
    print(data.shape)

if __name__ == "__main__":
    main()