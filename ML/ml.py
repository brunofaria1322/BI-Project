"""

"""

__author__ = "Bruno Faria & Dylan Perdigão"
__date__ = "May 2022"


import psycopg2

def connect_to_db():
	return psycopg2.connect(
		user="bi2022",
		password="in2022$$",
		host="bi2022.postgres.database.azure.com",
		port="5432",
		database="postgres"
	)

def read_data():
    connection = connect_to_db()

    print(connection)

    cursor = connection.cursor()
    cursor.execute("SELECT * FROM auction")
    print(cursor.fetchall())
    cursor.close()

    connection.close()


def main():
    data = read_data()
    

if __name__ == "__main__":
    main()