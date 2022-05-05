import psycopg2
import pandas as pd

df = pd.read_csv('Dataset/satisfaction.csv', sep=';', header=0)
print(df.head())

person_df = df[['id','Age','Gender', 'Customer Type']]
print(person_df.head())

ticket_df = df[['id', 'Class', 'Type of Travel']]
print(ticket_df.head())



connection = psycopg2.connect(
        user="bi2022",
        password="in2022$$",
        host="bi2022.postgres.database.azure.com",
        port="5432",
        database="postgres"
)
cursor = connection.cursor()

cursor.execute("SELECT * FROM ticket;")
ticket_query = cursor.fetchall()
#print(ticket_query)

cursor.execute("SELECT * FROM person;")
person_query = cursor.fetchall()
#print(person_query)

for i in range(len(df['id'])):
        cursor.execute("""
                select  person.id, 
                        --flight_id, 
                        --satisfaction_id, 
                        ticket.id
                        --overall_satisfaction
                from    person,
                        flight,
                        satisfaction,
                        ticket
                where   person.age=%s and
                        person.gender=%s and
                        person.loyalty=%s and
                        flight_class=%s and
                        type_travel=%s
        """, (str(df['Age'][i]),str(df['Gender'][i]), str(df['Customer Type'][i]), str(df['Class'][i]), str(df['Type of Travel'][i])))
        res = cursor.fetchall()
        print(res)
cursor.close()
"""
cursor.execute(""
                INSERT INTO participant (person_username, person_email, person_password)
                VALUES (%s,%s,%s);
                "", (username, email, password)
                )
        
        #res = cursor.fetchone()[0]
        #cursor.close()
        #self.connection.commit()



"""
