import psycopg2
import pandas as pd

def connect_to_db():
	connection = psycopg2.connect(
		user="bi2022",
		password="", # nao correr mais este script.... demora tanto
		host="bi2022.postgres.database.azure.com",
		port="5432",
		database="postgres"
	)
	return connection

def main():
	# dataset
	df = pd.read_csv('Dataset/satisfaction.csv', sep=';', header=0)
	print(df.head())

	# drop NA values
	df = df.dropna()

	# lowercase
	df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('/', '_')
	df['class'].str.lower()
	df['type_of_travel'].str.lower()
	df['gender'].str.lower()
	df['customer_type'].str.lower()
	print(df.head())

	# print unique ids
	old_ids = df['id']
	print(df.shape, len(old_ids.unique()))
	# print unique strings
	print(df['class'].unique())
	print(df['type_of_travel'].unique())
	print(df['gender'].unique())
	print(df['customer_type'].unique())
	print(df['satisfaction_v2'].unique())

	connection = connect_to_db()
	cursor = connection.cursor()
	# for each row of df
	for i in range(df.shape[0]):
		print(f"Writing: {i*100/df.shape[0]}%")
		try:
			# common
			old_id = df['id'][i]
			# person
			age = df['age'][i]
			gender = df['gender'][i]
			loyalty = df['customer_type'][i]
			person_id = hash(f"{age}{gender}{loyalty}")
			# satisfaction
			seat_comfort = df['seat_comfort'][i]
			time_convenience = df['departure_arrival_time_convenient'][i]
			food = df['food_and_drink'][i]
			gate_location = df['gate_location'][i]
			wifi_service = df['inflight_wifi_service'][i]
			entertainment = df['inflight_entertainment'][i]
			booking = df['ease_of_online_booking'][i]
			online_support = df['online_support'][i]
			onboard_service = df['on-board_service'][i]
			leg_room_service = df['leg_room_service'][i]
			baggage_handling = df['baggage_handling'][i]
			online_boarding = df['online_boarding'][i]
			checkin = df['checkin_service'][i]
			cleanliness = df['cleanliness'][i]
			satisfaction_id = hash(f"{seat_comfort}{time_convenience}{food}{gate_location}{wifi_service}{entertainment}{booking}{online_support}{onboard_service}{leg_room_service}{baggage_handling}{online_boarding}{checkin}{cleanliness}")
			# flight
			distance = df['flight_distance'][i]
			departure_delay = df['departure_delay_in_minutes'][i]
			arrival_delay = df['arrival_delay_in_minutes'][i]
			flight_id = hash(f"{distance}{departure_delay}{arrival_delay}")
			# ticket
			flight_class = df['class'][i]
			type_travel = df['type_of_travel'][i]
			ticket_id = hash(f"{flight_class}{type_travel}")
			# fact table
			overall_satisfaction = df['satisfaction_v2'][i]
			# insert into db
			cursor.execute("""
					INSERT INTO person (id, age, gender, loyalty) 
					VALUES (%s, %s, %s, %s)
					ON CONFLICT DO NOTHING;
					INSERT INTO satisfaction (id,seat_comfort,time_convenience,food,gate_location,wifi_service,entertainment,booking,online_support,onboard_service,leg_room_service,baggage_handling,online_boarding,checkin,cleanliness) 
					VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
					ON CONFLICT DO NOTHING;
					INSERT INTO flight (id,distance,departure_delay,arrival_delay) 
					VALUES (%s, %s, %s, %s)
					ON CONFLICT DO NOTHING;
					INSERT INTO ticket (id,flight_class,type_travel) 
					VALUES (%s, %s, %s)
					ON CONFLICT DO NOTHING;
					INSERT INTO fact (overall_satisfaction,old_id,ticket_id,flight_id,satisfaction_id,person_id) 
					VALUES (%s, %s, %s, %s, %s, %s)
					ON CONFLICT DO NOTHING;
				""",(str(person_id), str(age), str(gender), str(loyalty),
					str(satisfaction_id),str(seat_comfort),str(time_convenience),str(food),str(gate_location),str(wifi_service),str(entertainment),str(booking),str(online_support),str(onboard_service),str(leg_room_service),str(baggage_handling),str(online_boarding),str(checkin),str(cleanliness),
					str(flight_id),str(distance),str(departure_delay),str(int(arrival_delay)),
					str(ticket_id),str( flight_class),str(type_travel),
					str(overall_satisfaction),str(old_id),str(ticket_id),str(flight_id),str(satisfaction_id),str(person_id)))
		except Exception as e:
			print(e)
		connection.commit()
	cursor.close()
if __name__ == "__main__":
	main()

	