DROP TABLE IF EXISTS person CASCADE;
DROP TABLE IF EXISTS satisfaction CASCADE;
DROP TABLE IF EXISTS flight CASCADE;
DROP TABLE IF EXISTS ticket CASCADE;
DROP TABLE IF EXISTS fact CASCADE;

CREATE TABLE person (
	id	 		BIGINT,
	age	 		INTEGER NOT NULL,
	gender	 	VARCHAR(20) NOT NULL,
	loyalty	 	VARCHAR(20) NOT NULL,
	PRIMARY KEY(id)
);

CREATE TABLE satisfaction (
	id	 				BIGINT,
	seat_comfort	 	SMALLINT NOT NULL,
	time_convenience 	SMALLINT NOT NULL,
	food		 		SMALLINT NOT NULL,
	gate_location	 	SMALLINT NOT NULL,
	wifi_service	 	SMALLINT NOT NULL,
	entertainment	 	SMALLINT NOT NULL,
	booking		 		SMALLINT NOT NULL,
	online_support	 	SMALLINT NOT NULL,
	onboard_service	 	SMALLINT NOT NULL,
	leg_room_service 	SMALLINT NOT NULL,
	baggage_handling 	SMALLINT NOT NULL,
	online_boarding	 	SMALLINT NOT NULL,
	checkin		 		SMALLINT NOT NULL,
	cleanliness	 		SMALLINT NOT NULL,
	PRIMARY KEY(id)
);

CREATE TABLE flight (
	id	 				BIGINT,
	distance	 		BIGINT NOT NULL,
	departure_delay 	BIGINT,
	arrival_delay	 	BIGINT,
	PRIMARY KEY(id)
);

CREATE TABLE ticket (
	id			 		BIGINT,
	flight_class 		VARCHAR(20) NOT NULL,
	type_travel	 		VARCHAR(20) NOT NULL,
	PRIMARY KEY(id)
);

CREATE TABLE fact (
	ticket_id		 		BIGINT,
	flight_id		 		BIGINT,
	satisfaction_id 		BIGINT,
	person_id		 		BIGINT,
	old_id			 		BIGINT,
	overall_satisfaction 	VARCHAR(50) NOT NULL,
	PRIMARY KEY(ticket_id,flight_id,satisfaction_id,person_id,old_id)
);

ALTER TABLE fact ADD CONSTRAINT fact_fk1 FOREIGN KEY (ticket_id) REFERENCES ticket(id);
ALTER TABLE fact ADD CONSTRAINT fact_fk2 FOREIGN KEY (flight_id) REFERENCES flight(id);
ALTER TABLE fact ADD CONSTRAINT fact_fk3 FOREIGN KEY (satisfaction_id) REFERENCES satisfaction(id);
ALTER TABLE fact ADD CONSTRAINT fact_fk4 FOREIGN KEY (person_id) REFERENCES person(id);
