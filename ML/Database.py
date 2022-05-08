import psycopg2


class Database(object):
    def __init__(self, user, password, host, port, database):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.connect()

    def connect(self):
        self.connection = psycopg2.connect(
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database
        )


    def listUserAuctions(self, username):
        """
        Lista os leilões em que a pessoa tenha atividade

        :param username: nome de utilizador

        :return: leilões
        """
        cursor = self.connection.cursor()
        getPersonId = 'SELECT person_id FROM participant WHERE person_username = %s'
        cursor.execute(getPersonId, (username,))
        persoId = cursor.fetchone()[0]
        cursor.execute(
            """SELECT distinct on (auction.id) auction.id, description, version
                FROM auction
                left join  bid on auction.id = bid.auction_id
                join textual_description on auction.id = textual_description.auction_id
                WHERE (bid.participant_person_id = -1 or auction.participant_person_id = -1)
                ORDER BY auction.id, version desc""",
            (persoId, persoId))
        if cursor.rowcount < 1:
            res = []
        else:
            res = [{"leilaoId": row[0], "descricao": row[1]} for row in cursor.fetchall()]
        cursor.close()
        return res

    

    def listAuctions(self, param):
        cursor = self.connection.cursor()
        # Verifica se em alguma versão existe a descrição a pesquisar
        checkTextualDescription = 'select distinct auction_id from auction, textual_description WHERE auction.id = textual_description.auction_id and (auction.code::text = %s OR lower(textual_description.description) like %s AND isactive = true)'
        cursor.execute(checkTextualDescription, (param, '%' + param.lower() + '%'))
        # Não há o parametro a pesquisar
        if cursor.rowcount < 1:
            cursor.close()
            return 'noResults'

        id_auction = cursor.fetchone()[0]

        lastDescriptions = 'SELECT distinct on (auction.id) auction.id, description FROM auction, textual_description WHERE auction.id = textual_description.auction_id AND auction_id = %s ORDER BY auction.id, version desc'
        cursor.execute(lastDescriptions, (id_auction,))
        res = [{"leilaoId": row[0], "descricao": row[1]} for row in cursor.fetchall()]
        cursor.close()
        return res

   
