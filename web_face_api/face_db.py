# encoding utf-8
# restful_api database operation

import MySQLdb
import os
import numpy as np
import logging
import config

# face db class
class face_db:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        # load config file
        if config.has('database'):
            db_conf = config.get('database')
            if 'host' in db_conf and \
                'port' in db_conf and \
                'user' in db_conf and \
                'passwd' in db_conf and \
                'db_name' in db_conf:
                self.host = db_conf['host']
                self.port = db_conf['port']
                self.user = db_conf['user']
                self.passwd = db_conf['passwd']
                self.db_name = db_conf['db_name']
        else:
            self._logger.error("Can't get database config")
            return
        # save database connection
        self.db_conn = None
        
    def conn_face_db(self):
        try:
            self.db_conn = MySQLdb.connect(self.host, self.user, self.passwd, self.db_name)
            self._logger.info("Connect %s.%s database sucess!" % (self.host, self.db_name))
        except Exception as ex:
            self._logger.error("MySQL Error: %s" % str(ex))
            return False
        return True
    
    def insert_face(self, face_token, face_dist, face_name='None', face_path='None'):
        if face_token is not None:
            cursor = self.db_conn.cursor()
            sqlInsert = "INSERT INTO face_set(face_token, face_dist, face_name, face_path) VALUES (%s, %s, %s, %s)"
            val = (face_token, face_dist.tostring(), face_name, face_path)
            try:
                cursor.execute(sqlInsert, val)
                self.db_conn.commit()
                self._logger.info("Insert face_token:%s into face_token sucess" % face_token)
                return True
            except Exception:
                self._logger.error("MySQL Error: %s" % str(Exception))
                return False
        else:
            self._logger.error("face_token is None, Insert Error")
        return False

    def delete_face(self, face_token):
        if face_token is not None:
            cursor = self.db_conn.cursor()
            sqlDelete = "DELETE FROM `face_set` WHERE `face_token`='%s'"
            try:
                cursor.execute(sqlDelete, face_token)
                self.db_conn.commit()
                self._logger.info("Delete face_token:%s from face_set sucess" % face_token)
                return True
            except Exception as ex:
                self._logger.error("MySQL Error: %s" % str(ex))
                return False
        else:
            self._logger.error("face_token is None, Insert Error")
        return False

    def get_face_dist(self, face_token):
        if face_token is not None:
            cursor = self.db_conn.cursor()
            sqlSelect = "SELECT face_dist FROM face_set WHERE face_token='%s'" % face_token
            try:
                cursor.execute(sqlSelect)
                values = cursor.fetchall()
                face_dist = np.frombuffer(values[0][0], dtype=np.float32)
                return face_dist
            except Exception as ex:
                self._logger.error("MySQL Error: %s" % str(ex))
        else:
            self._logger.error("face_token is None, Search Error")
        return None

    def get_face_info_all(self):
        cursor = self.db_conn.cursor()
        sqlSelect = "SELECT face_token, face_dist, face_name, face_path FROM face_set"
        faces = []
        try:
            cursor.execute(sqlSelect)
            values = cursor.fetchall()
            for value in values:
                face = {}
                face['face_token'] = value[0]
                face['face_dist'] = np.frombuffer(value[1], dtype=np.float32)
                face['face_name'] = value[2]
                face['face_path'] = value[3]
                faces.append(face)
        except Exception as ex:
            self._logger.error("MySQL Error: %s" % str(ex))
        return faces

    def get_face_path(self, face_token):
        if face_token is not None:
            cursor = self.db_conn.cursor()
            sqlSelect = "SELECT face_path FROM face_set WHERE face_token='%s'" % face_token
            try:
                cursor.execute(sqlSelect)
                values = cursor.fetchall()
                face_path = values[0][0]
                return face_path
            except Exception as ex:
                self._logger.error("MySQL Error: %s" % str(ex))
        else:
            self._logger.error("face_token is None, Search Error")
        return None
