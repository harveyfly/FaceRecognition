# encoding utf-8
# restful_api 数据库操作

import MySQLdb
import os
import numpy as np
from configparser import ConfigParser

# 人脸信息数据库类
class face_db:
    def __init__(self):
        cfg = ConfigParser()
        cfg.read('_config.ini')
        self.host = cfg.get('database', 'host')
        self.user = cfg.get('database', 'user')
        self.passwd = cfg.get('database', 'passwd')
        self.db_name = cfg.get('database', 'db_name')
        self.db_conn = None
        
    def conn_face_db(self):
        try:
            self.db_conn = MySQLdb.connect(self.host, self.user, self.passwd, self.db_name)
            print(self.db_name, " Connected!")
        except Exception:
            print("MySQL Error: ", Exception)
            return False
        return True
    
    def insert_face(self, face_token, face_dist, face_name='None', face_path='None'):
        if face_token is not None:
            cursor = self.db_conn.cursor()
            sqlInsert = "INSERT INTO face_set(face_token, face_dist, face_name, face_path) VALUES (%s, %s, %s, %s)"
            val = (face_token, face_dist.tostring(), face_name, face_path)
            print(sqlInsert, val)
            try:
                cursor.execute(sqlInsert, val)
                self.db_conn.commit()
                return True
            except Exception:
                print("Execuet Insert Error!")
                return False
        else:
            print("face_token is None, Insert Error")
        return False

    def delete_face(self, face_token):
        if face_token is not None:
            cursor = self.db_conn.cursor()
            sqlDelete = "DELETE FROM `face_set` WHERE `face_token`='%s'"
            try:
                cursor.execute(sqlDelete, face_token)
                self.db_conn.commit()
                return True
            except Exception:
                print("Execute Delete Error!")
                return False
        else:
            print("face_token is None, Delete Error!")
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
                print("get face_dist by face_token Error!", ex)
        else:
            print("face_token is None, Search Error")
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
            print("get face_dist_all Error!", ex)
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
                print("get face_path Error!", ex)
        else:
            print("face_token is None, Search Error")
        return None


## TEST
# faceDB = face_db()
# faceDB.conn_face_db()
# dist = faceDB.get_face_dist_all()
# print(dist)
