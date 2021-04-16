# -*- coding: utf-8 -*-
# @Time    : 2020/8/1 21:06
# @Author  : Sijia
# @Email   : guosijia007@yeah.net
# @File    : dbutils.py
# @Software: PyCharm

import json
import logging
import os
import sys

from pymongo import MongoClient

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class DBconnection(object):
    i = 0  # counter for the number of objects indexed

    def __init__(self, index, host=None, port=None, mdbcollection=None,
                 user=None, password=None, recreateindex=False):
        assert index is not None
        self.index = index
        if port is not None and not isinstance(port, int):
            port = int(port)
        try:
            # TODO: option to specify config file
            # cfgfile = "./dbservers.json"
            # if not os.path.exists(cfgfile):
            #     if os.path.exists("./dbservers.json"):
            #         cfgfile = "./dbservers.json"
            #     elif os.path.exists("../conf/dbservers.json"):
            #         cfgfile = "../conf/dbservers.json"
            #     else:
            #         cfgfile = "../../conf/dbservers.json"
            # logger.info("Servers configuration file: %s" % cfgfile)

            conf_path = '/home/cqfnenu/DrugPrediction/dbservers.json'
            print(conf_path)
            # with open(cfgfile, "r") as cfgf:
            # --------------------------------
            with open(conf_path, 'r') as cfgf:
                conf = json.load(cfgf)
                print(conf)
            # --------------------------------
        except IOError:
            conf = {"es_host": "localhost", "es_port": 9200,
                    "mongodb_host": "localhost", "mongodb_port": 27017}


        if host is None:
            host = conf['mongodb_host']
        if port is None and 'mongodb_port' in conf:
            port = conf['mongodb_port']

        mc = MongoClient(host, port)
        if user is None:
            user = conf['mongodb_user']
        if password is None:
            password = conf['mongodb_password']
        if user not in ["", None] and password not in ["", None]:
            # # ------------------------
            # user = conf['mongodb_user']
            # password = conf['mongodb_password']
            # # ------------------------
            db_auth = mc['admin']
            db_auth.authenticate(user, password)

        logger.info("New MongoDB connection: '%s:%d'" % (host, port))
        self.mdbi = mc[index]
        # print(self.mdbi)
        if mdbcollection is not None:
            self.mdbcollection = mdbcollection
            if recreateindex:
                self.mdbi.drop_collection(mdbcollection)

    # Prints '.' to stdout as indication of progress after 'n' entries indexed
    def reportprogress(self, n=160):
        self.i += 1
        if self.i % n == 0:
            print(".", end='')
            sys.stdout.flush()
            if self.i % (n * 80) == 0:
                print("{}".format(self.i))


def dbargs(argp, mdbdb='biosets', mdbcollection=None,
           multipleindices=False):
    """ Given ArgumentParser object, argp, add database arguments """
    argp.add_argument('--mdbdb',
                      default=mdbdb,
                      help='Name of the MongoDB database')
    if not multipleindices:
        argp.add_argument('--mdbcollection',
                          default=mdbcollection,
                          help='Collection name for MongoDB')
    argp.add_argument('--recreateindex',
                      default=False,
                      help='Delete existing Elasticsearch index or MongoDB collection')
    argp.add_argument('--host',
                      help='Elasticsearch or MongoDB server hostname')
    argp.add_argument('--port', type=int,
                      help="Elasticsearch or MongoDB server port number")
    argp.add_argument('--dbtype', default='Elasticsearch',
                      help="Database: 'Elasticsearch' or 'MongoDB'")
    argp.add_argument('--user',
                      help="Database user name, "
                           "supported with PostgreSQL option only")
    argp.add_argument('--password',
                      help="Password for the database user, "
                           " supported with PostgreSQL option only")


def test():
    db = DBconnection("drugbank")
    print(db)


if __name__ == '__main__':
    test()
