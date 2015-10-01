# Pull from database
import pandas as pd
import requests
from sqlalchemy import *
import instaconfig
import pymysql as mdb

instagram, database, search_tags = instaconfig.config()

#engine = create_engine('mysql://%(user)s:%(pass)s@%(host)s' % database)
#result = engine.execute('use instagram')

con = mdb.connect(database['host'], database['user'], database['pass'], 'instagram') #host, user, password, #database

q = '''
    SELECT *
    FROM posts
    WHERE searched_tag IN 
    %s ;
    ''' % ("('" + "','".join(search_tags[:6]) + "')")  # :7

# Read database into pandas dataframe

df = pd.read_sql(q,con = con)

#result.close()
#engine.dispose()
