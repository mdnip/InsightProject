# Pull from database
import pandas as pd
import requests
from sqlalchemy import *
import instaconfig

instagram, database, search_tags = instaconfig.config()

engine = create_engine('mysql://%(user)s:%(pass)s@%(host)s' % database)
result = engine.execute('use instagram')

q = '''
    SELECT *
    FROM posts
    WHERE searched_tag IN 
    %s ;
    ''' % ("('" + "','".join(search_tags[:6]) + "')")  # :7

# Read database into pandas dataframe

df = pd.read_sql_query(q,con = engine)

result.close()
engine.dispose()