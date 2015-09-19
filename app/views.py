from flask import render_template, request
from app import app
import pymysql as mdb
from a_Model import ModelIt

import numpy as np
import pandas as pd
import folium
from geopy.geocoders import Nominatim

from sqlalchemy import *
import datetime
import instaconfig
from pytz import timezone


from sklearn.cluster import DBSCAN
from sklearn import metrics

instagram, database, search_tags = instaconfig.config()

geolocator = Nominatim()

engine = create_engine('mysql://%(user)s:%(pass)s@%(host)s' % database)
engine.execute('use instagram_master')

q = '''
    SELECT *
    FROM posts
    WHERE searched_tag IN 
    %s ;
    ''' % ("('" + "','".join(search_tags[:7]) + "')")  # :7

print q

df = pd.read_sql_query(q,con = engine)

db = mdb.connect(user="root", host="localhost", db="world", charset='utf8')

##############################################################################
# Compute DBSCAN
Xtrain = np.vstack((df.longitude, df.lat)).T
Xtrain *= np.pi/180

db = DBSCAN(eps=1e-5, min_samples=3).fit(Xtrain)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

colors = [
    '#d73027',
    '#f46d43',
    '#fdae61',
    '#fee090',
    '#e0f3f8',
    '#abd9e9',
    '#74add1',
    '#4575b4'
]

color_labels = [colors[label % 7] for label in labels]

print('Ready')


#@app.route('/')
#@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Miguel' },
       )

@app.route('/db')
def cities_page():
    with db:
        cur = db.cursor()
        cur.execute("SELECT Name FROM City LIMIT 15;")
        query_results = cur.fetchall()
    cities = ""
    for result in query_results:
        cities += result[0]
        cities += "<br>"
    return cities

@app.route("/db_fancy")
def cities_page_fancy():
    with db:
        cur = db.cursor()
        cur.execute("SELECT Name, CountryCode, Population FROM City ORDER BY Population LIMIT 15;")

        query_results = cur.fetchall()
    cities = []
    for result in query_results:
        cities.append(dict(name=result[0], country=result[1], population=result[2]))
    return render_template('cities.html', cities=cities)

@app.route('/input')
def cities_input():
    return render_template("input.html")

@app.route('/output')
def cities_output():
    #pull 'ID' from input field and store it
    city = request.args.get('ID')

    with db:
        cur = db.cursor()
        #just select the city from the world_innodb that the user inputs
        cur.execute("SELECT Name, CountryCode,  Population FROM City WHERE Name='%s';" % city)
        query_results = cur.fetchall()

    cities = []
    for result in query_results:
        cities.append(dict(name=result[0], country=result[1], population=result[2]))
    #call a function from a_Model package. note we are only pulling one result in the query
    pop_input = cities[0]['population']
    the_result = ModelIt(city, pop_input)
    return render_template("output.html", cities = cities, the_result = the_result)

@app.route('/')
@app.route('/index')
@app.route('/cover', methods = ['GET'])
def cover_page():
    # Displays cover page  
      
    
    return render_template('cover.html', 
                           title = 'busker.ly', 
                           cover_heading = 'busker.ly', 
                           lead = 'Find streetmusicians near you', 
                           masthead_brand = 'busker.ly')

@app.route('/map', methods = ['GET'])
def map_page():
    # Display map page
    location = request.args.get('location')
    distance = request.args.get('distance')
    
    if location == '':
        location = 'New Orleans, Louisiana'
    
    geolocation = geolocator.geocode(location)
    
    map_osm = folium.Map(location=[geolocation.latitude, geolocation.longitude],
                    tiles='OpenStreetMap')

    pics = df[['lat','longitude','stand_res_url']].values

    def make_circle(lat,lon,image_url,color_label): 
        map_osm.circle_marker(
            location = [lat,lon],
            radius=100,
            line_color=color_label,
            fill_color=color_label,
            popup = '<img src={url} width=200 height=200><br>'.format(
                url=image_url)
        )

    make_circle_vec = np.vectorize(make_circle)

    make_circle_vec(pics[:,0],pics[:,1],pics[:,2],color_labels)

    map_osm.create_map(path='app/templates/osm.html')
    
    
    return render_template('osm.html')








