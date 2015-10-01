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
from pull_from_database import *

from sklearn.cluster import DBSCAN
from sklearn import metrics

#instagram, database, search_tags = instaconfig.config()

geolocator = Nominatim()

#engine = create_engine('mysql://%(user)s:%(pass)s@%(host)s' % database)
#engine.execute('use instagram')

#con = mdb.connect(database['host'], database['user'], database['pass'], 'instagram') #host, user, password, #database



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


@app.route('/')
@app.route('/index')
@app.route('/cover', methods = ['GET'])
def cover_page():
    # Displays cover page  
      
    
    return render_template('cover.html', 
                           title = 'buskerbot.xyz', 
                           cover_heading = 'buskerbot.xyz', 
                           lead = 'Find streetmusicians near you', 
                           masthead_brand = 'buskerbot.xyz')

@app.route('/map', methods = ['GET'])
def map_page():
    # Display map page
    location = request.args.get('location')
    distance = request.args.get('distance')
    
    '''if location == '':
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

    map_osm.create_map(path='app/templates/osm.html')'''
    
    
    return render_template('gsm.html')








