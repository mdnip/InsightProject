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

geolocator = Nominatim()

#from backend import *


'''
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

'''
toolbar_vars = {'label1' : 'Home',
                'label2' : 'Details',
                'label3' : 'Contact',
                'link1' : '/index',
                'link2' : 'https://docs.google.com/presentation/d/1Q0dR91rene4724SfFQO7aMZDhTGU8inJMhAYVkoHU9c/pub?start=false&loop=false&delayms=3000',
                'link3' : 'https://mdnip.wordpress.com/not-work/'}

print('Ready')


@app.route('/')
@app.route('/index')
@app.route('/cover', methods = ['GET'])
def cover_page():
    # Displays cover page  
      
    
    return render_template('index.html', 
                           title = 'buskerbot.xyz', 
                           cover_heading = 'buskerbot.xyz', 
                           lead = 'Find streetmusicians near you', 
                           masthead_brand = 'buskerbot.xyz',
                           toolbar_link_1 = toolbar_vars['link1'],
                           toolbar_link_2 = toolbar_vars['link2'],
                           toolbar_link_3 = toolbar_vars['link3'],
                           toolbar_label_1 = toolbar_vars['label1'],
                           toolbar_label_2 = toolbar_vars['label2'],
                           toolbar_label_3 = toolbar_vars['label3'],)

@app.route('/map', methods = ['GET'])
def map_page():
    # Display map page
    location = request.args.get('location')
    #distance = request.args.get('distance')
    
    print location
    
    if location == '':
        location = 'New Orleans, Louisiana'
    
    geolocation = geolocator.geocode(location)
    
    '''map_osm = folium.Map(location=[geolocation.latitude, geolocation.longitude],
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
    
    
    
    
    return render_template('map.html',
                           gsm_lat = geolocation.latitude,
                           gsm_lon = geolocation.longitude,
                           toolbar_link_1 = toolbar_vars['link1'],
                           toolbar_link_2 = toolbar_vars['link2'],
                           toolbar_link_3 = toolbar_vars['link3'],
                           toolbar_label_1 = toolbar_vars['label1'],
                           toolbar_label_2 = toolbar_vars['label2'],
                           toolbar_label_3 = toolbar_vars['label3'],
                           render_map = 'gsm.html'
                          )

@app.route('/folium', methods = ['GET'])
def folium_map():
    location = request.args.get('location')
    distance = request.args.get('distance')
    
    print location
    
    if location == None:
        location = 'New Orleans, Louisiana'
    try:
        geolocation = geolocator.geocode(location)
    except:
        class geoc:
            latitude = 0
            longitude = 0
        
        geolocation = geoc()
    
    
        
    return render_template('gsm.html',
                          gsm_lat = geolocation.latitude,
                          gsm_lon = geolocation.longitude)








