from flask import render_template, request
from app import app
import pymysql as mdb
from a_Model import ModelIt

from geopy.geocoders import Nominatim

geolocator = Nominatim()

db = mdb.connect(user="root", host="localhost", db="world", charset='utf8')

@app.route('/')
@app.route('/index')
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

@app.route('/cover', methods = ['GET'])
def cover_page():
    # Displays cover page
    
    location = request.args.get('location')
    distance = request.args.get('distance')
    
    geolocation = geolocator.geocode(location)
    
    print geolocation.latitude, geolocation.longitude, 
    
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
    geolocation = geolocator.geocode(location)
    
    return render_template('map.html',
                           title = 'busker.ly',
                           cover_heading = ' busker.ly',
                           lead = 'Latitude: %s, Longitude: %s' % (geolocation.latitude, geolocation.longitude),
                           masthead_brand = 'busker.ly')










