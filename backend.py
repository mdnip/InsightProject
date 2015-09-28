# Database Interaction
import pandas as pd
import requests
from sqlalchemy import *
import datetime

from pull_from_database import *

#%matplotlib inline
#%pylab inline

## Split dataframe into last 24 hours and prior
num_days_back = 23

cutoff_time1 = datetime.datetime.fromtimestamp(df['created_time'].max())-datetime.timedelta(hours = 24*num_days_back)
cutoff_time2 = cutoff_time1 - datetime.timedelta(hours = 24)
last_day_df = df[df['created_time'].apply(lambda x: (datetime.datetime.fromtimestamp(x) > cutoff_time2) & (datetime.datetime.fromtimestamp(x) <= cutoff_time1))].reset_index()
train_df = df[df['created_time'].apply(lambda x: datetime.datetime.fromtimestamp(x)) <= cutoff_time2].reset_index()
#train_df['cluster_label_2'] = '' #add empty column for sub cluster label

# Analytics packages
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics

from scipy.stats import beta
from scipy.stats import poisson

########################
print 'Performing DBSCAN first pass'

Xtrain = np.vstack((train_df.longitude, train_df.lat)).T
Xtrain *= np.pi/180

# Compute DBSCAN
db = DBSCAN(eps=2e-4,         #4e-5
            min_samples=5,
            metric='haversine'
           ).fit(Xtrain)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = list(db.labels_)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

train_df['cluster_label'] = pd.Series(labels)



print 'Identifying large clusters to split'
################################3

# This is actually an upper bound on the set radius
from geopy.distance import vincenty

# Filter spatial outliers
cluster_df = train_df[train_df['cluster_label'] != -1]

# Find cluster centroids in (lat, long) space
cluster_centroids = cluster_df[['cluster_label','lat','longitude']].groupby(['cluster_label']).mean().reset_index()
cluster_point_centroids = pd.merge(left = cluster_df[['cluster_label','lat','longitude']], right = cluster_centroids, how = 'inner', on = 'cluster_label', suffixes = ('_point','_centroid')).reset_index()

# Distance measure
def dist(x1,x2,y1,y2):
    #print x1, x2, y1, y2
    if sum(np.isnan([x1,x2,y1,y2])):
        return 0;
    else:
        return vincenty((x1,x2),(y1,y2)).miles

# Calculate distance of each point from centroid
cluster_point_centroids['distance_from_centroid'] = cluster_point_centroids.apply(lambda x: dist(x['lat_point'], x['longitude_point'], x['lat_centroid'], x['longitude_centroid']),axis = 1)

# Max over all distances from centroid
cluster_radius = cluster_point_centroids.groupby('cluster_label').max().reset_index()
cluster_radius = cluster_radius[['cluster_label','lat_centroid','longitude_centroid','distance_from_centroid']]


####################################

def run_KMeans(n_clusters, sub_cluster):

    Xtrain = np.vstack((sub_cluster.longitude, sub_cluster.lat)).T
    Xtrain *= np.pi/180

    # Compute KMeans
    km = KMeans(
                n_clusters = n_clusters, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                tol=0.0001, 
                precompute_distances='auto', 
                verbose=0, 
                random_state=None, 
                copy_x=True, 
                n_jobs=1
               ).fit(Xtrain)


    labels = list(km.labels_)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    sub_cluster = sub_cluster.reset_index()
    #print sub_cluster

    sub_cluster['cluster_label_2'] = pd.Series(labels)

    # Attach centroid coordinates to each element
    clus_cent = km.cluster_centers_ * 180 /np.pi
    sub_cluster_centroid = pd.DataFrame(np.hstack((np.arange(n_clusters).reshape(n_clusters,1),clus_cent)), columns = ['cluster_label_2','longitude_centroid','lat_centroid'])
    sub_cluster1 = pd.merge(sub_cluster, sub_cluster_centroid, how = 'left', on = 'cluster_label_2')

    # Calculate distance of each point from centroid
    sub_cluster1['distance_from_centroid'] = sub_cluster1.apply(lambda x: dist(x['lat'], x['longitude'], x['lat_centroid'], x['longitude_centroid']),axis = 1)

    # Max over all distances from centroid
    sub_cluster_radius = sub_cluster1.groupby('cluster_label_2').max().reset_index()
    #sub_cluster_radius = sub_cluster_radius[['cluster_label_2','lat_centroid','longitude_centroid','distance_from_centroid']]

    return np.median(sub_cluster_radius['distance_from_centroid'].values), labels, sub_cluster



#############################3
print 'Splitting large clusters'

large_clusters = cluster_radius[cluster_radius['distance_from_centroid'] > 10]['cluster_label'].values

mod_sub_clusters = []
for large_cluster in large_clusters:
    print 'Subdividing cluster %d' % large_cluster 
    sub_cluster = train_df[train_df['cluster_label'] == large_cluster]
    #print len(sub_cluster)

    ## Compute cluster
    threshhold = 1 # miles
    top_value = 40
    bot_value = 1
    out = threshhold + 1

    print 'Defining search space'
    max_iter_expand_search = 10
    
    expand_search_count = 0
    # expand search space
    while (out > threshhold) and (expand_search_count <= max_iter_expand_search):
        out, labels, mod_sub_cluster = run_KMeans(top_value, sub_cluster)
        if out > threshhold:
            bot_value = top_value
            top_value *= 10
            top_value = min([top_value,100])
        expand_search_count += 1

    #print top_value, bot_value

    print 'Backtracking!'
    # binary search (backtrace)
    while top_value - bot_value > 1:
        probe = int(round(np.mean([top_value, bot_value])))
        out, labels, mod_sub_cluster = run_KMeans(probe, sub_cluster)
        if out > threshhold:
            bot_value = probe
        else:
            top_value = probe
    
    #print len(mod_sub_cluster)
    #print mod_sub_cluster['cluster_label_2']
    mod_sub_clusters.append(mod_sub_cluster)
    print out, probe
    
mod_sub_clusters1 = pd.concat(mod_sub_clusters)[['index','cluster_label_2']]


#mod_sub_clusters1['cluster_label_2'] = mod_sub_clusters1.apply(lambda x: -1 if np.isnan(x['cluster_label_2']) else x['cluster_label_2'], axis = 1)

train_df1 = pd.merge(train_df, mod_sub_clusters1, how = 'left', on = 'index')


# Check if point should be assigned to cluster
import random

def check_spatial_membership(pointx,pointy):
    point = (pointx,pointy)
    try:
        matches = cluster_radius[cluster_radius.apply(lambda x: vincenty(point,(x['lat_centroid'],x['longitude_centroid'])) <= x['distance_from_centroid'], axis = 1)]['cluster_label'].reset_index()
        if len(matches) < 1:
            return -1
        elif len(matches) == 1:
            return int(matches['cluster_label'])
        else:
            print "Something odd may happen"
            return int(matches.ix[random.sample(matches.index,1)]['cluster_label'])
    except ValueError:
        return -1
  
 #######################################

# Identify time scales for binning
df1 = train_df1[['created_time','cluster_label','cluster_label_2']]
df1['day_of_week'] = train_df1['created_time'].apply(lambda x: int(datetime.datetime.fromtimestamp(x).weekday()))
df1['day_of_month'] = train_df1['created_time'].apply(lambda x: datetime.datetime.fromtimestamp(x).day)
df1['hour_of_day'] = train_df1['created_time'].apply(lambda x: datetime.datetime.fromtimestamp(x).hour)
df1['month'] = train_df1['created_time'].apply(lambda x: datetime.datetime.fromtimestamp(x).month)
df1['year'] = train_df1['created_time'].apply(lambda x: datetime.datetime.fromtimestamp(x).year)

# Bin by some number of hours
time_bin_hours = 4
df1['hour_of_day'] = df1['hour_of_day'].apply(lambda x : x / time_bin_hours)

arrival_times = df1.groupby(['year','month','day_of_month','hour_of_day','day_of_week','cluster_label','cluster_label_2']).count().reset_index()


# Compute occupancy probability PER CLUSTER
bins_in_day = 24/time_bin_hours
bins_in_week = 7*bins_in_day

occupancy = (arrival_times.reset_index().groupby(['day_of_week','hour_of_day','cluster_label']).count()['year']).reset_index()
day_bins = np.zeros([7,bins_in_day,n_clusters_])

# unpack occupancy
for ii in occupancy.index:
     day_bins[int(occupancy.loc[ii]['day_of_week'])][int(occupancy.loc[ii]['hour_of_day'])][int(occupancy.loc[ii]['cluster_label'])] = occupancy.loc[ii]['year']

l, u, md = np.zeros(bins_in_week*n_clusters_), np.zeros(bins_in_week*n_clusters_), np.zeros(bins_in_week*n_clusters_)
day_bins = day_bins.reshape([bins_in_week*n_clusters_])

# average
md = np.divide(day_bins, 243)

# MAP estimates per bin
for ii in xrange(0,bins_in_week*n_clusters_):
    a, b = day_bins[ii], 243-day_bins[ii]             # Default = 243
    alpha = .05                                       # leftover probability
    l[ii] = beta.ppf(alpha / 2.0, a=a, b=b)           # lower threshhold
    u[ii] = beta.ppf(1.0 - alpha / 2.0, a=a, b=b)     # upper threshhold
    md[ii] = np.divide(a - 1.0, a + b - 2.0)          # mode
    
    
    
# Bin by cluster and by time period down to x hour blocks
temp_bin = df1.groupby(['cluster_label','day_of_week','day_of_month','hour_of_day','month','year']).count().reset_index()

# Remove outliers and relabel column
temp_bin_no_spatial_outliers = temp_bin[temp_bin['cluster_label'] != -1]
temp_bin_no_spatial_outliers.rename(columns = {'created_time' : 'num_posts_per_time_slot'}, inplace = True)

posts_per_time_period = temp_bin_no_spatial_outliers.groupby(['num_posts_per_time_slot','day_of_week','hour_of_day']).count()['cluster_label'].reset_index()

def arrival_statistics(time_bin,day_of_week):
    dist_posts_per_time_period = posts_per_time_period[(posts_per_time_period['hour_of_day'] == time_bin) & (posts_per_time_period['day_of_week'] == day_of_week)]['cluster_label'].reset_index()['cluster_label']
    emp_dist = (dist_posts_per_time_period/dist_posts_per_time_period.sum()).values
    mu = np.dot(np.array(range(0,len(emp_dist))),emp_dist)
    return mu


post_threshhold = np.zeros([bins_in_day,7])
for ii in xrange(0,bins_in_day):
    for jj in xrange(0,7):
        post_threshhold[ii][jj] = np.ceil(poisson.ppf(0.9999, arrival_statistics(ii,jj)))
        
#print post_threshhold


df2 = last_day_df[['post_id','created_time','lat','longitude','stand_res_url']]

day_of_week = datetime.datetime.fromtimestamp(max(df2['created_time'])).weekday()

df2['created_time'] = df2['created_time'].apply(lambda x: int(datetime.datetime.fromtimestamp(x).hour))

# Bin by some number of hours
time_bin_hours = 4
df2['created_time'] = df2['created_time'].apply(lambda x : x / time_bin_hours)

df2['cluster_label'] = df2[['lat','longitude']].apply(lambda x: check_spatial_membership(x['lat'],x['longitude']), axis = 1)


arrivals_outliers = df2[df2['cluster_label'] == -1].reset_index()

arrivals_to_clusters = df2[df2['cluster_label'] != -1].groupby(['cluster_label','created_time']).count().reset_index()

def check_if_anomalous(day_of_week,time_bin, arrivals):
    if arrivals > post_threshhold[time_bin][day_of_week]:
        return True
    else:
        return False

arrivals_anomalous = arrivals_to_clusters[arrivals_to_clusters.apply(lambda x: check_if_anomalous(day_of_week,x['created_time'],x['post_id']), axis = 1)]



temporal_outliers = df2[df2['cluster_label'].isin(arrivals_anomalous['cluster_label'])]

print 'Spatial outliers:'
print arrivals_outliers['stand_res_url']

print 'Temporal outliers'
print temporal_outliers['stand_res_url']




#############################################3

# Drops one circle per cluster, does not include spatial outliers
import folium
#from IPython.display import IFrame

from geopy.geocoders import Nominatim

geolocator = Nominatim()
location = 'New York, New York'
geolocation = geolocator.geocode(location)

map_osm = folium.Map(
                    location=[geolocation.latitude,geolocation.longitude],
                    tiles='OpenStreetMap',
                    zoom_start = 13
                    )

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

labels = range(0,len(cluster_radius))

color_labels = [colors[label % 7] for label in labels]

def make_circle(lat,lon,image_url,radius,color_label): 
    map_osm.circle_marker(
        location = [lat,lon],
        radius=1000*radius,
        line_color=colors[0], #color_label,
        fill_color=colors[0], #color_label,
        popup = str(image_url)
        
        #'<img src={url} width=200 height=200><br>'.format(
        #    url=image_url)
    )
    
def mark_outlier(lat,lon,image_url,color_label): 
    map_osm.simple_marker(
        location = [lat,lon],
        #radius=5e2,
        #line_color=color_label, #color_label,
        #fill_color=color_label, #color_label,
        popup = '<img src={url} width=200 height=200><br>'.format(
            url=image_url)
    )

def mark_spatial_outlier(lat,lon,image_url):
    map_osm.simple_marker(
        location = [lat,lon],
        popup = '<img src={url} width=200 height=200><br>'.format(
            url=image_url)
    )
    
#map_osm.simple_marker

pics = cluster_radius[['lat_centroid','longitude_centroid','cluster_label','distance_from_centroid']].values

pics_out = temporal_outliers[['lat','longitude','stand_res_url']].values
pics_spatial_out = arrivals_outliers[['lat','longitude','stand_res_url']].values
    
make_circle_vec = np.vectorize(make_circle)
mark_outlier_vec = np.vectorize(mark_outlier)
mark_spatial_outlier_vec = np.vectorize(mark_spatial_outlier)

make_circle_vec(pics[:,0],pics[:,1],pics[:,2],pics[:,3],color_labels)
mark_outlier_vec(pics_out[:,0],pics_out[:,1],pics_out[:,2],'#000000')
mark_spatial_outlier_vec(pics_spatial_out[:,0],pics_spatial_out[:,1],pics_spatial_out[:,2])

map_osm.create_map(path='gsm.html')

#IFrame('gsm.html', 700, 700)