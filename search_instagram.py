import json
import ast
import pandas as pd
import requests
import time
import traceback
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import datetime
import instaconfig

instagram, database, search_tags = instaconfig.config()

              

# Setup Tables and ORM

engine = create_engine('mysql://%(user)s:%(pass)s@%(host)s' % database) # connect to server
engine.execute('drop database %s' % database['name'])
engine.execute('create database %s' % database['name'])

# Select Database

engine.execute('use %s' % database['name']) # select db

# Build ORM
Base = declarative_base()
class post(Base):
    __tablename__ = 'posts'
    searched_tag = Column(String)
    created_time = Column(Integer)
    post_id = Column(String, primary_key=True)
    image_url = Column(String)
    lat = Column(Float)
    likes = Column(Integer)
    longitude = Column(Float)
    post_url = Column(String)
    text = Column(String)
    user_id = Column(Integer)
    date_time = Column(Integer)
    date_year = Column(Integer)
    date_month = Column(Integer)
    date_day = Column(Integer)
    day_of_week = Column(Integer)
    media_type = Column(String)
    low_res_url = Column(String)
    thumbnail_url = Column(String)
    stand_res_url = Column(String)
    live_music = Column(Boolean)
    genre = Column(String)

    def __repr__(self):
        return "<posts(id='%s', image_url='%s', likes='%s')>" % (self.post_id, self.image_url, self.likes)

class tag(Base):
    __tablename__ = 'tags'
    id = Column(Integer, primary_key = True)
    post_id = Column(String)
    tag = Column(String, ForeignKey('posts.post_id'))

    def __repr__(self):
        return "<posts(post_id='%s', tag='%s')>" % (self.post_id, self.tag)

class comment(Base):
    __tablename__ = 'comments'
    id = Column(Integer, primary_key = True)
    post_id = Column(String)
    text = Column(String, ForeignKey('posts.post_id'))

    def __repr__(self):
        return "<posts(post_id='%s', comment='%s')>" % (self.post_id, self.text)


# Build Tables
metadata = MetaData()

posts = Table('posts', metadata,
              Column('searched_tag',VARCHAR(255)),
              Column('created_time', BIGINT),
              Column('post_id', VARCHAR(255), primary_key = True, nullable = False),
              Column('image_url', VARCHAR(255)),
              Column('lat', FLOAT(53)),
              Column('likes', INT),
              Column('longitude', FLOAT(53)),
              Column('post_url', VARCHAR(255)),
              Column('text', VARCHAR(5000)),
              Column('user_id', BIGINT),
              Column('date_time', BIGINT),
              Column('date_year', INT),
              Column('date_month', INT),
              Column('date_week', INT),
              Column('date_day', INT),
              Column('day_of_week', INT),
              Column('media_type', VARCHAR(255)),
              Column('low_res_url', VARCHAR(255)),
              Column('thumbnail_url', VARCHAR(255)),
              Column('stand_res_url', VARCHAR(255)),
              Column('live_music', BOOLEAN),
              Column('genre', VARCHAR(255))
             )

tags = Table('tags', metadata,
             Column('id', BIGINT, primary_key = True, nullable = False),
             Column('post_id', VARCHAR(255), ForeignKey("posts.post_id"), nullable=False),
             Column('tag', VARCHAR(255))
            )

comments = Table('comments', metadata,
             Column('id',BIGINT, primary_key = True, nullable = False),
             Column('post_id', VARCHAR(255), ForeignKey("posts.post_id"), nullable=False),
             Column('text', VARCHAR(5000))
            )

Index("location_index", posts.c.lat, posts.c.longitude)
Index("date_time_index", posts.c.date_time)
Index("date_year_index", posts.c.date_year)
Index("date_month_index", posts.c.date_month)
Index("date_week_index", posts.c.date_week)
Index("date_day_index", posts.c.date_day)
Index("date_day_of_week_index", posts.c.day_of_week)
Index("searched_tag_index", posts.c.searched_tag)

metadata.create_all(engine)

# Open session, commit, close
Session = sessionmaker(bind=engine)
session = Session()

def search_instagram_tag(tag_name):
    #engine.execute('use %s_master' % database['name'])
    
    # Query Instagram API   MODIFY THIS IF YOU WANT TO CONNECT TO DIFFERENT ENDPOINT
    query_url = 'https://api.instagram.com/v1/tags/%s/media/recent?access_token=%s' % (tag_name,instagram['access_token'])
    resp = requests.get(query_url)

    #Set pull time period from current time back to now - scrape_time_length
    scrape_time_length = datetime.timedelta(days = 1095)
    date_time = datetime.datetime.now()
    end_time = date_time - scrape_time_length
    start = time.time()

    while date_time > end_time:
        # throttle requests to 3600/hour
        end = time.time()
        time.sleep(1)

        for insta_post in resp.json()['data']:

            try:
                if not insta_post['id'].encode('ascii','replace') in visited_ids:
                    visited_ids.add(insta_post['id'].encode('ascii','replace'))

                    if insta_post['location'] != None:
                        if insta_post['location']['latitude'] != None and insta_post['location']['longitude'] != None:
                            date_time = datetime.datetime.fromtimestamp(
                                int(insta_post['caption']['created_time'])
                            )

                            post_attr = {
                                'searched_tag' : tag_name,
                                'created_time' : int(insta_post['caption']['created_time']),
                                'post_id' : insta_post['id'].encode('ascii','replace'),
                                'image_url' : None,
                                'lat' : insta_post['location']['latitude'],
                                'likes' : insta_post['likes']['count'],
                                'longitude' : insta_post['location']['longitude'],
                                'post_url' : insta_post['link'].encode('ascii','replace'),
                                'text' : insta_post['caption']['text'].encode('ascii','replace'),
                                'user_id' : insta_post['caption']['id'].encode('ascii','replace'),
                                'date_time' : insta_post['caption']['created_time'].encode('ascii','replace'),
                                'date_year' : date_time.year,
                                'date_month' : date_time.month,
                                'date_day' : date_time.day,
                                'day_of_week' : date_time.weekday(),
                                'media_type' : insta_post['type'].encode('ascii','replace'),
                                'low_res_url' : insta_post['images']['low_resolution']['url'].encode('ascii','replace'),
                                'thumbnail_url' : insta_post['images']['thumbnail']['url'].encode('ascii','replace'),
                                'stand_res_url' : insta_post['images']['standard_resolution']['url'].encode('ascii','replace')
                            }

                            post_to_push = post(**post_attr)
                            session.add(post_to_push)
                            session.commit()

                            for insta_tag in insta_post['tags']:
                                #print tag.encode('ascii','replace')
                                tag_attr = {
                                    'post_id' : insta_post['id'].encode('ascii','replace'),
                                    'tag': insta_tag.encode('ascii','replace'),
                                }

                                tag_to_push = tag(**tag_attr)
                                session.add(tag_to_push)
                                #print tag

                            for insta_comment in insta_post['comments']['data']:
                                comment_attr = {
                                    'post_id' : insta_post['id'].encode('ascii','replace'),
                                    'text' : insta_comment['text'].encode('ascii','replace')
                                }
                                comment_to_push = comment(**comment_attr)
                                session.add(comment_to_push)
                                #print comment['text'].encode('ascii','replace')
            except Exception, e:
                print 'caught error in body, continuing: %s' % e
                traceback.print_exc()

        session.commit()
        query_url = resp.json()['pagination']['next_url']
        resp = requests.get(query_url)