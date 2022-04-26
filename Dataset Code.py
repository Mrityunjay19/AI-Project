#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
api_key="AIzaSyDJTRptxnf-H3fnILbvM16J7uJ5QPQ63Ik"
scopes = ["https://www.googleapis.com/auth/youtube.readonly"]

api_service_name = "youtube"
api_version = "v3"
client_secrets_file = "kr.json"

flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
client_secrets_file, scopes)
credentials = flow.run_console()
youtube = googleapiclient.discovery.build(api_service_name, api_version, credentials=credentials, developerKey = api_key)


# In[10]:


import pandas as pd


# In[ ]:





# In[11]:


channel_views = [ ]
channel_title = [ ]
channel_totalvideo=[ ] 


# In[12]:


request = youtube.channels().list(part="statistics,snippet,brandingSettings",id="UCs9rS7E4bverH-hxgHJtNRQ")
response = request.execute()
    #channel_title.append(response['items'][0]['snippet']['title'])
    #channel_views.append(response['items'][0]['statistics']['viewCount'])
    #channel_totalvideo.append(response['items'][0]['statistics']['videoCount'])


# In[13]:


response


# In[14]:


View_Count=[]
Video_Count=[]
Subscriber_Count=[]
Date_Of_Registration=[]
Number_Of_Featured_Channel=[]


# In[15]:


View_Count.append(response['items'][0]['statistics']['viewCount'])
Video_Count.append(response['items'][0]['statistics']['videoCount'])
Subscriber_Count.append(response['items'][0]['statistics']['subscriberCount'])
Date_Of_Registration.append(response['items'][0]['snippet']['publishedAt'][0:10])
Number_Of_Featured_Channel.append(len(response['items'][0]['brandingSettings']['channel']['featuredChannelsUrls']))


# In[16]:


from apiclient.discovery import build


# In[17]:


youtube = build('youtube', 'v3', developerKey=api_key)


# In[18]:


def get_channel_videos(channel_id):
    
    # get Uploads playlist id
    res = youtube.channels().list(id=channel_id, 
                                  part='contentDetails').execute()
    playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    
    videos = []
    next_page_token = None
    
    while 1:
        res = youtube.playlistItems().list(playlistId=playlist_id, 
                                           part='snippet', 
                                           maxResults=50,
                                           pageToken=next_page_token).execute()
        videos += res['items']
        next_page_token = res.get('nextPageToken')
        
        if next_page_token is None:
            break
    
    return videos


# In[19]:


videos = get_channel_videos('UCs9rS7E4bverH-hxgHJtNRQ')


# In[20]:


videos


# In[21]:


len(videos)


# In[22]:


for video in videos:
    print(video['snippet']['title'])


# In[23]:


def get_videos_stats(video_ids):
    stats = []
    for i in range(0, len(video_ids), 50):
        res = youtube.videos().list(id=','.join(video_ids[i:i+50]),
                                   part='statistics').execute()
        stats += res['items']
        
    return stats


# In[24]:


video_ids = list(map(lambda x:x['snippet']['resourceId']['videoId'], videos))


# In[25]:


stats = get_videos_stats(video_ids)


# In[26]:


len(stats)


# In[27]:


most_disliked = sorted(stats, key=lambda x:int(x['statistics']['dislikeCount']), reverse=True)
most_liked = sorted(stats, key=lambda x:int(x['statistics']['likeCount']), reverse=True)
dislike=1
like=1


# In[28]:


for video in most_disliked:
     dislike+=int(video['statistics']['dislikeCount'])


# In[29]:


for video in most_liked:
    like+=int(video['statistics']['likeCount'])


# In[30]:


Like_Dislike_Ratio=[]


# In[31]:


Like_Dislike_Ratio.append(like/dislike)


# In[32]:


View_Count


# In[33]:


data={'View Count':View_Count,'Subscriber Count':Subscriber_Count,'Video Count':Video_Count,'Date Of Registration':Date_Of_Registration,'Like Dislike Ratio':Like_Dislike_Ratio,'Number Of Featured Channel':Number_Of_Featured_Channel}
df=pd.DataFrame(data)


# In[34]:


df


# In[ ]:




