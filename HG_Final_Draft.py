#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing
import pandas as pd
from statistics import mean
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns; sns.set()
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import matplotlib.image as mpimg
from random import randrange
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from matplotlib.animation import PillowWriter
import time
from matplotlib.colors import LinearSegmentedColormap
import numpy as numpy
file = "UpdatedTemplate.png"
import moviepy.editor as mp


# In[2]:


# Defining Variables
sec = 705
slots = 15
gifname = "perri.gif"
mp4name = "perri.mp4"
# These variables may change with data
tempmax = 48.1875
tempmin = 25.75


# In[3]:


# Cleaning up data for usage 
file = "UpdatedTemplate.png"
filename = "perri_temp227a.csv"
headers = []
headers_ind = []
for i in range(1,slots-1): 
    headers.append("Remote Board " + str(i)) 
    headers.append("Local Board " + str(i)) 
    headers_ind.append(i)

time = []
for iter in range(0, sec): 
    time.append(iter)
dataset = pd.read_csv(filename)
df = pd.DataFrame(dataset)
df.columns = headers
remote = np.array(df.iloc[:, ::2])
local = np.array(df.iloc[:, 1::2])

#Zero Concatenate (Adding on the two zero blocks at the ends to represent the missing boards)
zero = numpy.zeros((sec, 2))

remote = np.concatenate([remote, zero], axis=1)
local = np.concatenate([local, zero], axis=1)

# Checking shapes (optional)
print(zero)
print(zero.shape)

# Establishing stacks
stack = np.stack((remote[1,], local[1,]))

# Making middle row black
zeromid = np.zeros(slots)

newdata = []
for i in range(0, len(remote)):
    stack2 = np.stack((remote[i,], local[i,]))
    newdata.append(np.stack((remote[i,], zeromid,local[i,])))


# In[ ]:


# Graph aesthetics
wd = plt.cm.winter._segmentdata # only has r,g,b
wd['alpha'] =  ((0.0, 0.0, 0.3),
               (0.3, 0.3, 1.0),
               (1.0, 1.0, 1.0))
# Modified colormap with changing alpha
al_winter = LinearSegmentedColormap('AlphaWinter', wd)
# Get the map image as an array so we can plot it
map_img = mpimg.imread('UpdatedTemplate.png')
fig = plt.figure()
# Manually put in 0 or 1, and skips the first value
hmax = sns.heatmap(newdata[0], vmin = tempmin, vmax = tempmax, cbar=True,
    zorder = 1)
hmax.imshow(map_img,
    aspect = hmax.get_aspect(),
    extent = hmax.get_xlim() + hmax.get_ylim(),
    zorder = 2
           )

# Graph layout
def animate(i):
    fig.clear()
    i = i+1
    # I'm removing the first frame... just getting rid of that random white bar
    hmax = sns.heatmap(newdata[i], vmin = tempmin, vmax = tempmax, cbar = True,
        zorder = 1)
    hmax.imshow(map_img,
        aspect = hmax.get_aspect(),
        extent = hmax.get_xlim() + hmax.get_ylim(),
        zorder = 2)

anim = animation.FuncAnimation(fig, animate, frames=len(time)-1, interval=500, repeat=False)

# Saving animation
writer = PillowWriter(fps = 10)
anim.save(gifname, writer=writer)
clip = mp.VideoFileClip(gifname)
clip.write_videofile(mp4name)

# Showing animation
plt.show

