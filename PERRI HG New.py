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
import seaborn as sns
from matplotlib import animation
from matplotlib.animation import FuncAnimation
get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib.colors import ListedColormap
import matplotlib.image as mpimg
from random import randrange
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from matplotlib.animation import PillowWriter
import time


# In[19]:


# Setup for animated graph

# Cleaning up data for usage 
filename = "perri_temp227a.csv"
headers = []
headers_ind = []
for i in range(1,14): 
    headers.append("Remote Board " + str(i)) 
    headers.append("Local Board " + str(i)) 
    headers_ind.append(i)

time = []
for iter in range(0,705): 
    time.append(iter)
dataset = pd.read_csv(filename)
df = pd.DataFrame(dataset)
df.columns = headers
remote = np.array(df.iloc[:, ::2])
local = np.array(df.iloc[:, 1::2])
stack = np.stack((remote[1,], local[1,]))

# Zero line (black area between boards on graph)
zeroes = np.full(
  shape=13,
  fill_value=100
)
newdata = []
for i in range(0, len(remote)):
    stack2 = np.stack((remote[i,], local[i,]))
    newdata.append(np.stack((remote[i,], zeroes, local[i,])))
    zeroes = np.zeros(13)
    
    
# Defining parameters
totmax = 48.1875
totmin = 25.75


# In[20]:


# Animated heat graph of boards

fig = plt.figure()

# Animation code
def animate(i):
    fig.clear()
    sns.heatmap(newdata[i], vmin = totmin, vmax = totmax, cbar=True)
    
plt.show()

anim = animation.FuncAnimation(fig, animate, frames=len(remote), repeat=False)

plt.show()


# In[22]:


# Example code of overlapping image: I tried a way of bringing the image to the foreground. This was a success
# However, when I tried applying this to our animated graph, it did not recognize the seaborn graph and not print the img as a result

data_1 = [randrange(0, 10) for _ in range(0, 10)]
data_2 = [randrange(0, 10) for _ in range(0, 10)]

plt.plot(data_1, label="Random Data", c="Red")
plt.bar(data_2, data_1, label="Random Data")

file = "UpdatedTemplate.png"
logo = mpimg.imread(file)

imagebox = OffsetImage(logo, zoom = 0.15)
ab = AnnotationBbox(imagebox, (5, 700), frameon = False)
ax.add_artist(ab)

plt.imshow(logo)
plt.show()


# In[7]:


# Example of animated plot with background: I wanted to see how img functions could work with an animated graph
# This did not transfer over well to seaborn

rng = np.random.default_rng()
fig = plt.figure(figsize=[10, 9])

img = mpimg.imread('UpdatedTemplate.png')

ims = []
for i in range(200):
    df = pd.DataFrame(rng.integers(0, 1300, size=(100, 2)), columns=list('xy'))
    x = df["x"]
    y = df["y"]
    im = plt.plot(x, y, "b.")
    ims.append(im)
#     print(i)   

ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
                                repeat_delay=1000)
plt.imshow(img)

writer = PillowWriter(fps=2)
ani.save("demo2.gif", writer=writer)


# In[14]:


# Example code of animated seaborn heat graph
# Just a randomly generated heat graph I referenced when I was trying to create our animated graph
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

fig = plt.figure()
dimension = (6, 3)
data = np.random.rand(dimension[0], dimension[1])
print(type(data))
sns.heatmap(data, vmax=.8)

def init():
    sns.heatmap(np.zeros(dimension), vmax=.8, cbar=False)

def animate(i):
    data = np.random.rand(dimension[0], dimension[1])
    sns.heatmap(data, vmax=.8, cbar=False)

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=95, repeat=False)
print(data)

plt.show()


# In[ ]:




