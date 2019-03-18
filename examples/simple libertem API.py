#!/usr/bin/env python
# coding: utf-8

# In[3]:

#%%
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


# In[4]:


get_ipython().run_line_magic('matplotlib', 'nbagg')


# In[5]:


import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from libertem import api


# In[6]:


ctx = api.Context()


# In[7]:


ds = ctx.load(
    "raw",
    path="C:/Users/weber/ownCloud/Projects/Open Pixelated STEM framework/Data/EMPAD/scan_11_x256_y256.raw",
    dtype="float32",
    scan_size=(256, 256),
    detector_size_raw=(130, 128),
    crop_detector_to=(128, 128),
)
(scan_y, scan_x, detector_y, detector_x) = ds.shape
mask_shape = np.array((detector_y, detector_x))
cx = detector_x/2
cy = detector_y/2


# In[6]:

origin=(scan_x//2,scan_y//2,scan_z//2)
pick_job = ctx.create_pick_job(dataset=ds,origin=origin)


# In[7]:


get_ipython().run_line_magic('time', 'pick_result = ctx.run(pick_job)')


# In[8]:


print(pick_result)


# In[9]:


fig, axes = plt.subplots()
axes.imshow(pick_result)


# In[10]:


def all_ones():
    return np.ones((detector_y, detector_x))


# In[11]:


def single_pixel():
    buf = np.zeros((detector_y, detector_x))
    buf[int(cy), int(cx)] = 1
    return buf


# Here we use a mask job. Jobs generally use the LiberTEM computation rather directly and return a simple numerical result. See further below for examples that use the high-level analysis interface instead.

# In[12]:


mask_job = ctx.create_mask_job(factories=[all_ones, single_pixel], dataset=ds)


# In[13]:


get_ipython().run_cell_magic('time', '', 'mask_job_result = ctx.run(mask_job)')


# In[14]:


print(mask_job_result)


# In[15]:


fig, axes = plt.subplots()
axes.imshow(mask_job_result[0], cmap=cm.gist_earth)


# In[16]:


fig, axes = plt.subplots()
axes.imshow(mask_job_result[1], cmap=cm.gist_earth)


# In[17]:


mask_analysis = ctx.create_mask_analysis(dataset=ds, factories=[all_ones, single_pixel])


# In[18]:


get_ipython().run_line_magic('time', 'mask_analysis_result = ctx.run(mask_analysis)')


# In[19]:


print(mask_analysis_result)


# In[20]:


print(mask_analysis_result[0])


# In[21]:


fig, axes = plt.subplots()
axes.imshow(mask_analysis_result[0].visualized)


# In[22]:


fig, axes = plt.subplots()
axes.imshow(mask_analysis_result[1].visualized)


# In[23]:


ro = min(detector_x,detector_y)/2
haadf_analysis = ctx.create_ring_analysis(dataset=ds, cx=cx, cy=cy, ro=ro, ri=ro*0.8)


# In[24]:


get_ipython().run_line_magic('time', 'haadf_result = ctx.run(haadf_analysis)')


# In[25]:


print(haadf_result)


# In[26]:


print(haadf_result.intensity)


# In[27]:


fig, axes = plt.subplots()
axes.imshow(haadf_result.intensity.visualized)


# In[28]:


bf_analysis = ctx.create_disk_analysis(dataset=ds, cx=cx, cy=cy, r=ro*0.3)


# In[29]:


get_ipython().run_line_magic('time', 'bf_result = ctx.run(bf_analysis)')


# In[30]:


print(bf_result)


# In[31]:


print(bf_result.intensity)


# In[32]:


fig, axes = plt.subplots()
axes.imshow(bf_result.intensity.visualized)


# In[33]:


point_analysis = ctx.create_point_analysis(dataset=ds, x=cx, y=cy)


# In[38]:


get_ipython().run_line_magic('time', 'point_result = ctx.run(point_analysis)')


# In[35]:


print(point_result)


# In[36]:


print(point_result.intensity)


# In[37]:


fig, axes = plt.subplots()
axes.imshow(point_result.intensity.visualized)


# In[8]:


sum_analysis = ctx.create_sum_analysis(dataset=ds)


# In[9]:


get_ipython().run_line_magic('time', 'sum_result = ctx.run(sum_analysis)')


# In[10]:


print(sum_result)


# In[12]:


fig, axes = plt.subplots()
axes.imshow(sum_result.intensity.raw_data)


# In[ ]:




