
# coding: utf-8

# In[20]:



# Copyright (c) 2017-present, WawLabs.
# All rights reserved.

import requests
import json
import cv2
import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


file_path = "E:\sample.jpg"


# In[22]:


url = "https://wawapps.com/api/car_cl_2"


# In[24]:


fin =  open(file_path, 'rb')
files = {'file': fin}

try:
    r = requests.post(url, files=files, verify = False)
finally:
    fin.close()


# In[25]:


json_data = json.loads(r.text)


# In[26]:


json_data


# In[27]:


img = cv2.imread(file_path)


# In[31]:


for i in range(len(json_data)):
    coords = json_data[i]['coords']
    xStart = coords[0]
    xEnd = coords[1]
    yStart = coords[2]
    yEnd   = coords[3]
    
    cv2.rectangle(img, (xStart, yStart), (xEnd, yEnd),(255,255,255), 2)
    
    label = list(json_data[i]['model_preds'][0].keys())[0] + " : " + "%.1f" % np.float32((json_data[i]['model_preds'][0][list(json_data[i]['model_preds'][0].keys())[0]]))
    
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, 1)
    
    cv2.putText(img, label, (xStart, yStart - labelSize[1]),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255,255,255), 2)


# In[32]:


plt.figure(figsize=(12,8))
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()

