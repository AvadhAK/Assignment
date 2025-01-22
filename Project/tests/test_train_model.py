#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from src.train_model import train_model

def test_train_model():
    # Train the model and check if the file is created
    train_model()
    assert os.path.exists("model.pkl"), "Model file was not created"

