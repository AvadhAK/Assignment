#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from src.train_model import train_model

def test_train_model():
    # Save the model
    parent_dir = os.path.dirname(os.getcwd())  # Get the parent directory
    model_path = os.path.join(parent_dir, "model.pkl")
    # Train the model and check if the file is created
    train_model()
    assert os.path.exists(model_path), "Model file was not created"

