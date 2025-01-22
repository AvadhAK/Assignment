#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
from src.predictions import load_model_and_predict
from tests.test_train_model import test_train_model

def test_predict():

    test_train_model()
    # Test prediction functionality
    sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = load_model_and_predict(sample_data)
    assert prediction is not None, "Prediction failed"
    assert len(prediction) == 1, "Prediction result length mismatch"

