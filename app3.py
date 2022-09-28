#!/usr/bin/env python
# coding: utf-8

# In[ ]:
 
import streamlit as st
import numpy as np
import pandas as pd
import librosa
from keras.models import load_model
import os
import sys
import h5py
from sklearn.preprocessing import StandardScaler

model = load_model("modelfinal.h5")
lables = np.load("labels.npy")
scaler = StandardScaler()


def main():
    st.title('Speech Emotion Recognition')
    #st.write(''' # Speech Emotion Recognition''')

    speech_file = st.file_uploader(label = 'Please Upload Audio File')
    if speech_file is not None:
        st.write('File Uploaded')
        st.audio(speech_file,format='audio / ogg')
        
        path = speech_file
        audio, sample_rate = librosa.load(speech_file,duration=2.5,offset=0.6) 
        mfcc =np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate,n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfcc, axis=1)
        x = np.expand_dims(x, axis=0)
        
        
       
        btn = st.button("predict")

        if btn:
    
            pred = model.predict(x)
            pred = lables[np.argmax(pred)]
            st.subheader(pred)
        
if __name__ == '__main__':
    main()