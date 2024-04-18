#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import uproot
import numba
from numba import cuda
from numba import njit, prange
import matplotlib.pyplot as plt
import random
from tensorflow.keras import datasets, layers, models, losses
from tensorflow import keras
import tensorflow as tf
import gc


# In[2]:


#Import MC Events
print("Reading ROOT files...")
targettree = uproot.open('Root_Files/Target_Train_QA_v2.root:QA_ana')
targetdata = targettree.arrays(library="np")
targetevents=len(targetdata['n_tracks'])
print("Done")

#This reads the dimuon tracks from the target into an array
pos_events=np.zeros((targetevents,54))
neg_events=np.zeros((targetevents,54))
print("Reading target events...")
for j in range(targetevents):
    #if(j%100==0):print(j,end="\r") #This is to keep track of how quickly the events are being generated
    first=targetdata['pid'][j][0]
    if(first>0):
        pos_events[j][0]=targetdata['D0U_ele'][j][0]
        neg_events[j][0]=targetdata['D0U_ele'][j][1]
        pos_events[j][1]=targetdata['D0Up_ele'][j][0]
        neg_events[j][1]=targetdata['D0Up_ele'][j][1]
        pos_events[j][2]=targetdata['D0X_ele'][j][0]
        neg_events[j][2]=targetdata['D0X_ele'][j][1]        
        pos_events[j][3]=targetdata['D0Xp_ele'][j][0]
        neg_events[j][3]=targetdata['D0Xp_ele'][j][1]
        pos_events[j][4]=targetdata['D0V_ele'][j][0]
        neg_events[j][4]=targetdata['D0V_ele'][j][1]
        pos_events[j][5]=targetdata['D0Vp_ele'][j][0]
        neg_events[j][5]=targetdata['D0Vp_ele'][j][1]
        pos_events[j][16]=targetdata['D2U_ele'][j][0]
        neg_events[j][16]=targetdata['D2U_ele'][j][1]
        pos_events[j][17]=targetdata['D2Up_ele'][j][0]
        neg_events[j][17]=targetdata['D2Up_ele'][j][1]
        pos_events[j][15]=targetdata['D2X_ele'][j][0]
        neg_events[j][15]=targetdata['D2X_ele'][j][1]
        pos_events[j][14]=targetdata['D2Xp_ele'][j][0]
        neg_events[j][14]=targetdata['D2Xp_ele'][j][1]
        pos_events[j][12]=targetdata['D2V_ele'][j][0]
        neg_events[j][12]=targetdata['D2V_ele'][j][1]
        pos_events[j][13]=targetdata['D2Vp_ele'][j][0]
        neg_events[j][13]=targetdata['D2Vp_ele'][j][1]
        pos_events[j][23]=targetdata['D3pU_ele'][j][0]
        neg_events[j][23]=targetdata['D3pU_ele'][j][1]
        pos_events[j][22]=targetdata['D3pUp_ele'][j][0]
        neg_events[j][22]=targetdata['D3pUp_ele'][j][1]
        pos_events[j][21]=targetdata['D3pX_ele'][j][0]
        neg_events[j][21]=targetdata['D3pX_ele'][j][1]
        pos_events[j][20]=targetdata['D3pXp_ele'][j][0]
        neg_events[j][20]=targetdata['D3pXp_ele'][j][1]
        pos_events[j][19]=targetdata['D3pV_ele'][j][0]
        neg_events[j][19]=targetdata['D3pV_ele'][j][1]
        pos_events[j][18]=targetdata['D3pVp_ele'][j][0]
        neg_events[j][18]=targetdata['D3pVp_ele'][j][1]
        pos_events[j][29]=targetdata['D3mU_ele'][j][0]
        neg_events[j][29]=targetdata['D3mU_ele'][j][1]
        pos_events[j][28]=targetdata['D3mUp_ele'][j][0]
        neg_events[j][28]=targetdata['D3mUp_ele'][j][1]
        pos_events[j][27]=targetdata['D3mX_ele'][j][0]
        neg_events[j][27]=targetdata['D3mX_ele'][j][1]
        pos_events[j][26]=targetdata['D3mXp_ele'][j][0]
        neg_events[j][26]=targetdata['D3mXp_ele'][j][1]
        pos_events[j][25]=targetdata['D3mV_ele'][j][0]
        neg_events[j][25]=targetdata['D3mV_ele'][j][1]
        pos_events[j][24]=targetdata['D3mVp_ele'][j][0]
        neg_events[j][24]=targetdata['D3mVp_ele'][j][1]
        pos_events[j][30]=targetdata['H1B_ele'][j][0]
        neg_events[j][30]=targetdata['H1B_ele'][j][1]
        pos_events[j][31]=targetdata['H1T_ele'][j][0]
        neg_events[j][31]=targetdata['H1T_ele'][j][1]
        pos_events[j][32]=targetdata['H1L_ele'][j][0]
        neg_events[j][32]=targetdata['H1L_ele'][j][1]
        pos_events[j][33]=targetdata['H1R_ele'][j][0]
        neg_events[j][33]=targetdata['H1R_ele'][j][1]
        pos_events[j][34]=targetdata['H2L_ele'][j][0]
        neg_events[j][34]=targetdata['H2L_ele'][j][1]
        pos_events[j][35]=targetdata['H2R_ele'][j][0]
        neg_events[j][35]=targetdata['H2R_ele'][j][1]
        pos_events[j][36]=targetdata['H2T_ele'][j][0]
        neg_events[j][36]=targetdata['H2T_ele'][j][1]
        pos_events[j][37]=targetdata['H2B_ele'][j][0]
        neg_events[j][37]=targetdata['H2B_ele'][j][1]
        pos_events[j][38]=targetdata['H3B_ele'][j][0]
        neg_events[j][38]=targetdata['H3B_ele'][j][1]
        pos_events[j][39]=targetdata['H3T_ele'][j][0]
        neg_events[j][39]=targetdata['H3T_ele'][j][1]
        pos_events[j][40]=targetdata['H4Y1L_ele'][j][0]
        neg_events[j][40]=targetdata['H4Y1L_ele'][j][1]
        pos_events[j][41]=targetdata['H4Y1R_ele'][j][0]
        neg_events[j][41]=targetdata['H4Y1R_ele'][j][1]
        pos_events[j][42]=targetdata['H4Y2L_ele'][j][0]
        neg_events[j][42]=targetdata['H4Y2L_ele'][j][1]
        pos_events[j][43]=targetdata['H4Y2R_ele'][j][0]
        neg_events[j][43]=targetdata['H4Y2R_ele'][j][1]
        pos_events[j][44]=targetdata['H4B_ele'][j][0]
        neg_events[j][44]=targetdata['H4B_ele'][j][1]
        pos_events[j][45]=targetdata['H4T_ele'][j][0]
        neg_events[j][45]=targetdata['H4T_ele'][j][1]
        pos_events[j][46]=targetdata['P1Y1_ele'][j][0]
        neg_events[j][46]=targetdata['P1Y1_ele'][j][1]
        pos_events[j][47]=targetdata['P1Y2_ele'][j][0]
        neg_events[j][47]=targetdata['P1Y2_ele'][j][1]
        pos_events[j][48]=targetdata['P1X1_ele'][j][0]
        neg_events[j][48]=targetdata['P1X1_ele'][j][1]
        pos_events[j][49]=targetdata['P1X2_ele'][j][0]
        neg_events[j][49]=targetdata['P1X2_ele'][j][1]
        pos_events[j][50]=targetdata['P2X1_ele'][j][0]
        neg_events[j][50]=targetdata['P2X1_ele'][j][1]
        pos_events[j][51]=targetdata['P2X2_ele'][j][0]
        neg_events[j][51]=targetdata['P2X2_ele'][j][1]
        pos_events[j][52]=targetdata['P2Y1_ele'][j][0]
        neg_events[j][52]=targetdata['P2Y1_ele'][j][1]
        pos_events[j][53]=targetdata['P2Y2_ele'][j][0]
        neg_events[j][53]=targetdata['P2Y2_ele'][j][1]
    else:
        pos_events[j][0]=targetdata['D0U_ele'][j][1]
        neg_events[j][0]=targetdata['D0U_ele'][j][0]
        pos_events[j][1]=targetdata['D0Up_ele'][j][1]
        neg_events[j][1]=targetdata['D0Up_ele'][j][0]
        pos_events[j][2]=targetdata['D0X_ele'][j][1]
        neg_events[j][2]=targetdata['D0X_ele'][j][0]        
        pos_events[j][3]=targetdata['D0Xp_ele'][j][1]
        neg_events[j][3]=targetdata['D0Xp_ele'][j][0]
        pos_events[j][4]=targetdata['D0V_ele'][j][1]
        neg_events[j][4]=targetdata['D0V_ele'][j][0]
        pos_events[j][5]=targetdata['D0Vp_ele'][j][1]
        neg_events[j][5]=targetdata['D0Vp_ele'][j][0]
        pos_events[j][16]=targetdata['D2U_ele'][j][1]
        neg_events[j][16]=targetdata['D2U_ele'][j][0]
        pos_events[j][17]=targetdata['D2Up_ele'][j][1]
        neg_events[j][17]=targetdata['D2Up_ele'][j][0]
        pos_events[j][15]=targetdata['D2X_ele'][j][1]
        neg_events[j][15]=targetdata['D2X_ele'][j][0]
        pos_events[j][14]=targetdata['D2Xp_ele'][j][1]
        neg_events[j][14]=targetdata['D2Xp_ele'][j][0]
        pos_events[j][12]=targetdata['D2V_ele'][j][1]
        neg_events[j][12]=targetdata['D2V_ele'][j][0]
        pos_events[j][13]=targetdata['D2Vp_ele'][j][1]
        neg_events[j][13]=targetdata['D2Vp_ele'][j][0]
        pos_events[j][23]=targetdata['D3pU_ele'][j][1]
        neg_events[j][23]=targetdata['D3pU_ele'][j][0]
        pos_events[j][22]=targetdata['D3pUp_ele'][j][1]
        neg_events[j][22]=targetdata['D3pUp_ele'][j][0]
        pos_events[j][21]=targetdata['D3pX_ele'][j][1]
        neg_events[j][21]=targetdata['D3pX_ele'][j][0]
        pos_events[j][20]=targetdata['D3pXp_ele'][j][1]
        neg_events[j][20]=targetdata['D3pXp_ele'][j][0]
        pos_events[j][19]=targetdata['D3pV_ele'][j][1]
        neg_events[j][19]=targetdata['D3pV_ele'][j][0]
        pos_events[j][18]=targetdata['D3pVp_ele'][j][1]
        neg_events[j][18]=targetdata['D3pVp_ele'][j][0]
        pos_events[j][29]=targetdata['D3mU_ele'][j][1]
        neg_events[j][29]=targetdata['D3mU_ele'][j][0]
        pos_events[j][28]=targetdata['D3mUp_ele'][j][1]
        neg_events[j][28]=targetdata['D3mUp_ele'][j][0]
        pos_events[j][27]=targetdata['D3mX_ele'][j][1]
        neg_events[j][27]=targetdata['D3mX_ele'][j][0]
        pos_events[j][26]=targetdata['D3mXp_ele'][j][1]
        neg_events[j][26]=targetdata['D3mXp_ele'][j][0]
        pos_events[j][25]=targetdata['D3mV_ele'][j][1]
        neg_events[j][25]=targetdata['D3mV_ele'][j][0]
        pos_events[j][24]=targetdata['D3mVp_ele'][j][1]
        neg_events[j][24]=targetdata['D3mVp_ele'][j][0]
        pos_events[j][30]=targetdata['H1B_ele'][j][1]
        neg_events[j][30]=targetdata['H1B_ele'][j][0]
        pos_events[j][31]=targetdata['H1T_ele'][j][1]
        neg_events[j][31]=targetdata['H1T_ele'][j][0]
        pos_events[j][32]=targetdata['H1L_ele'][j][1]
        neg_events[j][32]=targetdata['H1L_ele'][j][0]
        pos_events[j][33]=targetdata['H1R_ele'][j][1]
        neg_events[j][33]=targetdata['H1R_ele'][j][0]
        pos_events[j][34]=targetdata['H2L_ele'][j][1]
        neg_events[j][34]=targetdata['H2L_ele'][j][0]
        pos_events[j][35]=targetdata['H2R_ele'][j][1]
        neg_events[j][35]=targetdata['H2R_ele'][j][0]
        pos_events[j][36]=targetdata['H2T_ele'][j][1]
        neg_events[j][36]=targetdata['H2T_ele'][j][0]
        pos_events[j][37]=targetdata['H2B_ele'][j][1]
        neg_events[j][37]=targetdata['H2B_ele'][j][0]
        pos_events[j][38]=targetdata['H3B_ele'][j][1]
        neg_events[j][38]=targetdata['H3B_ele'][j][0]
        pos_events[j][39]=targetdata['H3T_ele'][j][1]
        neg_events[j][39]=targetdata['H3T_ele'][j][0]
        pos_events[j][40]=targetdata['H4Y1L_ele'][j][1]
        neg_events[j][40]=targetdata['H4Y1L_ele'][j][0]
        pos_events[j][41]=targetdata['H4Y1R_ele'][j][1]
        neg_events[j][41]=targetdata['H4Y1R_ele'][j][0]
        pos_events[j][42]=targetdata['H4Y2L_ele'][j][1]
        neg_events[j][42]=targetdata['H4Y2L_ele'][j][0]
        pos_events[j][43]=targetdata['H4Y2R_ele'][j][1]
        neg_events[j][43]=targetdata['H4Y2R_ele'][j][0]
        pos_events[j][44]=targetdata['H4B_ele'][j][1]
        neg_events[j][44]=targetdata['H4B_ele'][j][0]
        pos_events[j][45]=targetdata['H4T_ele'][j][1]
        neg_events[j][45]=targetdata['H4T_ele'][j][0]
        pos_events[j][46]=targetdata['P1Y1_ele'][j][1]
        neg_events[j][46]=targetdata['P1Y1_ele'][j][0]
        pos_events[j][47]=targetdata['P1Y2_ele'][j][1]
        neg_events[j][47]=targetdata['P1Y2_ele'][j][0]
        pos_events[j][48]=targetdata['P1X1_ele'][j][1]
        neg_events[j][48]=targetdata['P1X1_ele'][j][0]
        pos_events[j][49]=targetdata['P1X2_ele'][j][1]
        neg_events[j][49]=targetdata['P1X2_ele'][j][0]
        pos_events[j][50]=targetdata['P2X1_ele'][j][1]
        neg_events[j][50]=targetdata['P2X1_ele'][j][0]
        pos_events[j][51]=targetdata['P2X2_ele'][j][1]
        neg_events[j][51]=targetdata['P2X2_ele'][j][0]
        pos_events[j][52]=targetdata['P2Y1_ele'][j][1]
        neg_events[j][52]=targetdata['P2Y1_ele'][j][0]
        pos_events[j][53]=targetdata['P2Y2_ele'][j][1]
        neg_events[j][53]=targetdata['P2Y2_ele'][j][0]
print("Done")

del targettree, targetdata,targetevents


# In[3]:


#Import MC Events
print("Reading ROOT files...")
targettree = uproot.open('Root_Files/Target_Val_QA_v2.root:QA_ana')
targetdata = targettree.arrays(library="np")
targetevents=len(targetdata['n_tracks'])
print("Done")

#This reads the dimuon tracks from the target into an array
pos_events_val=np.zeros((targetevents,54))
neg_events_val=np.zeros((targetevents,54))
#print("Reading target events...")
for j in range(targetevents):
    #first=targetdata['pid'][j][0]
    if(first>0):
        pos_events_val[j][0]=targetdata['D0U_ele'][j][0]
        neg_events_val[j][0]=targetdata['D0U_ele'][j][1]
        pos_events_val[j][1]=targetdata['D0Up_ele'][j][0]
        neg_events_val[j][1]=targetdata['D0Up_ele'][j][1]
        pos_events_val[j][2]=targetdata['D0X_ele'][j][0]
        neg_events_val[j][2]=targetdata['D0X_ele'][j][1]        
        pos_events_val[j][3]=targetdata['D0Xp_ele'][j][0]
        neg_events_val[j][3]=targetdata['D0Xp_ele'][j][1]
        pos_events_val[j][4]=targetdata['D0V_ele'][j][0]
        neg_events_val[j][4]=targetdata['D0V_ele'][j][1]
        pos_events_val[j][5]=targetdata['D0Vp_ele'][j][0]
        neg_events_val[j][5]=targetdata['D0Vp_ele'][j][1]
        pos_events_val[j][16]=targetdata['D2U_ele'][j][0]
        neg_events_val[j][16]=targetdata['D2U_ele'][j][1]
        pos_events_val[j][17]=targetdata['D2Up_ele'][j][0]
        neg_events_val[j][17]=targetdata['D2Up_ele'][j][1]
        pos_events_val[j][15]=targetdata['D2X_ele'][j][0]
        neg_events_val[j][15]=targetdata['D2X_ele'][j][1]
        pos_events_val[j][14]=targetdata['D2Xp_ele'][j][0]
        neg_events_val[j][14]=targetdata['D2Xp_ele'][j][1]
        pos_events_val[j][12]=targetdata['D2V_ele'][j][0]
        neg_events_val[j][12]=targetdata['D2V_ele'][j][1]
        pos_events_val[j][13]=targetdata['D2Vp_ele'][j][0]
        neg_events_val[j][13]=targetdata['D2Vp_ele'][j][1]
        pos_events_val[j][23]=targetdata['D3pU_ele'][j][0]
        neg_events_val[j][23]=targetdata['D3pU_ele'][j][1]
        pos_events_val[j][22]=targetdata['D3pUp_ele'][j][0]
        neg_events_val[j][22]=targetdata['D3pUp_ele'][j][1]
        pos_events_val[j][21]=targetdata['D3pX_ele'][j][0]
        neg_events_val[j][21]=targetdata['D3pX_ele'][j][1]
        pos_events_val[j][20]=targetdata['D3pXp_ele'][j][0]
        neg_events_val[j][20]=targetdata['D3pXp_ele'][j][1]
        pos_events_val[j][19]=targetdata['D3pV_ele'][j][0]
        neg_events_val[j][19]=targetdata['D3pV_ele'][j][1]
        pos_events_val[j][18]=targetdata['D3pVp_ele'][j][0]
        neg_events_val[j][18]=targetdata['D3pVp_ele'][j][1]
        pos_events_val[j][29]=targetdata['D3mU_ele'][j][0]
        neg_events_val[j][29]=targetdata['D3mU_ele'][j][1]
        pos_events_val[j][28]=targetdata['D3mUp_ele'][j][0]
        neg_events_val[j][28]=targetdata['D3mUp_ele'][j][1]
        pos_events_val[j][27]=targetdata['D3mX_ele'][j][0]
        neg_events_val[j][27]=targetdata['D3mX_ele'][j][1]
        pos_events_val[j][26]=targetdata['D3mXp_ele'][j][0]
        neg_events_val[j][26]=targetdata['D3mXp_ele'][j][1]
        pos_events_val[j][25]=targetdata['D3mV_ele'][j][0]
        neg_events_val[j][25]=targetdata['D3mV_ele'][j][1]
        pos_events_val[j][24]=targetdata['D3mVp_ele'][j][0]
        neg_events_val[j][24]=targetdata['D3mVp_ele'][j][1]
        pos_events_val[j][30]=targetdata['H1B_ele'][j][0]
        neg_events_val[j][30]=targetdata['H1B_ele'][j][1]
        pos_events_val[j][31]=targetdata['H1T_ele'][j][0]
        neg_events_val[j][31]=targetdata['H1T_ele'][j][1]
        pos_events_val[j][32]=targetdata['H1L_ele'][j][0]
        neg_events_val[j][32]=targetdata['H1L_ele'][j][1]
        pos_events_val[j][33]=targetdata['H1R_ele'][j][0]
        neg_events_val[j][33]=targetdata['H1R_ele'][j][1]
        pos_events_val[j][34]=targetdata['H2L_ele'][j][0]
        neg_events_val[j][34]=targetdata['H2L_ele'][j][1]
        pos_events_val[j][35]=targetdata['H2R_ele'][j][0]
        neg_events_val[j][35]=targetdata['H2R_ele'][j][1]
        pos_events_val[j][36]=targetdata['H2T_ele'][j][0]
        neg_events_val[j][36]=targetdata['H2T_ele'][j][1]
        pos_events_val[j][37]=targetdata['H2B_ele'][j][0]
        neg_events_val[j][37]=targetdata['H2B_ele'][j][1]
        pos_events_val[j][38]=targetdata['H3B_ele'][j][0]
        neg_events_val[j][38]=targetdata['H3B_ele'][j][1]
        pos_events_val[j][39]=targetdata['H3T_ele'][j][0]
        neg_events_val[j][39]=targetdata['H3T_ele'][j][1]
        pos_events_val[j][40]=targetdata['H4Y1L_ele'][j][0]
        neg_events_val[j][40]=targetdata['H4Y1L_ele'][j][1]
        pos_events_val[j][41]=targetdata['H4Y1R_ele'][j][0]
        neg_events_val[j][41]=targetdata['H4Y1R_ele'][j][1]
        pos_events_val[j][42]=targetdata['H4Y2L_ele'][j][0]
        neg_events_val[j][42]=targetdata['H4Y2L_ele'][j][1]
        pos_events_val[j][43]=targetdata['H4Y2R_ele'][j][0]
        neg_events_val[j][43]=targetdata['H4Y2R_ele'][j][1]
        pos_events_val[j][44]=targetdata['H4B_ele'][j][0]
        neg_events_val[j][44]=targetdata['H4B_ele'][j][1]
        pos_events_val[j][45]=targetdata['H4T_ele'][j][0]
        neg_events_val[j][45]=targetdata['H4T_ele'][j][1]
        pos_events_val[j][46]=targetdata['P1Y1_ele'][j][0]
        neg_events_val[j][46]=targetdata['P1Y1_ele'][j][1]
        pos_events_val[j][47]=targetdata['P1Y2_ele'][j][0]
        neg_events_val[j][47]=targetdata['P1Y2_ele'][j][1]
        pos_events_val[j][48]=targetdata['P1X1_ele'][j][0]
        neg_events_val[j][48]=targetdata['P1X1_ele'][j][1]
        pos_events_val[j][49]=targetdata['P1X2_ele'][j][0]
        neg_events_val[j][49]=targetdata['P1X2_ele'][j][1]
        pos_events_val[j][50]=targetdata['P2X1_ele'][j][0]
        neg_events_val[j][50]=targetdata['P2X1_ele'][j][1]
        pos_events_val[j][51]=targetdata['P2X2_ele'][j][0]
        neg_events_val[j][51]=targetdata['P2X2_ele'][j][1]
        pos_events_val[j][52]=targetdata['P2Y1_ele'][j][0]
        neg_events_val[j][52]=targetdata['P2Y1_ele'][j][1]
        pos_events_val[j][53]=targetdata['P2Y2_ele'][j][0]
        neg_events_val[j][53]=targetdata['P2Y2_ele'][j][1]
    else:
        pos_events_val[j][0]=targetdata['D0U_ele'][j][1]
        neg_events_val[j][0]=targetdata['D0U_ele'][j][0]
        pos_events_val[j][1]=targetdata['D0Up_ele'][j][1]
        neg_events_val[j][1]=targetdata['D0Up_ele'][j][0]
        pos_events_val[j][2]=targetdata['D0X_ele'][j][1]
        neg_events_val[j][2]=targetdata['D0X_ele'][j][0]        
        pos_events_val[j][3]=targetdata['D0Xp_ele'][j][1]
        neg_events_val[j][3]=targetdata['D0Xp_ele'][j][0]
        pos_events_val[j][4]=targetdata['D0V_ele'][j][1]
        neg_events_val[j][4]=targetdata['D0V_ele'][j][0]
        pos_events_val[j][5]=targetdata['D0Vp_ele'][j][1]
        neg_events_val[j][5]=targetdata['D0Vp_ele'][j][0]
        pos_events_val[j][16]=targetdata['D2U_ele'][j][1]
        neg_events_val[j][16]=targetdata['D2U_ele'][j][0]
        pos_events_val[j][17]=targetdata['D2Up_ele'][j][1]
        neg_events_val[j][17]=targetdata['D2Up_ele'][j][0]
        pos_events_val[j][15]=targetdata['D2X_ele'][j][1]
        neg_events_val[j][15]=targetdata['D2X_ele'][j][0]
        pos_events_val[j][14]=targetdata['D2Xp_ele'][j][1]
        neg_events_val[j][14]=targetdata['D2Xp_ele'][j][0]
        pos_events_val[j][12]=targetdata['D2V_ele'][j][1]
        neg_events_val[j][12]=targetdata['D2V_ele'][j][0]
        pos_events_val[j][13]=targetdata['D2Vp_ele'][j][1]
        neg_events_val[j][13]=targetdata['D2Vp_ele'][j][0]
        pos_events_val[j][23]=targetdata['D3pU_ele'][j][1]
        neg_events_val[j][23]=targetdata['D3pU_ele'][j][0]
        pos_events_val[j][22]=targetdata['D3pUp_ele'][j][1]
        neg_events_val[j][22]=targetdata['D3pUp_ele'][j][0]
        pos_events_val[j][21]=targetdata['D3pX_ele'][j][1]
        neg_events_val[j][21]=targetdata['D3pX_ele'][j][0]
        pos_events_val[j][20]=targetdata['D3pXp_ele'][j][1]
        neg_events_val[j][20]=targetdata['D3pXp_ele'][j][0]
        pos_events_val[j][19]=targetdata['D3pV_ele'][j][1]
        neg_events_val[j][19]=targetdata['D3pV_ele'][j][0]
        pos_events_val[j][18]=targetdata['D3pVp_ele'][j][1]
        neg_events_val[j][18]=targetdata['D3pVp_ele'][j][0]
        pos_events_val[j][29]=targetdata['D3mU_ele'][j][1]
        neg_events_val[j][29]=targetdata['D3mU_ele'][j][0]
        pos_events_val[j][28]=targetdata['D3mUp_ele'][j][1]
        neg_events_val[j][28]=targetdata['D3mUp_ele'][j][0]
        pos_events_val[j][27]=targetdata['D3mX_ele'][j][1]
        neg_events_val[j][27]=targetdata['D3mX_ele'][j][0]
        pos_events_val[j][26]=targetdata['D3mXp_ele'][j][1]
        neg_events_val[j][26]=targetdata['D3mXp_ele'][j][0]
        pos_events_val[j][25]=targetdata['D3mV_ele'][j][1]
        neg_events_val[j][25]=targetdata['D3mV_ele'][j][0]
        pos_events_val[j][24]=targetdata['D3mVp_ele'][j][1]
        neg_events_val[j][24]=targetdata['D3mVp_ele'][j][0]
        pos_events_val[j][30]=targetdata['H1B_ele'][j][1]
        neg_events_val[j][30]=targetdata['H1B_ele'][j][0]
        pos_events_val[j][31]=targetdata['H1T_ele'][j][1]
        neg_events_val[j][31]=targetdata['H1T_ele'][j][0]
        pos_events_val[j][32]=targetdata['H1L_ele'][j][1]
        neg_events_val[j][32]=targetdata['H1L_ele'][j][0]
        pos_events_val[j][33]=targetdata['H1R_ele'][j][1]
        neg_events_val[j][33]=targetdata['H1R_ele'][j][0]
        pos_events_val[j][34]=targetdata['H2L_ele'][j][1]
        neg_events_val[j][34]=targetdata['H2L_ele'][j][0]
        pos_events_val[j][35]=targetdata['H2R_ele'][j][1]
        neg_events_val[j][35]=targetdata['H2R_ele'][j][0]
        pos_events_val[j][36]=targetdata['H2T_ele'][j][1]
        neg_events_val[j][36]=targetdata['H2T_ele'][j][0]
        pos_events_val[j][37]=targetdata['H2B_ele'][j][1]
        neg_events_val[j][37]=targetdata['H2B_ele'][j][0]
        pos_events_val[j][38]=targetdata['H3B_ele'][j][1]
        neg_events_val[j][38]=targetdata['H3B_ele'][j][0]
        pos_events_val[j][39]=targetdata['H3T_ele'][j][1]
        neg_events_val[j][39]=targetdata['H3T_ele'][j][0]
        pos_events_val[j][40]=targetdata['H4Y1L_ele'][j][1]
        neg_events_val[j][40]=targetdata['H4Y1L_ele'][j][0]
        pos_events_val[j][41]=targetdata['H4Y1R_ele'][j][1]
        neg_events_val[j][41]=targetdata['H4Y1R_ele'][j][0]
        pos_events_val[j][42]=targetdata['H4Y2L_ele'][j][1]
        neg_events_val[j][42]=targetdata['H4Y2L_ele'][j][0]
        pos_events_val[j][43]=targetdata['H4Y2R_ele'][j][1]
        neg_events_val[j][43]=targetdata['H4Y2R_ele'][j][0]
        pos_events_val[j][44]=targetdata['H4B_ele'][j][1]
        neg_events_val[j][44]=targetdata['H4B_ele'][j][0]
        pos_events_val[j][45]=targetdata['H4T_ele'][j][1]
        neg_events_val[j][45]=targetdata['H4T_ele'][j][0]
        pos_events_val[j][46]=targetdata['P1Y1_ele'][j][1]
        neg_events_val[j][46]=targetdata['P1Y1_ele'][j][0]
        pos_events_val[j][47]=targetdata['P1Y2_ele'][j][1]
        neg_events_val[j][47]=targetdata['P1Y2_ele'][j][0]
        pos_events_val[j][48]=targetdata['P1X1_ele'][j][1]
        neg_events_val[j][48]=targetdata['P1X1_ele'][j][0]
        pos_events_val[j][49]=targetdata['P1X2_ele'][j][1]
        neg_events_val[j][49]=targetdata['P1X2_ele'][j][0]
        pos_events_val[j][50]=targetdata['P2X1_ele'][j][1]
        neg_events_val[j][50]=targetdata['P2X1_ele'][j][0]
        pos_events_val[j][51]=targetdata['P2X2_ele'][j][1]
        neg_events_val[j][51]=targetdata['P2X2_ele'][j][0]
        pos_events_val[j][52]=targetdata['P2Y1_ele'][j][1]
        neg_events_val[j][52]=targetdata['P2Y1_ele'][j][0]
        pos_events_val[j][53]=targetdata['P2Y2_ele'][j][1]
        neg_events_val[j][53]=targetdata['P2Y2_ele'][j][0]
print("Done")

del targettree, targetdata,targetevents


# In[4]:


@njit(parallel=True)
def clean(events):
    for j in prange(len(events)):
        for i in prange(54):
            if(events[j][i]>1000):
                events[j][i]=0
    return events


pos_events=clean(pos_events).astype(int)
neg_events=clean(neg_events).astype(int)
pos_events_val=clean(pos_events_val).astype(int)
neg_events_val=clean(neg_events_val).astype(int)


# In[5]:


neg_events=neg_events[np.count_nonzero(pos_events[:,:30],axis=1)>=18]
pos_events=pos_events[np.count_nonzero(pos_events[:,:30],axis=1)>=18]

pos_events=pos_events[np.count_nonzero(neg_events[:,:30],axis=1)>=18]
neg_events=neg_events[np.count_nonzero(neg_events[:,:30],axis=1)>=18]

neg_events=neg_events[np.count_nonzero(pos_events[:,30:46],axis=1)>=8]
pos_events=pos_events[np.count_nonzero(pos_events[:,30:46],axis=1)>=8]

pos_events=pos_events[np.count_nonzero(neg_events[:,30:46],axis=1)>=8]
neg_events=neg_events[np.count_nonzero(neg_events[:,30:46],axis=1)>=8]

neg_events=neg_events[np.count_nonzero(pos_events[:,46:],axis=1)>=8]
pos_events=pos_events[np.count_nonzero(pos_events[:,46:],axis=1)>=8]

pos_events=pos_events[np.count_nonzero(neg_events[:,46:],axis=1)>=8]
neg_events=neg_events[np.count_nonzero(neg_events[:,46:],axis=1)>=8]


neg_events_val = neg_events_val[np.count_nonzero(pos_events_val[:,:30],axis=1)>=18]
pos_events_val = pos_events_val[np.count_nonzero(pos_events_val[:,:30],axis=1)>=18]

pos_events_val = pos_events_val[np.count_nonzero(neg_events_val[:,:30],axis=1)>=18]
neg_events_val = neg_events_val[np.count_nonzero(neg_events_val[:,:30],axis=1)>=18]

neg_events_val = neg_events_val[np.count_nonzero(pos_events_val[:,30:46],axis=1)>=8]
pos_events_val = pos_events_val[np.count_nonzero(pos_events_val[:,30:46],axis=1)>=8]

pos_events_val = pos_events_val[np.count_nonzero(neg_events_val[:,30:46],axis=1)>=8]
neg_events_val = neg_events_val[np.count_nonzero(neg_events_val[:,30:46],axis=1)>=8]

neg_events_val = neg_events_val[np.count_nonzero(pos_events_val[:,46:],axis=1)>=8]
pos_events_val = pos_events_val[np.count_nonzero(pos_events_val[:,46:],axis=1)>=8]

pos_events_val = pos_events_val[np.count_nonzero(neg_events_val[:,46:],axis=1)>=8]
neg_events_val = neg_events_val[np.count_nonzero(neg_events_val[:,46:],axis=1)>=8]


# In[6]:


@njit(parallel=True)
def track_injection(hits,pos_e,neg_e):
    #Start generating the events
    category=np.zeros((len(hits),1))
    track=np.zeros((len(hits),108))
    trackreal=np.zeros((len(hits),68))
    for z in prange(len(hits)):
        m = random.randrange(0,12)#For track finding, set this to m = 7 (only generates dimuons)
        if(m==2) or (m==3):#Single Muon
            j=random.randrange(len(pos_e)) 
            l=random.randrange(0,2)
            if(l==0):
                for k in range(54):
                    if(pos_e[j][k]>0):
                        if(random.random()<0.94) or (k>29):
                            hits[z][k][int(pos_e[j][k]-1)]=1
            if(l==1):
                for k in range(54):
                    if(neg_e[j][k]>0):
                        if(random.random()<0.94) or (k>29):
                            hits[z][k][int(neg_e[j][k]-1)]=1
            category[z][0]=1
        if(m==4) or (m==5):#Two Muons of the Same Sign
            j=random.randrange(len(pos_e))
            l=random.randrange(0,2)
            if(l==0):
                for k in range(54):
                    if(pos_e[j][k]>0):
                        if(random.random()<0.94) or (k>29):
                            hits[z][k][int(pos_e[j][k]-1)]=1
                j=random.randrange(len(pos_e))
                for k in range(54):
                    if(pos_e[j][k]>0):
                        if(random.random()<0.94) or (k>29):
                            hits[z][k][int(pos_e[j][k]-1)]=1
            if(l==1):
                for k in range(54):
                    if(neg_e[j][k]>0):
                        if(random.random()<0.94) or (k>29):
                            hits[z][k][int(neg_e[j][k]-1)]=1
                j=random.randrange(len(pos_e))
                for k in range(54):
                    if(neg_e[j][k]>0):
                        if(random.random()<0.94) or (k>29):
                            hits[z][k][int(neg_e[j][k]-1)]=1
            category[z][0]=2
        if(m==6):#Two Muons of opposite signs
            j=random.randrange(len(pos_e))
            j2=random.randrange(len(neg_e))
            for k in range(54):
                if(pos_e[j][k]>0):
                    if(random.random()<0.94) or (k>29):
                        hits[z][k][int(pos_e[j][k]-1)]=1
                if(neg_e[j2][k]>0):
                    if(random.random()<0.94) or (k>29):
                        hits[z][k][int(neg_e[j2][k]-1)]=1
            category[z][0]=3
        if(m==7):#Dimuon pair
            j=random.randrange(len(pos_e))
            j2=j#random.randrange(len(neg_e))
            for k in range(54):
                if(pos_e[j][k]>0):
                    if(random.random()<0.94) or (k>29):
                        hits[z][k][int(pos_e[j][k]-1)]=1
                    track[z][k]=pos_e[j][k]
                if(neg_e[j2][k]>0):
                    if(random.random()<0.94) or (k>29):
                        hits[z][k][int(neg_e[j2][k]-1)]=1
                    track[z][k+54]=neg_e[j2][k]
            category[z][0]=3
            
        if(m==8) or (m==9):#Three muons of the same sign
            j=random.randrange(len(pos_e)) #Select random event number
            l=random.randrange(0,2)
            if(l==0):
                for k in range(54):
                    if(pos_e[j][k]>0):
                        if(random.random()<0.94) or (k>29):
                            hits[z][k][int(pos_e[j][k]-1)]=1
                j=random.randrange(len(pos_e))
                for k in range(54):
                    if(pos_e[j][k]>0):
                        if(random.random()<0.94) or (k>29):
                            hits[z][k][int(pos_e[j][k]-1)]=1
                j=random.randrange(len(pos_e))
                for k in range(54):
                    if(pos_e[j][k]>0):
                        if(random.random()<0.94) or (k>29):
                            hits[z][k][int(pos_e[j][k]-1)]=1
            if(l==1):
                for k in range(54):
                    if(neg_e[j][k]>0):
                        if(random.random()<0.94) or (k>29):
                            hits[z][k][int(neg_e[j][k]-1)]=1
                j=random.randrange(len(pos_e))
                for k in range(54):
                    if(neg_e[j][k]>0):
                        if(random.random()<0.94) or (k>29):
                            hits[z][k][int(neg_e[j][k]-1)]=1
                j=random.randrange(len(pos_e))
                for k in range(54):
                    if(neg_e[j][k]>0):
                        if(random.random()<0.94) or (k>29):
                            hits[z][k][int(neg_e[j][k]-1)]=1
            category[z][0]=4
            
        if(m==10):#Three muons, one of the opposite sign of the other two.
            j=random.randrange(len(pos_e))
            j2=random.randrange(len(neg_e))
            for k in range(54):
                if(pos_e[j][k]>0):
                    if(random.random()<0.94) or (k>29):
                        hits[z][k][int(pos_e[j][k]-1)]=1
                if(neg_e[j2][k]>0):
                    if(random.random()<0.94) or (k>29):
                        hits[z][k][int(neg_e[j2][k]-1)]=1
            j=random.randrange(len(pos_e)) #Select random event number
            l=random.randrange(0,2)
            if(l==0):
                for k in range(54):
                    if(pos_e[j][k]>0):
                        if(random.random()<0.94) or (k>29):
                            hits[z][k][int(pos_e[j][k]-1)]=1
            if(l==1):
                for k in range(54):
                    if(neg_e[j][k]>0):
                        if(random.random()<0.94) or (k>29):
                            hits[z][k][int(neg_e[j][k]-1)]=1
            category[z][0]=5
        if(m==11): #Dimuon pair and random extra muon
            j=random.randrange(len(pos_e))
            for k in range(54):
                if(pos_e[j][k]>0):
                    if(random.random()<0.94) or (k>29):
                        hits[z][k][int(pos_e[j][k]-1)]=1
                if(neg_e[j][k]>0):
                    if(random.random()<0.94) or (k>29):
                        hits[z][k][int(neg_e[j][k]-1)]=1
            j=random.randrange(len(pos_e)) #Select random event number
            l=random.randrange(0,2)
            if(l==0):
                for k in range(54):
                    if(pos_e[j][k]>0):
                        if(random.random()<0.94) or (k>29):
                            hits[z][k][int(pos_e[j][k]-1)]=1
            if(l==1):
                for k in range(54):
                    if(neg_e[j][k]>0):
                        if(random.random()<0.94) or (k>29):
                            hits[z][k][int(neg_e[j][k]-1)]=1
            category[z][0]=5
    return hits,category,trackreal


# In[7]:


@numba.jit(nopython=True)
def hit_matrix(detectorid,elementid,hits,station): #Convert into hit matrices
    for j in range (len(detectorid)):
        rand = random.random()
        #St 1
        if(station==1) and (rand<0.85):
            if ((detectorid[j]<7) or (detectorid[j]>30)) and (detectorid[j]<35):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
        #St 2
        elif(station==2):
            if (detectorid[j]>12 and (detectorid[j]<19)) or ((detectorid[j]>34) and (detectorid[j]<39)):
                if((detectorid[j]<15) and (rand<0.76)) or ((detectorid[j]>14) and (rand<0.86)) or (detectorid[j]==17):
                    hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
        #St 3
        elif(station==3) and (rand<0.8):
            if (detectorid[j]>18 and (detectorid[j]<31)) or ((detectorid[j]>38) and (detectorid[j]<47)):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
        #St 4
        elif(station==4):
            if ((detectorid[j]>40) and (detectorid[j]<55)):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
        if(rand<0.25):
            if ((detectorid[j]>40) and (detectorid[j]<47)): 
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
    return hits


@numba.jit(nopython=True)
def hit_matrix_mc(detectorid,elementid,hits,station): #Convert into hit matrices
    for j in prange (len(detectorid)):
        #St 1
        if(station==1):
            if ((detectorid[j]<7) or (detectorid[j]>30)) and (detectorid[j]<35):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
        #St 2
        elif(station==2):
            if (detectorid[j]>12 and (detectorid[j]<19)) or ((detectorid[j]>34) and (detectorid[j]<39)):
                if((detectorid[j]<15) and (rand<0.76)) or ((detectorid[j]>14) and (rand<0.86)) or (detectorid[j]==17):
                    hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
        #St 3
        elif(station==3):
            if (detectorid[j]>18 and (detectorid[j]<31)) or ((detectorid[j]>38) and (detectorid[j]<47)):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
        #St 4
        elif(station==4):
            if ((detectorid[j]>40) and (detectorid[j]<55)):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
    return hits


def generate_e906(n_events, tvt):
    #Import NIM3 events and put them on the hit matrices.
    #Randomly choosing between 1 and 6 events per station gets approximately the correct occupancies.
    filelist=['output_part1.root:tree_nim3','output_part2.root:tree_nim3','output_part3.root:tree_nim3',
             'output_part4.root:tree_nim3','output_part5.root:tree_nim3','output_part6.root:tree_nim3',
             'output_part7.root:tree_nim3']
    targettree = uproot.open("NIM3/"+random.choice(filelist))
    detectorid_nim3=targettree["det_id"].arrays(library="np")["det_id"]
    elementid_nim3=targettree["ele_id"].arrays(library="np")["ele_id"]
    hits = np.zeros((n_events,54,201))
    for n in range (n_events): #Create NIM3 events
        g=random.choice([0,1,2,3,4,5,6])
        for m in range(g):
            i=random.randrange(len(detectorid_nim3))
            hits[n]=hit_matrix(detectorid_nim3[i],elementid_nim3[i],hits[n],1)
            i=random.randrange(len(detectorid_nim3))
            hits[n]=hit_matrix(detectorid_nim3[i],elementid_nim3[i],hits[n],2)
            i=random.randrange(len(detectorid_nim3))
            hits[n]=hit_matrix(detectorid_nim3[i],elementid_nim3[i],hits[n],3)
            i=random.randrange(len(detectorid_nim3))
            hits[n]=hit_matrix(detectorid_nim3[i],elementid_nim3[i],hits[n],4)
    del detectorid_nim3, elementid_nim3
    
    #Place the full tracks that are reconstructable
    if(tvt=="Train"):
        hits,category,track=track_injection(hits,pos_events,neg_events)    
    if(tvt=="Val"):
        hits,category,track=track_injection(hits,pos_events_val,neg_events_val)    
    return hits.astype(bool), category.astype(int), track.astype(int)



def evaluate(testin,testsignals,modelname):    
    # Use the trained model to predict the particle ID for each
    model = keras.models.load_model('Networks/'modelname)
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(testin,verbose=0)
    testpred=np.zeros(len(predictions))
    for i in range(len(testpred)):
        if predictions[i][1]>0.75:testpred[i]=1
        if predictions[i][2]>0.75:testpred[i]=2
        if predictions[i][3]>0.75:testpred[i]=3
        if predictions[i][4]>0.75:testpred[i]=4
        if predictions[i][5]>0.75:testpred[i]=5
    #testpred = np.argmax(predictions, axis=1)

    # Create arrays for true/false positives/negatives
    tp = np.zeros(6)
    tn = np.zeros(6)
    fp = np.zeros(6)
    fn = np.zeros(6)

    # Loop through the predictions and true signals to calculate true/false positives/negatives
    for i in range(len(testpred)):
        pred = testpred[i]
        signal = testsignals[i]
        for j in range(6):
            if pred == j:
                if signal == j:
                    tp[j] += 1
                else:
                    fp[j] += 1
                for k in range(6):
                    if k != j and signal != k:
                        tn[k] += 1
                    elif k != j and signal == k:
                        fn[k] += 1

    # Calculate precision and recall for each category
    precision = np.zeros(6)
    recall = np.zeros(6)
    for j in range(6):
        if tp[j] + fp[j] > 0:
            precision[j] = tp[j] / (tp[j] + fp[j])
        if tp[j] + fn[j] > 0:
            recall[j] = tp[j] / (tp[j] + fn[j])

    # Print the results
    print(precision[0],recall[0],precision[1],recall[1],precision[2],recall[2],precision[3],recall[3],precision[4],recall[4],precision[5],recall[5])


# In[10]:


learning_rate_filter=1e-6
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
n_train = 0


# In[11]:

print("Before while loop:", n_train)
while(n_train<1e7):
    trainin, trainsignals, traintrack = generate_e906(500000, "Train")
    n_train+=len(trainin)
    del traintrack
    print("Generated Training Data")
    valin, valsignals, valtrack = generate_e906(50000, "Val")
    del valtrack
    print("Generated Validation Data")
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()
    model = keras.models.load_model('Networks/event_filter')
    optimizer = tf.keras.optimizers.Adam(learning_rate_filter)
    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    val_loss_before=model.evaluate(valin,valsignals,batch_size=256,verbose=2)[0]
    history = model.fit(trainin, trainsignals,
                    epochs=1000, batch_size=256, verbose=2, validation_data=(valin,valsignals),callbacks=[callback])
    if(min(history.history['val_loss'])<val_loss_before):
        model.save('Networks/event_filter')
    evaluate(valin,valsignals,"event_filter")
    print('\n')
    tf.keras.backend.clear_session()
    del trainsignals,trainin,valin,valsignals,model
    gc.collect()  # Force garbage collection to release GPU memory
    print(n_train)
    if(n_train%2000000==0):learning_rate_filter /=10

