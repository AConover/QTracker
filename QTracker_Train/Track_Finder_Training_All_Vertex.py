#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import uproot
from numba import njit, prange
import random
import tensorflow as tf
import gc


# In[2]:


#Import MC Events
print("Reading ROOT files...")
targettree = uproot.open('Root_Files/All_Vertex_Train_QA_v2.root:QA_ana')
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
targettree = uproot.open('Root_Files/All_Vertex_Val_QA_v2.root:QA_ana')
targetdata = targettree.arrays(library="np")
targetevents=len(targetdata['n_tracks'])
print("Done")

#This reads the dimuon tracks from the target into an array
pos_events_val=np.zeros((targetevents,54))
neg_events_val=np.zeros((targetevents,54))
print("Reading target events...")
for j in range(targetevents):
    #if(j%100==0):print(j,end="\r") #This is to keep track of how quickly the events are being generated
    first=targetdata['pid'][j][0]
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


# In[7]:


@njit(parallel=True)
def track_injection(hits,pos_e,neg_e):
    #Start generating the events
    category=np.zeros((len(hits),1))
    track=np.zeros((len(hits),108))
    trackreal=np.zeros((len(hits),68))
    for z in prange(len(hits)):
        m = 7#random.randrange(0,12)#For track finding, set this to m = 7 (only generates dimuons)
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
            #Re-writes track information so that parallel detectors are stored in the same entry.
        trackreal[z][0]=track[z][0]    
        trackreal[z][1]=track[z][1]
        trackreal[z][2]=track[z][2]
        trackreal[z][3]=track[z][3]
        trackreal[z][4]=track[z][4]
        trackreal[z][5]=track[z][5]
        trackreal[z][6]=track[z][12]
        trackreal[z][7]=track[z][13]
        trackreal[z][8]=track[z][14]
        trackreal[z][9]=track[z][15]
        trackreal[z][10]=track[z][16]
        trackreal[z][11]=track[z][17]
        if(track[z][18]>0):
            trackreal[z][12]=track[z][18]
            trackreal[z][13]=track[z][19]
            trackreal[z][14]=track[z][20]
            trackreal[z][15]=track[z][21]
            trackreal[z][16]=track[z][22]
            trackreal[z][17]=track[z][23]
        else:
            trackreal[z][12]=-track[z][24]
            trackreal[z][13]=-track[z][25]
            trackreal[z][14]=-track[z][26]
            trackreal[z][15]=-track[z][27]
            trackreal[z][16]=-track[z][28]
            trackreal[z][17]=-track[z][29]
        
        if(track[z][30]>0):trackreal[z][18]=track[z][30]
        else:trackreal[z][18]=-track[z][31]
        if(track[z][32]>0):trackreal[z][19]=track[z][32]
        else:trackreal[z][19]=-track[z][33]
        
        if(track[z][34]>0):trackreal[z][20]=track[z][34]
        else:trackreal[z][20]=-track[z][35]
        if(track[z][36]>0):trackreal[z][21]=track[z][36]
        else:trackreal[z][21]=-track[z][37]
            
        if(track[z][38]>0):trackreal[z][22]=track[z][38]
        else:trackreal[z][22]=-track[z][39]
            
        if(track[z][40]>0):trackreal[z][23]=track[z][40]
        else:trackreal[z][23]=-track[z][41]
        if(track[z][42]>0):trackreal[z][24]=track[z][42]
        else:trackreal[z][24]=-track[z][43]
        if(track[z][44]>0):trackreal[z][25]=track[z][44]
        else:trackreal[z][25]=-track[z][45]
            
            
        trackreal[z][26]=track[z][46]
        trackreal[z][27]=track[z][47]
        trackreal[z][28]=track[z][48]
        trackreal[z][29]=track[z][49]
        trackreal[z][30]=track[z][50]
        trackreal[z][31]=track[z][51]
        trackreal[z][32]=track[z][52]
        trackreal[z][33]=track[z][53]
        
        
        rc = 34
        tc = 54
        trackreal[z][0+34]=track[z][0+54]
        trackreal[z][1+34]=track[z][1+54]
        trackreal[z][2+34]=track[z][2+54]
        trackreal[z][3+34]=track[z][3+54]
        trackreal[z][4+34]=track[z][4+54]
        trackreal[z][5+34]=track[z][5+54]
        trackreal[z][6+34]=track[z][12+54]
        trackreal[z][7+34]=track[z][13+54]
        trackreal[z][8+34]=track[z][14+54]
        trackreal[z][9+34]=track[z][15+54]
        trackreal[z][10+34]=track[z][16+54]
        trackreal[z][11+34]=track[z][17+54]
        if(track[z][18+54]>0):
            trackreal[z][12+34]=track[z][18+54]
            trackreal[z][13+34]=track[z][19+54]
            trackreal[z][14+34]=track[z][20+54]
            trackreal[z][15+34]=track[z][21+54]
            trackreal[z][16+34]=track[z][22+54]
            trackreal[z][17+34]=track[z][23+54]
        else:
            trackreal[z][12+34]=-track[z][24+54]
            trackreal[z][13+34]=-track[z][25+54]
            trackreal[z][14+34]=-track[z][26+54]
            trackreal[z][15+34]=-track[z][27+54]
            trackreal[z][16+34]=-track[z][28+54]
            trackreal[z][17+34]=-track[z][29+54]
        
        if(track[z][30+54]>0):trackreal[z][18+34]=track[z][30+54]
        else:trackreal[z][18+34]=-track[z][31+54]
        if(track[z][32+54]>0):trackreal[z][19+34]=track[z][32+54]
        else:trackreal[z][19+34]=-track[z][33+54]
        
        if(track[z][34+54]>0):trackreal[z][20+34]=track[z][34+54]
        else:trackreal[z][20+34]=-track[z][35+54]
        if(track[z][36+54]>0):trackreal[z][21+34]=track[z][36+54]
        else:trackreal[z][21+34]=-track[z][37+54]
            
        if(track[z][38+54]>0):trackreal[z][22+34]=track[z][38+54]
        else:trackreal[z][22+34]=-track[z][39+54]
            
        if(track[z][40+54]>0):trackreal[z][23+34]=track[z][40+54]
        else:trackreal[z][23+34]=-track[z][41+54]
        if(track[z][42+54]>0):trackreal[z][24+34]=track[z][42+54]
        else:trackreal[z][24+34]=-track[z][43+54]
        if(track[z][44+54]>0):trackreal[z][25+34]=track[z][44+54]
        else:trackreal[z][25+34]=-track[z][45+54]
            
            
        trackreal[z][26+34]=track[z][46+54]
        trackreal[z][27+34]=track[z][47+54]
        trackreal[z][28+34]=track[z][48+54]
        trackreal[z][29+34]=track[z][49+54]
        trackreal[z][30+34]=track[z][50+54]
        trackreal[z][31+34]=track[z][51+54]
        trackreal[z][32+34]=track[z][52+54]
        trackreal[z][33+34]=track[z][53+54]
    return hits,category,trackreal


# In[8]:


@njit(nopython=True)
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


@njit(nopython=True)
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


# In[9]:


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


# In[10]:


max_ele = [200, 200, 168, 168, 200, 200, 128, 128,  112,  112, 128, 128, 134, 134, 
           112, 112, 134, 134,  20,  20,  16,  16,  16,  16,  16,  16,
        72,  72,  72,  72,  72,  72,  72,  72, 200, 200, 168, 168, 200, 200,
        128, 128,  112,  112, 128, 128, 134, 134, 112, 112, 134, 134,
        20,  20,  16,  16,  16,  16,  16,  16,  72,  72,  72,  72,  72,
        72,  72,  72]


# In[14]:


learning_rate_finder=1e-6
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
n_train=0

# In[ ]:

print("Before while loop:", n_train)
while(n_train<8e6):
    trainin, trainsignals, traintrack = generate_e906(750000, "Train")
    print("Generated Training Data")
    traintrack = traintrack/max_ele
    #del traintrack
    
    valin, valsignals, valtrack = generate_e906(75000, "Val")
    print("Generated Validation Data")
    valtrack = valtrack/max_ele
    #del valtrack
    
    tf.keras.backend.clear_session()
    
    model = tf.keras.models.load_model('event_filter')
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    val_predictions = probability_model.predict(valin,batch_size=225)
    train_predictions = probability_model.predict(trainin,batch_size=225)


    trainin=trainin[train_predictions[:,3]>0.75]
    traintrack=traintrack[train_predictions[:,3]>0.75]

    valin=valin[val_predictions[:,3]>0.75]
    valtrack=valtrack[val_predictions[:,3]>0.75]
    
    n_train+=len(trainin)
    
    del train_predictions, val_predictions
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    # Specify the optimizer, and compile the model with loss functions for both outputs
    model = tf.keras.models.load_model('Track_Finder_All')
    optimizer = tf.keras.optimizers.Adam(learning_rate_finder)
    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.mse,
              metrics=tf.keras.metrics.RootMeanSquaredError())

    val_loss_before=model.evaluate(valin,valtrack,batch_size=100,verbose=2)[0]
    print(val_loss_before)
    history = model.fit(trainin, traintrack,
                    epochs=1000, batch_size=100, verbose=2, validation_data=(valin,valtrack),callbacks=[callback])
    if(min(history.history['val_loss'])<val_loss_before):
        model.save('Track_Finder_All')
        learning_rate_finder=learning_rate_finder*2
    learning_rate_finder=learning_rate_finder/2
    tf.keras.backend.clear_session()
    del trainin, valin, traintrack, valtrack,model
    gc.collect()  # Force garbage collection to release GPU memory
    print(n_train)


