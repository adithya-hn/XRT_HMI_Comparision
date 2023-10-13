# =========
# Improved code of Al_mesh intensity study
# > Uses list of Al mesh instead of searching, Status print is added
# > Centre correction done peoperly
# > Dedicated code for XBP using source extraxtor
#   * Fuzzy region fix need to be done for XBPs 
#   * Center correction shoud be done for source extractor segment file also84
# output values kept to 6 decimals-accurate up to 30 sec DOB can be used as Unique ID-use DOB code to ge back the date
#
# >> AR regejcted above 100 pix
#
# 16/7/23 [Version 2]
#  > Radius
#  > New date
#  > 
# =========


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.io import fits
from astropy import visualization as av
from astropy.time import Time
import scipy.misc
import math as mt
from jdcal import gcal2jd, jd2gcal
import datetime
import os
import timeit
import scipy as sp
import cv2
from cv2 import contourArea
from cv2 import drawContours
from cv2 import findContours
from cv2 import cvtColor
import pathlib
import imageio
import numpy.ma as ms
from scipy.ndimage import label, generate_binary_structure, find_objects, measurements, map_coordinates, shift
import scipy.stats as si
import zipfile
import subprocess
import sys
from math import sqrt
from reproject import reproject_interp
from astropy.wcs import WCS
import copy

from skimage.transform import resize
from skimage import data, color



from skimage.transform import rescale, resize, downscale_local_mean

startTime = timeit.default_timer()
totelIm = 0
filetime=0
prvTime=startTime

pathlib.Path("Al_XRT").mkdir(parents=True, exist_ok=True)
pathlib.Path("HMI_imgs").mkdir(parents=True, exist_ok=True)
# pathlib.Path("Histograms").mkdir(parents=True, exist_ok=True)

CHarray = []
BParray = []
ARarray = []
BGarray = []
FDarray = []

mCHarray = []
mBParray = []
mARarray = []
mBGarray = []
mFDarray = []
#mBParrayD=[]
#mBParrayB=[]

CHarea = []
BParea = []
ARarea = []
BGarea = []

CHa = []
BPa = []
ARa = []
BGa = []
FDa = []

CHi = []
BPi = []
ARi = []
BGi = []
Fdi = []

CHim = []
BPim = []
ARim = []
BGim = []
Fdim = []

n_AR = []
n_BP = []
n_CH = []
l_DOB = []
Ls_mFln =[]
Ls_Fln = []
l_CHth = []
l_ARth = []
l_BPth = []
l_size = []
shapeArry = []

l_LD=[]
l_TH=[]

l_BTxbp = []
l_DIMxbp = []

sf = np.loadtxt('V1_xrt_HMI_Pairs_D08_10_23.dat', dtype='str')
Segfold='/media/adithyahn/New Volume/DB_Irradiance/DB_V1/RGNmaps_DB_V1'
#mf = np.loadtxt('hmi_List.dat', dtype='str').transpose()

Length = sf.shape[0]
print(Length)

twoKcount = 0  # two count how many 2k images are there
start = 0
#print(sf[1][1],sf[1][4])
for l in range(Length):
    try:
        mFname=(os.path.splitext((sf[l][1].split(os.sep))[-1]))[0]
        Fname=(os.path.splitext((sf[l][2].split(os.sep))[-1]))[0]
        seg_zip_file=Segfold+'/'+Fname+'_seg.fits.zip'
        seg_file=Fname+'_seg.fits' 
        
        with zipfile.ZipFile(seg_zip_file, 'r') as zip_file:
          zip_file.extract(seg_file)
        sg = fits.open(seg_file)  
        sgData=sg[0].data
        sgData_=sg[0].data
        
        x1scale = sg[0].header['XSCALE']
        y1scale = sg[0].header['YSCALE']
        Rad=(sg[0].header['RSUN_OBS']/x1scale)
        Rad_=copy.deepcopy(Rad)
        segx1cen =int(sg[0].header['XCEN']/x1scale)
        segy1cen =int(sg[0].header['YCEN']/x1scale)
        img2 = fits.open(sf[l][1])
        scidata2 = np.rot90(img2[1].data,2)
        img1 = fits.open(sf[l][2])
        scidata1 = img1[0].data
        hmi_R=int(img2[1].header['RSUN_OBS']/img2[1].header['CDELT1'])
        mh=int(img2[1].header['CRPIX1'])
        mk=int(img2[1].header['CRPIX2'])
        xcen=int(img1[0].header['CRPIX1'])
        ycen=int(img1[0].header['CRPIX2'])
        DOB2=img2[1].header['DATE-OBS']
        DOB1 = sg[0].header['DATE_OBS']
        size1 = sgData.shape
        
        hmi_r_sc=int(hmi_R/Rad)
        hmi_R_=hmi_R*0.95
        RC=hmi_R/Rad
        #print(RC)
        center = (mh,mk)#(int(mh-(segx1cen*hmi_r_sc)),int(mk-(segy1cen*hmi_r_sc)))
        #scidata1 = shift(scidata1, (-(1024-center[0]), -(1024-center[1])), cval=1)
        Rad=Rad*RC
        Reduce=int(Rad*0.05) #5% of radius
        Radlb=int(Rad*1.05)
        R_ = int(Rad_*0.95)
        R  = int(Rad*0.95)
        h_=segx1cen
        k_=segy1cen
        start=1


    except:
        print('**Exception**')
        print('error is:', sys.exc_info()[0], sf[l][1])
        start=0
        pass

    if start == 1:
        size1 = scidata1.shape
        size = size1
        k4=scidata2.shape
        shapeArry.append(size1[0])
        dob_str = DOB1
        dob_obj = datetime.datetime.strptime(dob_str, '%Y-%m-%dT%H:%M:%S.%f')
        dtm = Time(dob_obj,format='datetime') #date and time
        obsDate=dtm.decimalyear
        l_DOB.append(np.round(obsDate, 6))
        aa = np.zeros((size1))
        dim=int(size[0]*RC)
        extLn=dim-4096
        #print(dim-4096)

        sgData=resize(sgData,(4096, 4096)) 
        sgData=sgData[0:4096,0:4096]
        xs=2048-mh+int(segx1cen*hmi_r_sc)
        ys=2048-mk+int(segy1cen*hmi_r_sc)
        mxs=int(2048-mh)
        mys=int(2048-mk)
        sgData=np.round(sgData,3)

        CHmask = (np.where(sgData_==8,255,0)).astype(np.uint8)
        BPmask =(np.where(sgData_==16,255,0)).astype(np.uint8)
        ARmask =(np.where(sgData_==32,255,0)).astype(np.uint8)
        BGmask = (np.where(sgData_==2,255,0)).astype(np.uint8)

        #sgData=shift(sgData,(xs,ys),cval=1)
        sgData=np.roll(sgData,shift=(xs,ys),axis=(1,1)) #correcting for HMI img
        #sgData=np.roll(sgData,shift=(-mxs,-mys),axis=(1,1)) #correcting for HMI img
        #scidata2=np.roll(scidata2,shift=(mxs,mys),axis=(1,1)) #correcting for HMI img
        CHmask_sc  =(np.where(sgData==0.031,255,0)).astype(np.uint8) #0.031372549
        BPmask_sc =(np.where(sgData==0.063,255,0)).astype(np.uint8) #0.062745098
        ARmask_sc =(np.where(sgData==0.125,255,0)).astype(np.uint8)
        BGmask_sc  =(np.where(sgData==0.008,255,0)).astype(np.uint8) #0.00784313725490196        
 
        kernel = np.ones((15, 15), np.uint8)
        kernel1 = np.ones((30, 30), np.uint8)
        CHmask = cv2.morphologyEx(CHmask, cv2.MORPH_CLOSE, kernel)
        CHmask_sc = cv2.morphologyEx(CHmask_sc, cv2.MORPH_CLOSE, kernel)

        CH_ConT, hierarchy = cv2.findContours(CHmask_sc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        BP_ConT, hierarchy = cv2.findContours(BPmask_sc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        AR_ConT, hierarchy = cv2.findContours(ARmask_sc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #BG_ConT, hierarchy = cv2.findContours(BGmask_sc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        CH_cont, hierarchy = cv2.findContours(CHmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        BP_cont, hierarchy = cv2.findContours(BPmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        AR_cont, hierarchy = cv2.findContours(ARmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #BG_cont, hierarchy = cv2.findContours(BGmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        MarMsk =np.zeros(k4, np.uint8)
        MbpMsk =np.zeros(k4, np.uint8)
        MchMsk =np.zeros(k4, np.uint8)
        MbgMsk =np.zeros(k4, np.uint8)
        #MbpMskB =np.zeros(k4, np.uint8)
        #MbpMskD =np.zeros(k4, np.uint8)

        #print('center',center)
        h = int(center[0])
        k = int(center[1])
        #hmi_r_sc=hmi_R/Rad
        #print(h_,k_,R_)


        circ = cv2.circle(aa, (h_, k_), R_, (255, 0, 0), -1)  # disk
        circle = circ.astype(np.bool_)
        Circle = np.invert(circle)  # hole
        mask = ms.array(scidata1, mask=Circle)
        disk = ms.array(scidata1, mask=circle)  # hided disk
        DD = disk * 0
        dd = ms.array(DD, mask=Circle)
        DiskCirc = Circle.astype(np.uint8)  # for sep
        sun = DD.data  # only solar disk
 
        
        zimage = av.ZScaleInterval(n_samples=600, contrast=0.25, max_reject=0.5, min_npixels=5, krej=2.5,max_iterations=5)
        z = zimage.get_limits(scidata1)
        ZI = np.clip(scidata1, z[0], z[1])  # zimage(scidata)
        #print(z,ZI)
        ZS = (ZI / ZI.max()) * 255  #
        SUN = ((ZS).astype(np.uint8))
        
          
        zm = zimage.get_limits(scidata2.astype(np.uint32)) 
        ZIm = np.clip((scidata2.astype(np.uint32)), zm[0], zm[1])  # zimage(scidata)
        ZSm = (ZIm / ZIm.max()) * 255  #
        SUNm = ZSm.astype(np.uint8)
        comb_img_m = cvtColor(SUNm, cv2.COLOR_GRAY2BGR)
        
        aaM=np.zeros((scidata2.shape))
        cirM = cv2.circle(aaM, (mh,mk), int(R), (255, 0, 0), -1)  # disk

        circleM = cirM.astype(np.bool_)
        CircleM = np.invert(circleM)
        Mdisk = ms.array(scidata2, mask=circleM)  # hided disk
        mDD = Mdisk * 0
        Mdd = ms.array(mDD, mask=CircleM)
        #Msun = mDD.data
        Msun    = abs(np.where(CircleM,0,scidata2)) 
        MsunJ=(Msun).astype(np.uint8)
        #imageio.imwrite('M_sun{}.jpg'.format(Fname),MsunJ)
        
        #--------------

        comb_img = cvtColor(SUN, cv2.COLOR_GRAY2BGR)
        Comb_img = cvtColor(SUN, cv2.COLOR_GRAY2BGR)  # for background
        #Org_img = cvtColor(SUN, cv2.COLOR_GRAY2BGR)
        #CH_img = cvtColor(SUN, cv2.COLOR_GRAY2BGR)
        #BP_img = cvtColor(SUN, cv2.COLOR_GRAY2BGR)
        #AR_img = cvtColor(SUN, cv2.COLOR_GRAY2BGR)
        #AR_img1 = cvtColor(SUN, cv2.COLOR_GRAY2BGR)
        BG_img = cvtColor(MsunJ, cv2.COLOR_GRAY2BGR)
        BG_img= np.where(BGmask_sc,0,MsunJ)

        '''
        nAR = len(AR_cont)  
        nch = len(CH_cont)
        #CH_cont = [CHcont[ch] for ch in range (nch) if (contourArea(CHcont[ch]))>10]
        for ch in range(nch):
            ch_a = cv2.contourArea(CH_cont[ch])
            xrt_chCor=(CH_cont[ch]+[((size[0]/2)-int(h)),((size[0]/2)-int(k))])*hmi_r_sc 
            CH_ConT.append((xrt_chCor+[mh-2048,mk-2048]).astype(int))
                

        for j in range(nAR):
            b1 = contourArea(AR_cont[j])
            ARarea.append(b1)
            xrt_arCor=(AR_cont[j]+[((size[0]/2)-int(h)),((size[0]/2)-int(k))])*hmi_r_sc #AR coord correction
            AR_ConT.append((xrt_arCor+[mh-2048,mk-2048]).astype(int))
          

        for b in range (len(BP_cont)):
            bp_area= cv2.contourArea(BP_cont[b])
            xrt_bpCor=(BP_cont[b]+[((size[0]/2)-int(h)),((size[0]/2)-int(k))])*hmi_r_sc #AR coord correction
            BP_ConT.append((xrt_bpCor+[mh-2048,mk-2048]).astype(int))
            
        '''
        
        #drawContours(MarMsk, AR_ConT, -1, (255, 0, 0), cv2.FILLED)
        #drawContours(MchMsk, CH_ConT, -1, (255, 0, 0), cv2.FILLED)
        #drawContours(MbpMsk, BP_ConT, -1, (255, 0, 0), cv2.FILLED)
        #drawContours(MbpMskB, BP_ConTB, -1, (255, 0, 0), cv2.FILLED)
        #drawContours(MbpMskD, BP_ConTD, -1, (255, 0, 0), cv2.FILLED)
        #drawContours(MbgMsk, BP_ConT, -1, (255, 0, 0), cv2.FILLED)
        #drawContours(MbgMsk, AR_ConT, -1, (255, 0, 0), cv2.FILLED)
        #drawContours(MbgMsk, CH_ConT, -1, (255, 0, 0), cv2.FILLED)
        
        MarMsk=ARmask_sc
        MchMsk=CHmask_sc
        MbgMsk=BGmask_sc
        MbpMsk=BPmask_sc
        

        drawContours(comb_img, CH_cont, -1, (255, 0, 255), 2)
        drawContours(comb_img, BP_cont, -1, (250, 0, 0), 1)
        drawContours(comb_img, AR_cont, -1, (255, 255, 0), 2)
        drawContours(comb_img_m,AR_ConT,-1, (250, 255, 0), 8)
        drawContours(comb_img_m,BP_ConT,-1, (250, 0, 0), 8)
        drawContours(comb_img_m,CH_ConT,-1, (250, 0, 255), 8)
        #drawContours(BG_img,BG_ConT,-1, (250, 0, 255),cv2.FILLED)

        Bo_CHmask = CHmask.astype(np.bool_)
        Bo_ARmask1= ARmask.astype(np.bool_)
        Bo_BPmask = BPmask.astype(np.bool_)
        Bo_BGmask = BGmask.astype(np.bool_)
        Bo_MchMsk = MchMsk.astype(np.bool_)
        Bo_MarMsk = MarMsk.astype(np.bool_)
        Bo_MbpMsk = MbpMsk.astype(np.bool_)
        #Bo_MbpMskB = MbpMskB.astype(np.bool_)
        #Bo_MbpMskD = MbpMskD.astype(np.bool_)
        Bo_MbgMsk = MbgMsk.astype(np.bool_)
        
        In_BGmask = np.invert(Bo_BGmask)
        In_CHmask = np.invert(Bo_CHmask)
        In_ARmask1 =np.invert(Bo_ARmask1)
        In_BPmask = np.invert(Bo_BPmask)
        
        
        In_MchMsk = np.invert(Bo_MchMsk)
        In_MarMsk = np.invert(Bo_MarMsk)
        In_MbpMsk = np.invert(Bo_MbpMsk)
        #In_MbpMskB = np.invert(Bo_MbpMskB)
        #In_MbpMskD = np.invert(Bo_MbpMskD)
        In_MbgMsk = np.invert(Bo_MbgMsk)

        # masking
        CH_masked_sun = ms.array(sun, mask=In_CHmask)
        BG_masked_sun = ms.array(sun, mask=In_BGmask)  # mask all fearure
        AR_masked1_sun =ms.array(sun, mask=In_ARmask1)
        BP_masked_sun = ms.array(sun, mask=In_BPmask)
        
        CH_Msun = ms.array(Msun, mask=In_MchMsk)
        BG_Msun = ms.array(Msun, mask=In_MbgMsk)  # mask all fearure
        AR_Msun = ms.array(Msun, mask=In_MarMsk)
        BP_Msun = ms.array(Msun, mask=In_MbpMsk)
        #BP_MsunB = ms.array(Msun, mask=In_MbpMskB)
        #BP_MsunD = ms.array(Msun, mask=In_MbpMskD)

        # Numebrs'
        no_of_AR = (len(AR_cont))
        no_of_BP = (len(BP_cont))  # total xbps, including ar excluded xbps
        no_of_CH = len(CH_cont)
        n_AR.append(no_of_AR)
        n_BP.append(no_of_BP)
        n_CH.append(no_of_CH)

        # Totel area
        cha = np.count_nonzero(MchMsk)
        bpa = np.count_nonzero(MbpMsk)
        ara = np.count_nonzero(MarMsk)
        bga=np.count_nonzero(MbgMsk)


        A4 = np.count_nonzero(In_MbgMsk)  # Bg + oudside disk
        A5 = np.pi *hmi_R_*hmi_R_#np.count_nonzero(Msun)  # fuldisk size

        CHa.append(cha)
        BPa.append(bpa)
        ARa.append(ara)
        BGa.append(bga)
        FDa.append(A5)

        #print('Total area',A6 ,'FD',bga+ara+bpa+cha+A7)
        #print('A4',A4,'=',ara+bpa+cha)

        # Totel Intensity
        bpsum = BP_masked_sun.sum()
        chsum = CH_masked_sun.sum()
        bgsum = BG_masked_sun.sum()
        arsum = AR_masked1_sun.sum()
        
        Mbpsum  = BP_Msun.sum()
        #MbpsumB  = BP_MsunB.sum()
        #MbpsumD  = BP_MsunD.sum()
        Mchsum  = CH_Msun.sum()
        Mbgsum  = BG_Msun.sum()
        Marsum  = AR_Msun.sum()
        
        tsum = sun.sum()
        Mtsum = Msun.sum()

        if no_of_BP == 0:
            bgsum = 0
            Mbgsum = 0
        if no_of_CH == 0:
            chsum = 0
            Mchsum = 0
        if no_of_AR == 0:
            arsum = 0
            Marsum = 0
        Totel = chsum + bgsum + bpsum + arsum
        mTotel = Mchsum + Mbgsum + Mbpsum + Marsum
        
        #print('tot',(mTotel-Mtsum))
        #print()
        # print(chsum,arsum,bgsum,bgsum,Totel)
        # print('Number', no_of_CH,no_of_AR)

        CHarray.append(chsum)
        BParray.append(bpsum)
        ARarray.append(arsum)
        BGarray.append(bgsum)
        FDarray.append(tsum)
        
        mCHarray.append(Mchsum)
        mBParray.append(Mbpsum)
        #mBParrayB.append(MbpsumB)
        #mBParrayD.append(MbpsumD)
        mARarray.append(Marsum)
        mBGarray.append(Mbgsum)
        mFDarray.append(Mtsum)

        ch_p = (((chsum) / Totel) * 100)
        bp_p = (((bpsum) / Totel) * 100)
        ar_p = (((arsum) / Totel) * 100)
        bg_p = (((bgsum) / Totel) * 100)
        
        Mch_p = (((Mchsum) / mTotel) * 100)
        Mbp_p = (((Mbpsum) / mTotel) * 100)
        Mar_p = (((Marsum) / mTotel) * 100)
        Mbg_p = (((Mbgsum) / mTotel) * 100)
        # print('%',ch_p,ar_p)

        CHi.append(ch_p)
        BPi.append(bp_p)
        ARi.append(ar_p)
        BGi.append(bg_p)
        
        CHim.append(Mch_p)
        BPim.append(Mbp_p)
        ARim.append(Mar_p)
        BGim.append(Mbg_p)

        Ls_Fln.append(Fname)
        Ls_mFln.append(mFname)
       
        color = (255, 255, 10)
        color1 = (255, 165, 0) 
        color2 = (64, 254, 208)
        cv2.circle(comb_img,(h,k), R_, color2, 1)
        #print('Rad_',R)
        #cv2.circle(comb_img_m,(h,k), int(R), color1, 4)
        cv2.circle(comb_img_m,(mh,mk), int(hmi_R_), color2, 4)
        imageio.imwrite('Al_XRT/{}.jpg'.format(Fname), comb_img[::-1])
        imageio.imwrite('HMI_imgs/{}.jpg'.format(mFname), comb_img_m[::-1])
        #imageio.imwrite('Imgs/{}.jpg'.format(Fname), SUN[::-1])
        #imageio.imwrite('xbp_removed/XBP_{}.jp
        # g'.format(Fname), BGI[::-1])
        
        os.remove("{}_seg.fits".format(Fname))

        f = open('Al_X-ray_data.dat', 'a')
        np.savetxt('Al_X-ray_data.dat', np.c_[
            CHarray, BParray, ARarray, BGarray, CHi, BPi, ARi, BGi, n_BP, n_AR, n_CH, l_DOB, CHa, BPa, ARa, BGa, FDarray, FDa, shapeArry],
                   fmt='%11.6f',
                   header=' CH Int,   XBP Int,     AR Int,  Background | %intensity-CH   XBP          AR         BG		nXBP		nAR		nCH		l_DOB	    CHa       BPa       ARa        BGa     FD-int    FDa       Shape')

        f.close()

        fa1 = open('Irradianc_Database_Al_mesh.dat', 'a')  # area, bg and xbp
        np.savetxt('Irradianc_Database_Al_mesh.dat', np.c_[
            Ls_Fln, l_DOB, FDarray, ARarray, BParray, CHarray, BGarray, ARi, BPi, CHi, BGi, n_AR, n_BP, n_CH, ARa, BPa, CHa, BGa, FDa, shapeArry],
                   fmt='%s',
                   header=' File name                    DOB | FDI | ARint | XBPint | CHint | BGint | AR% | XBP% | CH% | BG% | nAR | nXBP | nCH | ARa | XBPa | CHa | BGa | FDa | Shape')
        fa1.close()
        
        fa2 = open('Mag_Flux_Database_Al_mesh.dat', 'a')  # area, bg and xbp
        np.savetxt('Mag_Flux_Database_Al_mesh.dat', np.c_[
            Ls_Fln,Ls_mFln, l_DOB, mFDarray, mARarray, mBParray, mCHarray, mBGarray,FDa,ARa,BPa,CHa,BGa,n_AR,n_BP,n_CH,shapeArry],
                   fmt='%s',
                   header='XRT File name                |HMI File name          DOB | FDm | ARm | XBPm | CHm | BGm |FDa | ARa | XBPa | CHa | BGa |  ARn | XBPn | CHn | Shape ')
        fa2.close()
        tempstopTime = timeit.default_timer()
        filetime = np.round((tempstopTime - prvTime),2)
        prvTime=tempstopTime
        print('[', l + 1, '/', Length, ']', DOB1,DOB2,'[Time:',filetime,']')

stopTime = timeit.default_timer()
runtime = (stopTime - startTime)
TotTime = runtime / 3600  # in Hours

print('')
print('......COMPLETED.....')
print('Time taken', TotTime, 'Avg. Time per image', runtime / Length)
print('--------------------------------------------------')
