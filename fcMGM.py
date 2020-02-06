import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import fcsparser

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import multivariate_normal
from scipy.stats import chi
from scipy.spatial import distance

from matplotlib.patches import Ellipse
from matplotlib.ticker import NullFormatter

import scipy.stats as st
import seaborn as sns
from matplotlib.widgets import Slider, Button, RadioButtons

def prDataSingGauss(samples,z,mean,cov):
    x = samples[z]
    if (not np.isfinite(mean).all()) or (not np.isfinite(cov).all()):
        print('mean,conv ', mean,cov)
    p = multivariate_normal.pdf(x, mean,cov)
    return p

def getWik(ps,alpha):
    ws = [ps[k,:]*alpha[k,:]/np.sum(ps*alpha,axis=0) for k in range(len(ps))]
    ws = np.array(ws)
    return ws

def runCondEM(samples,z0,means,Cs,m,show=True):
    means = np.array(means)
    n = samples.shape[0]
    dim = samples.shape[1]
    if not dim == 2:
        print('------------------------ERROR-----------------------')
    Mt = {}
    Ct = {}
    for k in range(m):
        for d in range(dim):
            print('k,d', k, d)
            Mt[k, d] = []
            Ct[k, d] = []
    alpha = (1.0 / float(m)) * np.ones((m, n))
    for t in range(1000):
        #################
        # estimate
        ##################
        ps = np.array([prDataSingGauss(samples, z0, means[k], Cs[k]) for k in range(m)])
        if not np.isfinite(ps).all(): print('not finite ps el. ',ps[np.isfinite(ps)==False],ps)
        if np.sum(ps==0):
            ps[ps==0] = np.finfo(ps.dtype).tiny
        ws = getWik(ps, alpha)
        #print ws
        #################
        # maximization
        ##################
        nk = np.sum(ws, axis=1)
        # print nk
        alpha = nk / n
        # print alpha
        alpha = np.array([alpha] * n).T
        if not np.isfinite(samples).all(): print('not finite samples el. ',samples[np.isfinite(samples)==False])
        if not np.isfinite(ws).all(): print('not finite ws el. ',ws[np.isfinite(ws)==False],ws)
        if not np.isfinite(nk).all(): print('not finite nk el. ',nk[np.isfinite(nk)==False],nk)
        ymeans = np.sum(ws*samples[:,1],axis=1)/nk
        #print ymeans
        #print means
        means[:,1] = ymeans.T
        for k in range(m):
            for i in range(dim):
                Mt[k,i].append(means[k,i])
        oldCs = Cs
        Cs = []
        for k in range(m):
            C = []
            for i in range(dim):
                Crow = [np.sum(ws[k] * (samples[:, i] - means[k, i]) * (samples[:, j] - means[k, j])) / nk[k] for j in
                        range(dim)]
                C.append(Crow)
            C = np.array(C)
            v = np.linalg.eigvals(C)
            if oldCs[k][0,0] < np.max(v):
                C[0,0]  = oldCs[k][0,0]
                if np.linalg.det(C) < 10 ** -3:
                    C = oldCs[k]
            Cs.append(C)
            # print 'C',C
            for i in range(dim):
                    Ct[k, i].append(C[i, i])
        for k in range(m):
            if np.linalg.det(Cs[k]) < 10 ** -8:
                print('error !!!!!!!!!')
                return nk, means, Cs, ps, ws, alpha
    if show:
        plt.figure()
        plt.title('Mean Evolution')
        for k in range(m):
            for d in range(dim):
                plt.plot(Mt[k, d], label='dim' + str(d) + 'k' + str(k))
        plt.legend()
        plt.figure()
        plt.title('Var Evolution')
        for k in range(m):
            for d in range(dim):
                plt.plot(Ct[k, d], label='dim' + str(d) + 'k' + str(k))
        plt.legend()
    return nk, means, Cs, ps, ws, alpha

def runConEM(samples,z0,means,Cs,m,const='mean-shift',show=True):
    n = samples.shape[0]
    if samples.ndim==2:
        dim = samples.shape[1]
    else:
        dim = 1
    Mt = {}
    Ct = {}
    for k in range(m):
        for d in range(dim):
            print('k,d',k,d)
            Mt[k,d]=[]
            Ct[k,d]=[]
    alpha = (1.0/float(m))*np.ones((m,n))
    for t in range(1000):#(1000):
        #estimation
        #print '------ t =',t
        ps = np.array([prDataSingGauss(samples,z0,means[k],Cs[k]) for k in range(m)])
        if not np.isfinite(ps).all(): print('not finite ps el. ',ps[np.isfinite(ps)==False],ps)
        if np.sum(ps==0):
            #print 'null ps el. ',ps[ps==0],ps,ps.dtype
            ps[ps==0] = np.finfo(ps.dtype).tiny
        #print 'ps.shape',ps.shape
        #print 'alpha.shape',alpha.shape
        ws = getWik(ps,alpha)
        #print ws
        #print 'ws.shape',ws.shape
        
        #maximization
        nk = np.sum(ws,axis=1)
        #print nk
        alpha = nk/n
        #print alpha
        alpha = np.array([alpha]*n).T

        if dim > 1:
            if not np.isfinite(samples).all(): print('not finite samples el. ',samples[np.isfinite(samples)==False])
            if not np.isfinite(ws).all(): print('not finite ws el. ',ws[np.isfinite(ws)==False],ws)
            if not np.isfinite(nk).all(): print('not finite nk el. ',nk[np.isfinite(nk)==False],nk)
            means = [np.sum(ws*samples[:,d],axis=1)/nk for d in range(dim)]
            means = np.array(means).T
            if const == 'mean-shift':
                mean0 = np.mean(means+np.array([np.arange(m)]*dim).T,axis=0)#np.mean(means*2**np.array([np.arange(1,m+1),np.arange(1,m+1)]).T,axis=0)
                means = np.array([mean0-i for i in np.arange(m)]) #np.array([mean0/2**i for i in range(1,m+1)])
            for k in range(m):
                for i in range(dim):
                    Mt[k,i].append(means[k,i]) 
        else:
            fact = 1.0
            means = np.sum(ws*samples,axis=1)/nk
            #mean0 = np.mean(means * np.arange(m))
            #means = np.array([mean0 * i for i in np.arange(m)])
            #print 'no fact',mean0,means
            if const == 'mean-shift':
                mean0 = np.mean(means+ fact * np.arange(m))
                means = np.array([mean0- fact * i for i in np.arange(m)])
            for k in range(m):
                Mt[k,0].append(means[k]) 
            #print 'fact',mean0,means        
        #print 'means',means
        Cs = []
        if dim > 1:
            for k in range(m):
                C = []
                for i in range(dim):
                    Crow = [np.sum(ws[k]*(samples[:,i]-means[k,i])*(samples[:,j]-means[k,j]))/nk[k] for j in range(dim)]
                    C.append(Crow)
                C=np.array(C)
                Cs.append(C)
                #print 'C',C
                for i in range(dim):
                    Ct[k,i].append(C[i,i])
        else:
            for k in range(m):
                #print 'shapes', samples.shape,ws[k].shape,means[k].shape,nk[k].shape
                #print 'nk',nk[k]
                #print 'meank',means[k]
                C = np.sum(ws[k]*(samples-means[k])*(samples-means[k]))/nk[k]
                Cs.append(C)
                #print 'C',C
                Ct[k,0].append(C)
        #print Cs
        #plotStuff(dim,m,nk,samples,ps,ws,alpha,means,Cs)
        if dim > 1:                
            for k in range(m):
                if np.linalg.det(Cs[k]) < 10**-8:
                    print('error !!!!!!!!!')
                    return nk,means,Cs,ps,ws,alpha
    if show:
        plt.figure()
        plt.title('Mean Evolution')
        for k in range(m):
            for d in range(dim):
                plt.plot(Mt[k,d],label='dim'+str(d)+'k'+str(k))
        plt.legend()
        plt.figure()
        plt.title('Var Evolution')
        for k in range(m):
            for d in range(dim):
                plt.plot(Ct[k,d],label='dim'+str(d)+'k'+str(k))
        plt.legend()
    return nk,means,Cs,ps,ws,alpha

    
def drawCovEllipse(mean,C,ax,num):
    lam, v = np.linalg.eig(C)
    lam = np.sqrt(lam)
    c = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,0)]
    #print 'lamda',lam
    #print 'vec',v
    for j in range(1, 3):
        if lam[1]>lam[0]:
            ell = Ellipse(xy=(mean[0], mean[1]), 
                          width=lam[0]*j*2, height=lam[1]*j*2,angle=np.rad2deg(np.arctan(v[1, 0]/v[1, 1])),
                           linewidth=2, fill=False,edgecolor=c[num],label=str(num),alpha=0.8)
        else:
            ell = Ellipse(xy=(mean[0], mean[1]), 
                      width=lam[1]*j*2, height=lam[0]*j*2,angle=np.rad2deg(np.arctan(v[0, 0]/v[0, 1])),
                       linewidth=2, fill=False,edgecolor=c[num],label=str(num),alpha=0.8)   
        ax.add_artist(ell)
    #ax.legend()
    return lam, v

def plotStuff(dim,m,nk,samples,ps,ws,alpha,means,Cs,time,sufx,xlabel='V450-A',ylabel='PE-A',zlabel='FITC-A',outf=False,path='plots/'):
    print('outf',outf)
    plt.figure()
    plt.bar(range(m),nk)

    if dim == 2:
        for k in range(m): 
            fig = pyplot.figure()
            ax = Axes3D(fig)        
            ax.scatter(samples[:,0],samples[:,1],ws[k,:],alpha=0.6)
    if dim == 1: 
        plt.figure()
        plt.hist(samples,bins=100,alpha=0.4,color='y')
        for k in range(m):
            x = np.linspace(samples.min(),samples.max(),100)
            dx = .5*(x[1]-x[0])
            plt.scatter(samples,ws[k,:]*dx*nk[k])
	
    if dim == 2:
        for k in range(m):
            fig = pyplot.figure()
            ax = Axes3D(fig)
            ax.scatter(samples[:,0],samples[:,1],ps[k,:],alpha=0.6)
    if dim > 1:
        for d in range(dim):
            plt.figure()
            plt.hist(samples[:,d],bins=100,alpha=0.4,color='y')
            x = np.linspace(samples[:,d].min(),samples[:,d].max())
            dx = .5*(x[1]-x[0])
            mix = np.zeros_like(x)
            for k in range(m):
                plt.plot(x, multivariate_normal.pdf(x, means[k,d],Cs[k][d,d])*dx*nk[k])
                mix = mix + multivariate_normal.pdf(x, means[k,d],Cs[k][d,d])*dx*nk[k]
            plt.plot(x, mix)
    if dim == 1: 
        plt.figure()
        plt.hist(samples,bins=100,alpha=0.4,color='y')
        x = np.linspace(samples.min(),samples.max())
        dx = .5*(x[1]-x[0])
        mix = np.zeros_like(x)
        for k in range(m): 
            #plt.scatter(samples,ps[k,:]*dxp*nk[k])
            plt.plot(x, multivariate_normal.pdf(x, means[k],Cs[k])*dx*nk[k])
            mix = mix + multivariate_normal.pdf(x, means[k],Cs[k])*dx*nk[k]
        plt.plot(x, mix)


    if dim == 2:
        #plt.figure()
        #ax = plt.subplot(111, aspect='equal')
        #ax.scatter(samples[:,0],samples[:,1],c=np.argmax(ws.T,axis=1),alpha=0.6)
        nbins = 100.0
        llim = np.min(samples) - 2
        lim = 18
        axScatterxy,axHistx,axHisty = plot2dloglog2(samples[:,0],samples[:,1],time,llim=llim,lim = lim,xlabel=xlabel,ylabel=ylabel,nbins = nbins) 
        x = np.linspace(llim,lim,nbins)       
        mixx = np.zeros_like(x)
        mixy = np.zeros_like(x)        
        for k in range(len(ps)):
            print('k',k)
            print('nk',nk[k])
            print('mean',means[k])
            print('Sigma',Cs[k])
            drawCovEllipse(means[k],Cs[k],axScatterxy,k)
            dx = (x[1]-x[0])
            axHistx.plot(x, dx*nk[k]*multivariate_normal.pdf(x, means[k,0],Cs[k][0,0])) 
            axHisty.plot(dx*nk[k]*multivariate_normal.pdf(x, means[k,1],Cs[k][1,1]),x) 
            mixx = mixx + multivariate_normal.pdf(x, means[k,0],Cs[k][0,0])*dx*nk[k] 
            mixy = mixy + multivariate_normal.pdf(x, means[k,1],Cs[k][1,1])*dx*nk[k]
        axHistx.plot(x, mixx)
        axHisty.plot(mixy,x)
        axScatterxy.scatter(np.array(means)[:,0],np.array(means)[:,1],c='black')
    if dim == 3:
        nbins = 100.0
        llim = np.min(samples) - 2
        lim = 18
        #print 'sampeles',samples
        axScatterxy,axScatterxz,axScatteryz,axHistx,axHisty,axHistz = plot3dloglog(samples[:,0],samples[:,1],samples[:,2],time,llim,lim,xlabel,ylabel,zlabel,nbins)
        for k in range(len(ps)):
            drawCovEllipse(means[k][[0,1]],Cs[k][np.ix_([0,1],[0,1])],axScatterxy,k)
            drawCovEllipse(means[k][[0,2]],Cs[k][np.ix_([0,2],[0,2])],axScatterxz,k)
            drawCovEllipse(means[k][[1,2]],Cs[k][np.ix_([1,2],[1,2])],axScatteryz,k)
            for d,ax in enumerate([axHistx,axHisty,axHistz]):
                x = np.linspace(llim,lim,nbins)
                dx = (x[1]-x[0])
                ax.plot(x, dx*nk[k]*multivariate_normal.pdf(x, means[k,d],Cs[k][d,d]))
        axScatterxy.scatter(np.array(means)[:,0],np.array(means)[:,1],c='black')
        axScatterxz.scatter(np.array(means)[:,0],np.array(means)[:,2],c='black')
        axScatteryz.scatter(np.array(means)[:,1],np.array(means)[:,2],c='black')
    print('outf', outf)
    if outf: plt.savefig(path+str(dim)+'dmixture'+str(time)+'h'+sufx+'.png')
        #plt.figure()
        #ax = plt.subplot(111, aspect='equal')
        #ax.scatter(samples[z0,0],samples[z0,1],c=np.argmax(ws.T,axis=1),alpha=0.6)
        #for k in range(len(ps)):
        #    drawCovEllipse(means[k][[0,1]],np.array([Cs[k][[0,1],0],Cs[k][[0,1],1]]),ax)
        #ax.scatter(np.array(means)[:,0],np.array(means)[:,1],c='black')
        #plt.figure()
        #ax = plt.subplot(111, aspect='equal')
        #ax.scatter(samples[z0,0],samples[z0,2],c=np.argmax(ws.T,axis=1),alpha=0.6)
        #for k in range(len(ps)):
        #    drawCovEllipse(means[k][[0,2]],np.array([Cs[k][[0,2],0],Cs[k][[0,2],2]]),ax)
        #ax.scatter(np.array(means)[:,0],np.array(means)[:,2],c='black')
        #plt.figure()
        #ax = plt.subplot(111, aspect='equal')
        #ax.scatter(samples[z0,1],samples[z0,2],c=np.argmax(ws.T,axis=1),alpha=0.6)
        #for k in range(len(ps)):
        #    drawCovEllipse(means[k][[1,2]],np.array([Cs[k][[1,2],1],Cs[k][[1,2],2]]),ax)
        #ax.scatter(np.array(means)[:,1],np.array(means)[:,2],c='black')

def plot2dloglog(x,y):
    # Plot data
    fig1,((ax01, ax02),(ax1, ax2)) = plt.subplots(2,2)
    ax01.hist(x,bins=100)
    ax02.hist(y,bins=100)
    ax1.plot(x,y,'.r')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim((2,12))
    ax1.set_ylim((2,12))
    #plt.loglog()

    # Estimate the 2D histogram
    nbins = 100
    H, xedges, yedges = np.histogram2d(x,y,bins=nbins)

    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)

    # Mask zeros
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero

    # Plot 2D histogram using pcolor
    ax2.pcolormesh(xedges,yedges,Hmasked)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlim((2,12))
    ax2.set_ylim((2,12))
    #cbar = ax2.colorbar()
    #cbar.ax2.set_ylabel('Counts')


def initAx2dloglog(fig):
    # Plot data
    nullfmt = NullFormatter()         # no labels
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    #plt.loglog()
    return axScatter,axHistx,axHisty

def plotScatter2dloglog(axScatter,x,y,llim,lim,maskVal=0, nbins = 100.0):
    # Estimate the 2D histogram
    H, xedges, yedges = np.histogram2d(x,y,bins=int(nbins))
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
    # Mask zeros
    Hmasked = np.ma.masked_where(H<=maskVal,H) # Mask pixels with a value of zero
    # Plot 2D histogram using pcolor
    axScatter.pcolormesh(xedges,yedges,Hmasked)
    axScatter.set_xlim((llim, lim))
    axScatter.set_ylim((llim, lim))
    axScatter.plot([llim, lim],[llim, lim])
    return H    

def plot2dloglog2(x,y,time,llim=8,lim = 18,xlabel='x',ylabel='y',nbins = 100.0):
    # Plot data
    fig = plt.figure(figsize=(8, 8))
    axScatter,axHistx,axHisty = initAx2dloglog(fig)
    axHistx.set_title(str(time)+' h '+ str(len(x)) +' events')
    #xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    plotScatter2dloglog(axScatter,x,y,llim,lim,nbins=nbins)
    axScatter.set_xlabel('x')
    axScatter.set_ylabel('y')
    binwidth = (lim-llim)/nbins
    bins = np.arange(llim, lim + binwidth, binwidth)
    #print bins
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    axHistx.set_ylabel(xlabel)
    axScatter.set_ylabel(ylabel)
    axScatter.set_xlabel(xlabel)
    axHisty.set_xlabel(ylabel)
    return axScatter,axHistx,axHisty

def initAx3dloglog(fig):
    # Plot data
    nullfmt = NullFormatter()         # no labels
    left, width = 0.1, 0.26
    bottom, height = 0.1, 0.26
    rect_scatter = [left, bottom+height+ 0.03, width, height]
    rect_scatter2 = [left, bottom, width, height]
    rect_scatter3 = [left+width+0.03, bottom, width, height]
    rect_histx = [left, bottom+ 2*(height+ 0.03), width, height]
    rect_histy = [left+width+0.03, bottom + height+ 0.03, width, height]
    rect_histz = [left+ 2*(width+0.03), bottom, width, height]
    axScatterxy = plt.axes(rect_scatter)
    axScatterxz = plt.axes(rect_scatter2)
    axScatteryz = plt.axes(rect_scatter3)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    axHistz = plt.axes(rect_histz)    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.xaxis.set_major_formatter(nullfmt)
    axHistz.xaxis.set_major_formatter(nullfmt)
    #plt.loglog()
    return axScatterxy,axScatterxz,axScatteryz,axHistx,axHisty,axHistz

def getKernel(ssca,fsca):
    xmin = ssca.min()
    xmax = ssca.max()
    ymin = fsca.min()
    ymax = fsca.max()
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([ssca, fsca])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    return xx, yy,f,kernel

def pltSSCAFSCA(ssca,fsca,gate1,gate2):
    plt.figure()
    #for ['none','gate fsca','gate ssca','gate ssca and fsca']:
    plt.scatter(ssca[(~gate1)&(~gate2)],fsca[(~gate1)&(~gate2)],0.01,label='none')
    plt.scatter(ssca[(gate1)&(~gate2)],fsca[(gate1)&(~gate2)],0.01,label='gate 1')
    plt.scatter(ssca[(~gate1)&(gate2)],fsca[(~gate1)&(gate2)],0.01,label='gate 2')
    plt.scatter(ssca[(gate1)&(gate2)],fsca[(gate1)&(gate2)],0.01,label='gate 1 and 2')
    plt.xlabel('ssca')
    plt.ylabel('fsca')
    plt.legend()

    xx, yy,f,kernel = getKernel(ssca,fsca)
    xy = np.vstack([x,y])
    valK = kernel(xy)
    plt.scatter(x,y,0.01,c = valK)
    #cset = plt.contour(xx, yy, f, colors='k')
    #plt.clabel(cset, inline=1, fontsize=10)
    plt.xlim((0,270000))
    plt.ylim((0,270000))
    return xx, yy,f,kernel


def regenax(x,y,valK,means,sig,ax):
    ax.clear()            
    ax.scatter(x,y,1,c = valK)
    if type(sig) == float:
        C = np.diagflat([sig,sig])
        for i,me in enumerate(means):
            print(me)
            lam,v = drawCovEllipse(me,C,ax,i) 
    else:
        for i,me,s in zip(range(len(means)),means,sig):
            print('mean',me,'Cs',s)
            #C = np.diagflat(s)
            lam,v = drawCovEllipse(me,s,ax,i) 

def plot3dloglog(x,y,z,time,llim=8,lim = 18,xlabel='x',ylabel='y',zlabel='z',nbins = 100.0,Hthr=0.0,dfAuto=None):
    fig = plt.figure(figsize=(12, 12))
    axScatterxy,axScatterxz,axScatteryz,axHistx,axHisty,axHistz = initAx3dloglog(fig)
    axHistx.set_title(str(time)+' h '+str(len(x)) +' events')
    #xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    #(int(xymax/binwidth) + 1) * binwidth
    #print llim,lim
    plotScatter2dloglog(axScatterxy,x,y,llim,lim,maskVal=Hthr,nbins=nbins)
    plotScatter2dloglog(axScatterxz,x,z,llim,lim,maskVal=Hthr,nbins=nbins)
    plotScatter2dloglog(axScatteryz,y,z,llim,lim,maskVal=Hthr,nbins=nbins)
    binwidth = (lim-llim)/nbins
    bins = np.arange(llim, lim + binwidth, binwidth)
    #print bins
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins)
    axHistz.hist(z, bins=bins)
    axHistx.set_xlim(axScatterxy.get_xlim())
    axHisty.set_xlim(axScatterxy.get_ylim())
    axHistz.set_xlim(axScatteryz.get_ylim())
    axHistx.set_ylabel(xlabel)
    axScatterxy.set_ylabel(ylabel)
    axScatterxz.set_ylabel(zlabel)
    axScatterxz.set_xlabel(xlabel)
    axScatteryz.set_xlabel(ylabel)
    axHistz.set_xlabel(zlabel)
    if isinstance(dfAuto, pd.DataFrame):
        axHistx.hist(dfAuto[xlabel],range=(llim,lim),bins=bins,histtype='step')
        axHisty.hist(dfAuto[ylabel],range=(llim,lim),bins=bins,histtype='step')
        axHistz.hist(dfAuto[zlabel],range=(llim,lim),bins=bins,histtype='step')
    return axScatterxy,axScatterxz,axScatteryz,axHistx,axHisty,axHistz

def plotGaussianComp(k,samples,ws,means,Cs,xlabel='x',ylabel='y',zlabel='z',llim = 0,lim=15):
    if samples.shape[1] == 2:
        plt.figure()
        ax = plt.subplot(121, aspect='equal')
        sample3 = samples[np.argmax(ws.T,axis=1)==k,:]
        ax.scatter(sample3[:,0],sample3[:,1],alpha=0.6)
        ax =  plt.subplot(122, aspect='equal')
        nbins = [32,32]
        H, xedges, yedges = np.histogram2d(samples[:,0],samples[:,1],bins=nbins,weights=ws[k,:])
        # H needs to be rotated and flipped
        H = np.rot90(H)
        H = np.flipud(H)
        # Mask zeros
        Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
        # Plot 2D histogram using pcolor
        ax.pcolormesh(xedges,yedges,Hmasked)
        #ax.scatter(samples[:,0],samples[:,1],c=cms.jet(ws[k,:]),alpha=0.2)
        drawCovEllipse(means[k],Cs[k],ax)#means[k],Cs[k],ax)
    if samples.shape[1] == 3:
        fig = pyplot.figure()
        ax = Axes3D(fig)  
        samples3 = samples[np.argmax(ws.T,axis=1)==k,:]
        ax.scatter(samples3[:,0],samples3[:,1],samples3[:,2])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        axScatterxy,axScatterxz,axScatteryz,axHistx,axHisty,axHistz = plot3dloglog(samples3[:,0],samples3[:,1],samples3[:,2],0,llim,lim,xlabel,ylabel,zlabel)
        drawCovEllipse(means[k][[0,1]],Cs[k][np.ix_([0,1],[0,1])],axScatterxy)
        drawCovEllipse(means[k][[0,2]],Cs[k][np.ix_([0,2],[0,2])],axScatterxz)
        drawCovEllipse(means[k][[1,2]],Cs[k][np.ix_([1,2],[1,2])],axScatteryz)

def binObservations(k,samples,ws,nbins = 16):
    # Estimate the 2D histogram
    #H, xedges, yedges = np.histogram2d(samples[:,0],samples[:,1],bins=nbins,weights=ws[k,:])
    H, edges = np.histogramdd(samples,bins=nbins,weights=ws[k,:])      
    # H needs to be rotated and flipped
    if samples.shape[1] == 2:
        H = np.rot90(H)
        H = np.flipud(H)
        xedges,yedges = edges
        dxp = (xedges[1]-xedges[0])#/2.0
        dyp = (yedges[1]-yedges[0])#/2.0
        xx,yy=np.meshgrid(0.5*(xedges[:-1]+xedges[1:]),0.5*(yedges[:-1]+yedges[1:]))
        pos = np.empty(xx.shape + (2,))
        pos[:, :, 0] = xx
        pos[:, :, 1] = yy
        binWidth = [dxp,dyp]
    if samples.shape[1] == 3:  
        H = np.rot90(H)
        H = np.flipud(H)
        xedges,yedges,zedges = edges
        dxp = (xedges[1]-xedges[0])#/2.0
        dyp = (yedges[1]-yedges[0])#/2.0
        dzp = (zedges[1]-zedges[0])#/2.0
        xx,yy,zz=np.meshgrid(0.5*(xedges[:-1]+xedges[1:]),0.5*(yedges[:-1]+yedges[1:]),0.5*(zedges[:-1]+zedges[1:]))
        pos = np.empty(xx.shape + (3,))
        pos[:, :, :, 0] = xx
        pos[:, :, :, 1] = yy
        pos[:, :, :, 2] = zz
        binWidth = [dxp,dyp,dzp]
    return H,binWidth,pos

def msDist(samples,means,Cs,k):
    x = (samples - means[k])
    CI = np.linalg.inv(Cs[k])
    interm = np.dot(x,CI)
    rr2 = np.sqrt(np.sum(interm*x,axis=1))
    return rr2

def binObservationsRad(means,Cs,k,samples,ws,nbins = 16):
    # Estimate the 1D histogram
    #H, xedges, yedges = np.histogram2d(samples[:,0],samples[:,1],bins=nbins,weights=ws[k,:])
    rr = msDist(samples,means,Cs,k)
    #rr = np.sqrt(np.sum((samples-means)**2,axis=1))
    H, edges = np.histogram(rr,bins=nbins,weights=ws[k,:])      
    dxp = (edges[1]-edges[0])#/2.0
    pos=0.5*(edges[:-1]+edges[1:])
    return H,dxp,pos,edges
    

def binEstimates(k,nk,pos,binArea,means,Cs):
    #print 'Cs[k]',Cs[k]
    p = multivariate_normal.pdf(pos, means[k],Cs[k])#means[k],Cs[k])
    #ax.scatter(xpos+dx, ypos+dy, dz)   
    est = binArea*nk[k]*p 
    return est

def binEstimates2(k,nk,pos,binWidths,binArea,means,Cs):
    #print 'Cs[k]',Cs[k]
    us = np.random.uniform(-0.5,0.5,(200,) + pos.shape) 
    randPos = np.array([ pos + np.array(binWidths * u) for u in us])
    ps = multivariate_normal.pdf(randPos, means[k],Cs[k])
    p = np.mean(ps,axis=0)
    #ax.scatter(xpos+dx, ypos+dy, dz)   
    est = binArea*nk[k]*p
    return est

def binEstimatesRad(k,nk,edges,means,Cs):
    p = chi.cdf(edges[1:],len(means[k]),0,1) - chi.cdf(edges[:-1],len(means[k]),0,1)
    return nk[k]*p

def plotEstObsComp(k,pos,H,est,means,Cs):
    if H.ndim == 1:   
        xx = pos
        fig = plt.figure()
        plt.plot(xx,H,'+',label = 'H')
        #bar3d(xpos, ypos, zpos, dx, dy, dz, color='r', zsort='average',alpha=0.2)
        #ax.plot_surface(xx,yy,H,alpha=0.8)
        plt.plot(xx,est,'.',label = 'est',alpha=0.4)
        plt.xlabel('x')
        plt.legend() 
        plt.figure()
        plt.scatter(xx[H>5], (est[H>5] - H[H>5])**2/(est[H>5]) )
    if H.ndim == 2:   
        xx = pos[:, :, 0]
        yy = pos[:, :, 1]
        fig = pyplot.figure()
        ax = Axes3D(fig)
        ax.scatter(xx,yy, H)
        #bar3d(xpos, ypos, zpos, dx, dy, dz, color='r', zsort='average',alpha=0.2)
        #ax.plot_surface(xx,yy,H,alpha=0.8)
        ax.scatter(xx,yy,est,alpha=0.4)
        ax.set_xlabel('x')
        ax.set_ylabel('y') 
        plt.figure()
        deltax=xx-means[k][0]
        deltay=yy-means[k][1]
        plt.scatter(np.sqrt(deltax**2+deltay**2), est,label = 'est')
        plt.scatter(np.sqrt(deltax**2+deltay**2), H,label = 'H')
        plt.legend()
        ps = np.array([pos[:, :, 0].flatten(),pos[:, :, 1].flatten()]).T
        rr = msDist(ps,means,Cs,k)
        plt.scatter(rr, est.flatten(),label = 'est')
        plt.scatter(rr, H.flatten(),label = 'H')
        plt.figure()
        plt.figure()
        plt.scatter(np.sqrt(deltax[H>5]**2+deltay[H>5]**2), (est[H>5] - H[H>5])**2/(est[H>5]) )
    if H.ndim == 3:
        fig = plt.figure(figsize=(10,5))
        Hprojz = np.sum(H,axis=2)
        estprojz = np.sum(est,axis=2)
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(pos[:, :, 0, 0],pos[:, :, 0, 1], Hprojz)
        #bar3d(xpos, ypos, zpos, dx, dy, dz, color='r', zsort='average',alpha=0.2)
        #ax.plot_surface(xx,yy,H,alpha=0.8)
        ax.scatter(pos[:, :, 0, 0],pos[:, :, 0, 1],estprojz,alpha=0.4)
        ax.set_xlabel('V450-A')
        ax.set_ylabel('PE-A') 
        ax.set_ylim([6,18])
        ax.set_xlim([6,18])
        Hprojy = np.sum(H,axis=1)
        estprojy = np.sum(est,axis=1)
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        ax1.scatter(pos[:, :, 0, 0],pos[:, 0,  :, 2].T,Hprojy)
        #bar3d(xpos, ypos, zpos, dx, dy, dz, color='r', zsort='average',alpha=0.2)
        #ax.plot_surface(xx,yy,H,alpha=0.8)
        ax1.scatter(pos[:, :, 0, 0],pos[:, 0,  :, 2].T,estprojy)
        ax1.set_xlabel('V450-A')
        ax1.set_ylabel('FITC-A') 
        ax1.set_ylim([6,18])
        ax1.set_xlim([6,18])
        plt.figure()
        deltax=pos[:, :, :, 0]-means[k][0]
        deltay=pos[:, :, :, 1]-means[k][1]
        deltaz=pos[:, :, :, 2]-means[k][2]
        plt.scatter(np.sqrt(deltax**2+deltay**2+deltaz**2), est,label = 'est')
        plt.scatter(np.sqrt(deltax**2+deltay**2+deltaz**2), H,label = 'H')
        plt.legend()
        plt.figure()
        ps = np.array([pos[:, :, :, 0].flatten(),pos[:, :, :, 1].flatten(),pos[:, :, :, 2].flatten()]).T
        rr = msDist(ps,means,Cs,k)
        plt.scatter(rr, est.flatten(),label = 'est')
        plt.scatter(rr, H.flatten(),label = 'H')
        plt.legend()
        plt.figure()
        plt.scatter(np.sqrt(deltax[H>5]**2+deltay[H>5]**2+deltaz[H>5]**2), (est[H>5] - H[H>5])**2/(est[H>5]) )

def driftMean(rs,k,nk,pos,binWidths,binArea,H,means,Cs,dim):
    drfChi = []
    for r in rs:
        if means.shape[1]==2:
            if dim == 'x': drift = [r,0.]
            else: drift = [r,0.]
        if means.shape[1]==3:
            if dim == 'x': drift = [r,0.,0.]
            if dim == 'y': drift = [r,0.,0.]
            else: drift = [r,0.,0.]
        means2 = means.copy()
        means2[k] = means2[k] + drift
        #print means,means2,drift
        est = binEstimates2(k,nk,pos,binWidths,binArea,means2,Cs)
        #mgm.plotEstObsComp(k,xx,yy,H,est,means2)
        drfChi.append(np.mean( (est[H>5] - H[H>5])**2/(est[H>5]) ))
    mind = rs[np.argmin(drfChi)]
    return mind,drfChi

def driftMeanRad(rs,k,nk,samples,ws,nBins,H,means,Cs,dim):
    drfChi = []
    for r in rs:
        if means.shape[1]==2:
            if dim == 'x': drift = [r,0.]
            else: drift = [r,0.]
        if means.shape[1]==3:
            if dim == 'x': drift = [r,0.,0.]
            if dim == 'y': drift = [r,0.,0.]
            else: drift = [r,0.,0.]
        means2 = means.copy()
        means2[k] = means2[k] + drift
        #print means,means2,drift
        H,dxp,pos,edges = binObservationsRad(means2,Cs,k,samples,ws,nbins = nBins)
        est = binEstimatesRad(k,nk,edges,means2,Cs)
        #plotEstObsComp(k,pos,H,est,means)
        drfChi.append(np.mean( (est[H>5] - H[H>5])**2/(est[H>5]) ))
        #print np.mean( (est[H>5] - H[H>5])**2/(est[H>5]) )
    mind = rs[np.argmin(drfChi)]
    return mind,drfChi

def driftVar(rs,k,nk,pos,binWidths,binArea,H,means,Cs,dim):
    drfChi = []
    for r in rs:
        if means.shape[1]==2:
            if dim == 'xx': drift = np.array([[r,0.],[0.,0.]]) 
            elif dim == 'yy': drift = np.array([[0.,0.],[0.,r]]) 
            else: drift = np.array([[0.,r],[r,0.]]) 
        if means.shape[1]==3:
            if dim == 'xx': drift = np.array([[r,0.,0.],[0.,0.,0.],[0.,0.,0.]]) 
            elif dim == 'yy': drift = np.array([[0.,0.,0.],[0.,r,0.],[0.,0.,0.]]) 
            else: drift = np.array([[0.,r,0.],[r,0.,0.],[0.,0.,0.]]) 
        Cs2 = list(Cs)
        Cs2[k] = Cs2[k] + drift
        est = binEstimates2(k,nk,pos,binWidths,binArea,means,Cs2)
        #mgm.plotEstObsComp(k,xx,yy,H,est,means2)
        drfChi.append(np.mean( (est[H>5] - H[H>5])**2/(est[H>5]) ))
    mind = rs[np.argmin(drfChi)]
    return mind,drfChi

def pairgrid_heatmap(x, y, **kws):
    #print kws
    color = kws.pop("color")
    cmap = sns.light_palette(color, as_cmap=True)
    xmin,xmax = np.min(x),np.max(x)
    ymin,ymax = np.min(y),np.max(y)
    #print ymin,ymax
    plt.hist2d(x, y, cmap = cmap, range=[[0,xmax],[0,ymax]], cmin=0.1, bins=10,alpha=0.8)

    
def rewriteGates(gate1,gate2,gate3):
    gateN = ['None']*len(gate1)
    for i,g in enumerate(gate1):
        if g == True:
            if not gateN[i] == 'None':
                gateN[i] = gateN[i] + ' and gate 1'
            else: gateN[i] = 'gate 1'
             #gateN[i] = gateN[i] + ' gate'
    for i,g in enumerate(gate2):
        if g == True:
            if not gateN[i] == 'None':
                gateN[i] = gateN[i] + ' and gate 2'
            else: gateN[i] = 'gate 2'
            #gateN[i] = gateN[i] + 'ssca gate'
    for i,g in enumerate(gate3):
        if g == True:
            if not gateN[i] == 'None':
                gateN[i] = gateN[i] + ' and gate 3'
            else: gateN[i] = 'gate 3'
    return gateN

def prepData(x,y,z,ssca,fsca,gate1,gate2,gate3,name):
    #boolVal = (x>2**4)*(y>2**4)*(z>2**4)#*(np.isfinite(x))*(np.isfinite(y))*(np.isfinite(z))
    #x=x[boolVal]
    #y=y[boolVal]
    #z=z[boolVal]
    #ssca = ssca[boolVal]
    #fsca = fsca[boolVal]
    #gate1 = (fsca<10000)|(fsca>260000)
    #gate2 = (ssca<10000)|(ssca>250000)
    #gate3 = (x<2**8)|(y<2**8)|(z<2**8)
    #print len(gate1),len(gate2),len(gate3)
    gateN = rewriteGates(gate1,gate2,gate3)
    #print len(gateN),len(np.log2(x))
    df = pd.DataFrame({'log V450-A':np.log2(x),'log PE-A':np.log2(y),'log FITC-A':np.log2(z),'ssca':ssca,'fsca':fsca,'gate':gateN})
    #df = df[np.isfinite(df['log V450-A']) & np.isfinite(df['log PE-A']) & np.isfinite(df['log FITC-A'])]
    g = sns.PairGrid(df, vars= ['log V450-A','log PE-A','log FITC-A','ssca','fsca'] ,diag_sharey=False,size=1.5, hue="gate")
    #g.map_lower(pairgrid_heatmap)
    g.map_lower(plt.scatter, s = 2.0, alpha = 0.3)
    g.map_diag(plt.hist, bins=40)
    g = g.add_legend()
    g.axes[0,0].set_title(name)
    g.axes[0,1].set_ylim([0.1,18]) 
    g.axes[0,2].set_ylim([0.1,18])
    g.axes[1,0].set_ylim([0.1,18])
    g.axes[1,2].set_ylim([0.1,18]) 
    g.axes[2,0].set_ylim([0.1,18])
    g.axes[2,1].set_ylim([0.1,18]) 
    g.axes[3,0].set_xlim([0.1,18])
    g.axes[3,1].set_xlim([0.1,18])
    g.axes[3,2].set_xlim([0.1,18]) 
    g.axes[4,0].set_xlim([0.1,18])
    g.axes[4,1].set_xlim([0.1,18])
    g.axes[4,2].set_xlim([0.1,18])

    df = df[df['gate'] == 'None']
    return g,df

def noDropOutliers(df,lx,ly,lz):
    x=df[lx]
    y=df[ly]
    z=df[lz]
    return df,x,y,z

def noDropOutliersBgrCorr(df,dfAuto,lx,ly,lz):
    xauto,yauto,zauto = dfAuto[lx].mean(),dfAuto[ly].mean(),dfAuto[lz].mean()
    x=df[lx]
    y=df[ly]
    z=df[lz]
    cut = (x>xauto) & (y>yauto) & (z>zauto)
    x = x[cut]
    y = y[cut]
    z = z[cut]
    x = np.log2(2**x - 2**xauto)
    y = np.log2(2**y - 2**yauto)
    z = np.log2(2**z - 2**zauto)
    df = df[cut]
    df.loc[:,lx]=x
    df.loc[:,ly]=y
    df.loc[:,lz]=z
    return df,x,y,z

def dropOutliers(df,label,bins,th):
    x = df[label]
    H, edges = np.histogram(x,bins)   
    Hmask = H<=th
    for x0,x1 in zip(edges[:-1][Hmask],edges[1:][Hmask]):
        #print x0,x1
        df= df[(df[label]<x0)|(df[label]>=x1)]
    return df

def dropOutliers3d(df,lx,ly,lz,nBins = 100, nObs = 2):
    df = dropOutliers(df,lx,nBins,nObs)
    df = dropOutliers(df,ly,nBins,nObs)
    df = dropOutliers(df,lz,nBins,nObs)

    x=df[lx]
    y=df[ly]
    z=df[lz]
    return df,x,y,z

def dropOutliers3d2(df,lx,ly,lz,nBins = 100, nObs = 2):
    x=df[lx]
    y=df[ly]
    z=df[lz]
    samples = np.array([x,y,z]).T
    H, edges = np.histogramdd(samples,nBins)  
    H = np.rot90(H)
    H = np.flipud(H) 
    Hmask = H<=nObs
    df['mask']=0.0
    xx,yy,zz = np.meshgrid(edges[0][:-1],edges[1][:-1],edges[2][:-1])
    xxe,yye,zze = np.meshgrid(edges[0][1:],edges[1][1:],edges[2][1:])
    for x0,x1,y0,y1,z0,z1 in zip(xx[Hmask],xxe[Hmask],yy[Hmask],yye[Hmask],zz[Hmask],zze[Hmask]):
        df.loc[(df[lx]>=x0)&(df[lx]<=x1)&(df[ly]>=y0)&(df[ly]<=y1)&(df[lz]>=z0)&(df[lz]<=z1),'mask'] = 1.0
    df=df[df['mask'] == 0.0]
    x=df[lx]
    y=df[ly]
    z=df[lz]
    return df,x,y,z


def getMeanCsFromRes(results,m,dim,xlabel='V450-A',ylabel='PE-A',zlabel='FITC-A', errors=False):
    means = []
    Cs = []
    for k in range(m):
        if dim == 1:
            means.append(results.loc[0,'means '+str(k)])
            C = results.loc[0,'Cs '+xlabel+' '+str(k)]
        if dim > 1:
            means.append(results['means '+str(k)].values)
            C = np.zeros((dim,dim))
            C[0,:] = results['Cs '+xlabel+' '+str(k)].values
            C[1,:] = results['Cs '+ylabel+' '+str(k)].values
        if dim > 2:
            C[2,:] = results['Cs '+zlabel+' '+str(k)].values
        Cs.append(C)
    if errors:
        stdMeans = []
        stdCs = []
        for k in range(m):
            if dim == 1:
                stdMeans.append(results.loc[0, 'std means ' + str(k)])
                C = results.loc[0, 'std Cs ' + xlabel + ' ' + str(k)]
            if dim > 1:
                stdMeans.append(results['std means ' + str(k)].values)
                C = np.zeros((dim, dim))
                C[0, :] = results['std Cs ' + xlabel + ' ' + str(k)].values
                C[1, :] = results['std Cs ' + ylabel + ' ' + str(k)].values
            if dim > 2:
                C[2, :] = results['std Cs ' + zlabel + ' ' + str(k)].values
            stdCs.append(C)
        return means,Cs, stdMeans,stdCs
    return means,Cs

def runGM(hour,dim,m,data,res0,sufx,outf=True,show=True,cond=False,xlabel='V450-A',ylabel='PE-A',zlabel='FITC-A'):
    #print(data.columns)
    x = data['log '+xlabel]
    y = data['log '+ylabel]
    if not zlabel == None:
        z = data['log '+zlabel]

    if show:
        if not zlabel == None:
            plot3dloglog(x,y,z,hour,llim=2,xlabel=xlabel,ylabel=ylabel,zlabel=zlabel)
        else:
            plot2dloglog2(x, y, hour, llim=2, lim=18, xlabel=xlabel, ylabel=ylabel)


    if dim == 1: samples = x.values
    if dim == 2: samples = np.array([x,y]).T
    if dim == 3: samples = np.array([x,y,z]).T

    n = samples.shape[0]
    if samples.ndim >= 2:
        dim = samples.shape[1]
    else:
        dim = 1
    #print 'dim',dim

    if show:
        if dim == 3:
            fig = pyplot.figure()
            ax = Axes3D(fig)  
            ax.scatter(samples[:,0],samples[:,1],samples[:,2])
        if dim == 2:
            plt.figure()
            plt.scatter(samples[:,0],samples[:,1])
        if dim == 1:
            plt.figure()
            plt.hist(samples,bins=100)

    meanFRange = 10

    if len(res0) == 0 :
        if dim > 2:
            meanF = np.array([8.9,10.13, 10.40])#samples[range(0,meanFRange),:].mean(axis=0) #np.array([13,13,13])#samples[range(0,meanFRange),:].mean(axis=0)
            var = np.array([0.14,0.5,0.5])#samples[range(0,meanFRange),:].var(axis=0) #np.array([0.1,0.1,0.1])#samples[range(0,meanFRange),:].var(axis=0)
        elif dim == 2:
            meanF = np.array([8.9,10.13])#samples[range(0,meanFRange),:].mean(axis=0) #np.array([13,13,13])#samples[range(0,meanFRange),:].mean(axis=0)
            var = np.array([0.05,0.05])
        else:
            meanF = 8.9#samples[range(0,meanFRange)].mean(axis=0)
            var = 0.02#amples[range(0,meanFRange)].var(axis=0)
        means = np.array([meanF+i for i in np.arange(m-1,-1,-1)]) #np.array([mean0/2**i for i in range(1,m+1)])
        print('----------init',meanF,var,n)
        #print(range(m-1,-1,-1)
        if dim > 1:
            C = var*np.eye(dim)
        else:
            C=var
        Cs = [C for i in range(m)]
    else:
        means , Cs = getMeanCsFromRes(res0,m,dim,xlabel,ylabel,zlabel)
        
    print('init means',means)
    print('init Cs',Cs)

    z0 = range(0,n)#range(0,10) + range(n-10,n)
    if cond == False:
        nk,means,Cs,ps,ws,alpha = runConEM(samples,z0,means,Cs,m,const=None,show=show)
    else:
        print('-----run conditioned on X------------')
        nk, means, Cs, ps, ws, alpha = runCondEM(samples, z0, means, Cs, m, show=show)

    print('------result --------')
    print('means',means)
    print('Cs',Cs)

    if show:
        print('outf',outf,' figure labels ',xlabel,ylabel,zlabel)
        plotStuff(dim,m,nk,samples,ps,ws,alpha,means,Cs,hour,sufx,outf=outf,xlabel=xlabel,ylabel=ylabel,zlabel=zlabel)
    return means,Cs,ws



def getAq(datafile):
    aqName = [n for n in datafile.keys() if  isinstance(n, int)]
    aqStrName = [n for n in datafile.keys() if not isinstance(n, int)]
    aqName.sort()
    #print(aqName)
    names = aqStrName + aqName
    return names

def readPreProcPars(aqName):
    dataPars = {}
    fname = 'preproc-pars.csv'
    if os.path.isfile(fname):
        with open(fname) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    a = int(row['aq. name'])
                except:
                    a = row['aq. name']
                #print(a)
                d = {}
                for k,v in row.items():
                    try:
                        d[k] = int(v)
                    except:
                        try:
                            d[k] = float(v)
                        except:
                            d[k] = v
                dataPars[a] = d
    else:    
        for a in aqName:
            d = {'aq. name': a,'th':10**-11,'fsca lf':10000,'fsca hf':250000,'ssca lf':10000,'ssca hf':250000}
            dataPars[a] = d
    for a in aqName:
        try:
            print(dataPars[a])
        except:
            d = {'aq. name': a,'th':10**-11,'fsca lf':10000,'fsca hf':250000,'ssca lf':10000,'ssca hf':250000}
            dataPars[a] = d
            print('new ', dataPars[a])
    return dataPars

def doPreproc(datafile,sufx,aqName2,path):
    dataframe = {}
    aqName = getAq(datafile)
    dataPars = readPreProcPars(aqName)

    for k in aqName2:
        f = datafile[k]
        meta, data = fcsparser.parse(f, meta_data_only=False, reformat_meta=True)
        print('acquisition ',k)
        print(meta['EXPORT TIME'],meta['$ETIM'])
        x=data['V450-A']
        y=data['PE-A']
        z=data['FITC-A']
        gate0 = (x>0)&(y>0)&(z>0)
        data = data[gate0]
        x=data['V450-A']
        y=data['PE-A']
        z=data['FITC-A']
        ssca = data['SSC-A']
        fsca = data['FSC-A']
        vs,vf = dataPars[k]['ssca hf'],dataPars[k]['fsca hf']
        gate1 = (fsca<dataPars[k]['fsca lf'])|(fsca>vf)
        gate2 = (ssca<dataPars[k]['ssca lf'])|(ssca>vs)

        #xx, yy,f,kernel =mgm.pltSSCAFSCA(ssca,fsca,gate1,gate2)
        xx, yy,f,kernel =getKernel(ssca,fsca)
        th = dataPars[k]['th']
        #print th
        v = -np.log10(th)
        #print 'th',th,v

        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.15, bottom=0.35)
        xy = np.vstack([ssca,fsca])
        valK = kernel(xy)
        ax.scatter(ssca,fsca)
        ax.scatter(ssca[valK>th],fsca[valK>th],c=valK[valK>th])
        ax.scatter(ssca[gate1],fsca[gate1],c='red')
        ax.scatter(ssca[gate2],fsca[gate2],c='yellow')
        ax.set_title(str(k)+' h '+meta['EXPORT TIME'][:6]+' '+meta['$ETIM']+' th :'+str(th))
        dataPars[k]['date'] = meta['EXPORT TIME'][:6]
        dataPars[k]['time'] = meta['$ETIM']

        axcolor = 'lightgoldenrodyellow'
        axTh = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
        axThssca = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
        axThfsca = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)

        sth = Slider(axTh, 'Th neg exp:', 7.0, 15.0,v)
        sthssca = Slider(axThssca, 'Th ssca:', 7.0, 280000.0,dataPars[k]['ssca hf'])
        sthfsca = Slider(axThfsca, 'Th fsca:', 7.0, 280000.0,dataPars[k]['fsca hf'])

        def update(val):
            global th,vs,vf
            global gate1,gate2
            v = sth.val
            vs = sthssca.val
            vf = sthfsca.val
            th = (10**(-v))
            gate1 = (fsca<dataPars[k]['fsca lf'])|(fsca>vf)
            gate2 = (ssca<dataPars[k]['ssca lf'])|(ssca>vs)
            #print 'th',th
            #print v
            ax.clear()
            ax.scatter(ssca,fsca)
            ax.scatter(ssca[valK>th],fsca[valK>th],10.,c=valK[valK>th])
            ax.scatter(ssca[gate1],fsca[gate1],10.,c='red')
            ax.scatter(ssca[gate2],fsca[gate2],10.,c='yellow')
            ax.set_title(str(k)+' h '+meta['EXPORT TIME'][:6]+' '+meta['$ETIM']+' th :'+str(th))
            fig.canvas.draw_idle()
        sth.on_changed(update)
        sthssca.on_changed(update)
        sthfsca.on_changed(update)

        def updateTh(val):
            #print 'update th', th
            v = sth.val
            th = (10**(-v))
            gateTh = valK<th
            graph,dfTh = prepData(x,y,z,ssca,fsca,gateTh,gate1,gate2,str(k)+'h')
            print('drop rate ',float(len(dfTh))/float(len(data)),' original sample size ',len(data), ' th. sample size ',len(dfTh))
            dataPars[k]['preprocessing drop rate'] = float(len(dfTh))/float(len(data))
            dataPars[k]['th'] = th
            dataPars[k]['ssca hf'] = sthssca.val
            dataPars[k]['fsca hf'] = sthfsca.val
            plt.show()
            
        axUpdate = plt.axes([0.81, 0.20, 0.1, 0.075])
        bUp = Button(axUpdate, 'Update')
        bUp.on_clicked(updateTh)
        
        gateTh = valK<th
        graph,dfTh = prepData(x,y,z,ssca,fsca,gateTh,gate1,gate2,str(k)+' h')
        dataPars[k]['preprocessing drop rate'] = float(len(dfTh))/float(len(data))
        dataPars[k]['th'] = th
        dataPars[k]['ssca hf'] = sthssca.val
        dataPars[k]['fsca hf'] = sthfsca.val
        print('drop rate ',float(len(dfTh))/float(len(data)),' original sample size ',len(data), ' th. sample size ',len(dfTh))
            
        plt.show()
        
        fig.savefig(path+'SSCAFSCAPlot-'+str(k)+'h.pdf')
        graph.savefig(path+'Plot-'+str(k)+'h.pdf')
        dfTh.to_pickle(path+'cleaned'+str(k)+'h.pkl')

        dataframe[k] = dfTh
               


    for a in aqName:
        print(dataPars[a])

    with open('preproc-pars.csv', 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, dataPars[a].keys())
        w.writeheader()
        for a in aqName:
            w.writerow(dataPars[a])
    return dataframe


def writeGaussians(hour,dim,m,means,Cs,suffix,sufx,means_chunck=[],vars_chunck=[],xlabel='V450-A',ylabel='PE-A',zlabel='FITC-A'):
    results = pd.DataFrame()

    for k in range(m):
        if dim == 1:
            results.loc[0,'means '+str(k)] = means[k]
            results.loc[0,'Cs '+xlabel+' '+str(k)] =  Cs[k]
        if dim > 1:
            results['means '+str(k)] = means[k]
            results['Cs '+xlabel+' '+str(k)] =  Cs[k][0,:]
            results['Cs '+ylabel+' '+str(k)] =  Cs[k][1,:]
        if dim > 2: results['Cs '+zlabel+' '+str(k)] =  Cs[k][2,:]

    if not means_chunck==[]:
        for k in range(m):
            if dim == 1:
                results.loc[0,'std means '+str(k)] = np.std(means_chunck[k])
                results.loc[0,'std Cs '+xlabel+' '+str(k)] =  np.std(vars_chunck[k])
            if dim > 1:
                results['std means '+str(k)] = np.std(means_chunck[k,:],axis=1)
                results.loc[0,'std Cs '+xlabel+' '+str(k)] =  np.std(vars_chunck[k,0])
                results.loc[1,'std Cs '+ylabel+' '+str(k)] =  np.std(vars_chunck[k,1])
            if dim > 2: results.loc[2,'std Cs '+zlabel+' '+str(k)] =  np.std(vars_chunck[k,2])

    results.to_csv(suffix+'-dim'+str(dim)+'-'+str(hour)+'h-'+sufx+'.csv')

def resample(data,dim,m,hour,res0,means,Cs,sufx,chunkNum = 10,xlabel='V450-A',ylabel='PE-A',zlabel='FITC-A'):
    dimMap = {0:xlabel,1:ylabel}
    # resample
    data = data.sample(frac=1).reset_index(drop=True)
    if dim >1:
        print(len(data))
        
        chunck = {}

        for g, df in data.groupby(np.arange(len(data)) // (len(data)/chunkNum) ):
            print(g, len(df))
            chunck[g] = df

        means_chunck = np.zeros((m,dim,chunkNum))
        vars_chunck = np.zeros((m,dim,chunkNum))

        for i in range(0,chunkNum):
            print('chunk',i,len(chunck[i]))
            means_ch,Cs_ch,ws = runGM(hour,dim,m,chunck[i],res0,sufx,outf=False,show=False,xlabel=xlabel,ylabel=ylabel,zlabel=zlabel)
            means_chunck[:,:,i] = means_ch
            for k in range(m):
                for d in range(dim):
                    vars_chunck[k,d,i] = Cs_ch[k][d,d]

        for d in range(dim):
            plt.figure()
            plt.title('means '+dimMap[d])
            for k in range(m):
                plt.hist(means_chunck[k,d],label=str(k))
                plt.scatter(means[k,d],4.5)
                print(np.mean(means_chunck[k,d]),np.std(means_chunck[k,d]))
            plt.legend()


        for d in range(dim):
            plt.figure()
            plt.title('vars '+dimMap[d])
            for k in range(m):
                plt.hist(vars_chunck[k,d],label=str(k))
                plt.scatter(Cs[k][d,d],4.5)
                print(np.mean(vars_chunck[k,d]),np.std(vars_chunck[k,d]))
            plt.legend()
        return means_chunck,vars_chunck
