import os
import sys
import pandas as pd
#import csv
import matplotlib.pyplot as plt
import numpy as np

#from matplotlib.colors import LogNorm
#from skimage.morphology import square
import fcGMM as gmm
#from pandas.plotting import scatter_matrix
#import scipy.stats as st
#from skimage.morphology import square
#from matplotlib.widgets import Slider, Button, RadioButtons
#import seaborn as sns
import argparse
from ast import literal_eval


def parse_arguments():
    """
    Parser import arguments from command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessing', action='store_true', default=False)
    parser.add_argument('--setInit', action='store_true', default=False)
    parser.add_argument('--dim', action='store', default=2)
    parser.add_argument('-i', action='store', type = str, default='PKH-CTV', help='file with fcs file paths')
    #parser.add_argument('--gaussianDetection', action='store_true', default=False)
    parser.add_argument("--times",  nargs="*",  # 0 or more values expected => creates a list
                            type=int, default=[-1,-2,-3],  # default if nothing is provided
                            )
    return parser.parse_args()
args=parse_arguments()
print(args)

if __name__=="__main__":
    dirEx = os.getcwd() +'/'
    print('cwd', dirEx)

    dirRes = dirEx+'result-data/'
    if not os.path.exists(dirRes):
        os.makedirs(dirRes)
    dirInit = dirEx+'init/'
    if not os.path.exists(dirInit):
        os.makedirs(dirInit)
    dirClean = dirEx+'cleaned/'
    if not os.path.exists(dirClean):
        os.makedirs(dirClean)
    dirPlots = dirEx+'plots/'
    if not os.path.exists(dirPlots):
        os.makedirs(dirPlots)

    datafile = {}
    sufx = 'data'+args.i#args.i.split('.')[0]
    print('sufx', sufx)

    print('i',args.i)
    ylab = None
    if ('PKH' in args.i) and (not 'MITO' in args.i):#args.i == 'dataPKH-CTV.dat':
        xlab = 'V450-A'
        ylab = 'PE-A'
        zlab = None
    if (not 'PKH' in args.i) and ('MITO' in args.i):#args.i == 'dataMITO-CTV.dat':
        xlab = 'V450-A'
        ylab = 'FITC-A'
        zlab = None
    if ('CTV' in args.i) and ('PKH' in args.i) and ('MITO' in args.i): #args.i == 'dataPKH-CTV-MITO.dat':
        xlab = 'V450-A'
        ylab = 'PE-A'
        zlab = 'FITC-A'
    if ('CTY' in args.i) and ('MITOFR' in args.i):#args.i == 'dataPKH-CTV.dat':
        xlab = 'PE-A'
        ylab = 'APC-A'
        zlab = None
    if ('CTY' in args.i) and ('PKHFitc' in args.i):#args.i == 'dataPKH-CTV.dat':
        xlab = 'PE-A'
        ylab = 'FITC-A'
        zlab = None
    if ('CTFR' in args.i) and ('PKHFitc' in args.i):#args.i == 'dataPKH-CTV.dat':
        xlab = 'APC-A'
        ylab = 'FITC-A'
        zlab = None
    if ('CTV' in args.i) and (int(args.dim) == 1):
        xlab = 'V450-A'
        ylab = None
        zlab = None    
    print('xlab',xlab,'ylab',ylab,'zlab',zlab)

    datadir = ''
    with open(sufx+'.dat') as f:
        print('read .dat')
        for line in f:
            line = line[:-1]
            #print(line)
            #print(line.split(':'))
            if os.path.isdir(dirEx+line):
                datadir = dirEx+line
            elif os.path.isfile(datadir+line.split(':')[-1]):
                try:
                    datafile[int(line.split(':')[0])] = datadir+line.split(':')[-1]
                except:
                    datafile[line.split(':')[0]] = datadir+line.split(':')[-1]
            else:
                print('line error', datadir+line.split(':')[-1])
                print('file does not exist',os.path.isfile(datadir+line.split(':')[-1]))
                print('dir does not exist',os.path.isdir(dirEx+line) )
    
    # names = []
    # #print('get names')
    # #print(args.times,args.times == [-1,-2,-3])
    # #print(datafile.keys())
    # if args.times == [-1,-2,-3]:
    #     names = [n for n in datafile.keys() if isinstance(n, int)]
    # else:
    #     names = args.times
    # names.sort()
    # print('-----names-----',names)
    # names.append('AutoFl')

    if args.times == [-1, -2, -3]:
        names = gmm.getAq(datafile)
    else:
        #print('times',args.times)
        names = args.times
        names.sort()
        #names = ['AutoFl']+names
    #print('-----names-----', names)

    if args.preprocessing:
        print('-------------------run preprocessing-------------------------')
        dataframe = gmm.doPreproc(datafile, sufx, names, dirClean,xlab,ylab,zlab)

        name = 'AutoFl'
        dfAuto = dataframe[name]
        lx = 'log '+xlab# V450-A'
        if not ylab == None: ly = 'log '+ylab#PE-A'
        else: ly = None
        if not zlab == None: lz ='log '+zlab
        else: lz = None
        #if ('CTY' in args.i) and ('MITOFR' in args.i):#args.i == 'dataPKH-CTV.dat':
        #    lx = 'log PE-A'
        #    ly = 'log APC-A'
        #    lz = None
        
        
        if not lz == None:
            temp,x,y,z = gmm.getCols(dfAuto,lx,ly,lz)
            gmm.plot3dloglog(x,y,z,name,llim=1,xlabel=lx,ylabel=ly,zlabel=lz,dfAuto=dfAuto)
            plt.savefig(dirPlots+'3dplot'+name+'h'+sufx+'.png')
        elif not ly == None:
            temp,x,y = gmm.getCols(dfAuto,lx,ly,lz)
            gmm.plot2dloglog2(x,y,name,llim=1,lim = 18,xlabel=lx,ylabel=ly,nbins = 100.0,dfAuto=dfAuto)  
        else:
            temp,x = gmm.getCols(dfAuto,lx,ly,lz)
            gmm.plot1dlog(x,name,llim=1,lim = 18,xlabel=lx,nbins = 100.0,dfAuto=dfAuto)
        
        #names = dataframe.keys()
        #names.sort()
        for name in names: 
            df = dataframe[name]
            if isinstance(name, int):
                namestr = str(name)+' h '
                #print(namestr)
                #print('df shape',df.shape)
                if not lz == None:
                    df,x,y,z = gmm.getCols(df,lx,ly,lz)
                    #print('df shape',df.shape)
                    gmm.plot3dloglog(x,y,z,namestr,llim=1,
                                    xlabel=lx,ylabel=ly,zlabel=lz,dfAuto=dfAuto)
                    plt.savefig(dirPlots+'3dplot'+str(name)+'h-cor.png')
                elif not ly == None:
                    df,x,y = gmm.getCols(df,lx,ly,lz)
                    gmm.plot2dloglog2(x,y,namestr,llim=1,lim = 18,xlabel=lx,ylabel=ly,nbins = 100.0,dfAuto=dfAuto)
                    plt.savefig(dirPlots+'2dplot'+str(name)+'h-cor.png')
                else:
                    df,x = gmm.getCols(df,lx,ly,lz)
                    gmm.plot1dlog(x,namestr,llim=1,lim = 18,xlabel=lx,nbins = 100.0,dfAuto=dfAuto)
                    plt.savefig(dirPlots+'1dplot'+str(name)+'h-cor.png')
        plt.show()
    else:
        aqName = gmm.getAq(datafile)
        dataPars = gmm.readPreProcPars(aqName)#,path=dirClean)
        dataframe = {}
        for k in aqName:
            df = pd.read_pickle(dirClean+'cleaned'+str(k)+'h.pkl')
            dataframe[k] = df


    if args.setInit and (int(args.dim) == 2):
        print('-------------------set init values-------------------------')
        names = []
        if args.times == [-1, -2, -3]:
            names = gmm.getAq(datafile)
        else:
            names = args.times
            names.sort()

        for name in names:
            if isinstance(name, int):
                from matplotlib.widgets import Slider, Button, RadioButtons
                data = dataframe[name]
                hour = name
                dim = 2
                x = data.loc[:,'log '+xlab]#V450-A']
                y = data.loc[:, 'log '+ylab]
                xx, yy,f,kernel = gmm.getKernel(x,y)
                xy = np.vstack([x,y])
                valK = kernel(xy)
                print('hour ',hour,' dim ',dim)
                initFN = dirInit + 'init-dim'+str(dim)+'-'+str(hour)+'h-'+sufx+'.csv'
                print(initFN)
                if os.path.isfile(initFN):
                    res0 = pd.read_csv(initFN)
                    m = np.sum(['mean' in r for r in res0.columns])
                    means , Cs = gmm.getMeanCsFromRes(res0,m,dim,
                                                        xlabel=xlab,ylabel=ylab,zlabel=zlab)
                    print(means , Cs)
                    sig = 0.04
                else:               
                    means = []
                    sig = 0.04
                    Cs = []

                fig, ax = plt.subplots()
                plt.subplots_adjust(left=0.25, bottom=0.30)  
                ax.scatter(x,y,1,c = valK)
                #print('means',means)
                #for i,m,c in zip(range(len(means)),means,Cs):
                    # lam,v = gmm.drawCovEllipse(m,c,ax,i)
                title = 'Timepoint Hour '+str(hour)+'. Set single gaussain centroid'
                gmm.regenax(x,y,valK,means,Cs,ax,title,xlab,ylab)
                #gmm.regenax(x,y,valK,m,mean,sig,ax)
                #m = input("Enter number of Goussians: ") 
                #plt.show()

                def onclick(event):
                    global means
                    ix, iy = event.xdata, event.ydata
                    print('x = %d, y = %d'%(ix, iy) )
                    means.append(np.array([ix, iy]))
                    Cs.append(np.array([[sig, 0],[0,sig]]))
                    gmm.regenax(x,y,valK,means,Cs,ax,title,xlab,ylab)
                    fig.canvas.draw_idle()
                    #fig.canvas.mpl_disconnect(cid)
                    #if len(means) == m:
                    #    fig.canvas.mpl_disconnect(cid)
                cid = fig.canvas.mpl_connect('button_press_event', onclick)
                plt.show()
                print(means)
                m = len(means)

                fig, ax = plt.subplots()
                plt.subplots_adjust(left=0.25, bottom=0.30)
                title = 'Timepoint Hour '+str(hour)+'. Set single gaussain mean and var'
                #sigs = []
                #if Cs == None:            
                #    for i in range(m):
                #        sigs.append(np.array([[sigx,0],[0,sigy]]))
                #else:
                #for C in Cs:
                #    sigs.append(C)
                gmm.regenax(x,y,valK,means,Cs,ax,title,xlab,ylab)

                gauss = 0
                axM = plt.axes([0.25, 0.25, 0.65, 0.03])#, facecolor=axcolor)
                axM2 = plt.axes([0.25, 0.20, 0.65, 0.03])#, facecolor=axcolor)
                axSig = plt.axes([0.25, 0.15, 0.65, 0.03])#, facecolor=axcolor)
                axSig2 = plt.axes([0.25, 0.10, 0.65, 0.03])#, facecolor=axcolor)
                sMx = Slider(axM, 'mean x: ', 6., 18., valinit=means[0][0])
                sMy = Slider(axM2, 'mean y: ', 6., 18., valinit=means[0][1])
                sSigx = Slider(axSig, 'sigma x: ', 0.01, 0.6, valinit=Cs[0][0,0])
                sSigy = Slider(axSig2, 'sigma y: ', 0.01, 0.6, valinit=Cs[0][1,1])

                raxM = plt.axes([0.025, 0.5, 0.15, 0.15])
                rbs = []
                for i in range(m):
                    rbs.append('m='+str(i+1))
                print(m,tuple(rbs))
                radioM = RadioButtons(raxM, tuple(rbs))
                  

                def update(val):
                    #m = int(label)#sM.val)
                    mx = sMx.val
                    my = sMy.val
                    means[gauss] = [mx,my]
                    sigx = sSigx.val
                    sigy = sSigy.val
                    Cs[gauss] = np.array([[sigx,0],[0,sigy]])
                    print('Cs',Cs)
                    gmm.regenax(x,y,valK,means,Cs,ax,title,xlab,ylab)
                    fig.canvas.draw_idle()
                #sM.on_changed(update)
                sMx.on_changed(update)
                sMy.on_changed(update)
                sSigx.on_changed(update)
                sSigy.on_changed(update)

                def mfunc(label):
                    global gauss,sSigx,sSigy
                    gauss = int(label[-1]) -1
                    mx = means[gauss][0]
                    my = means[gauss][1]
                    sigx = Cs[gauss][0]
                    sigy = Cs[gauss][1]
                    #axSig.clear()
                    #axSig2.clear()
                    #sSigx = Slider(axSig, 'sigma x: ', 0.01, 0.2, valinit=sigx)
                    #sSigy = Slider(axSig2, 'sigma y: ', 0.01, 0.2, valinit=sigy)
                    sMx.set_val(mx)
                    sMy.set_val(my)
                    sSigx.set_val(sigx)
                    sSigy.set_val(sigy)
                    fig.canvas.draw_idle()
                    print(gauss)
                #    gmm.regenax(x,y,valK,means,sig,ax)
                #    fig.canvas.draw_idle()
                radioM.on_clicked(mfunc)
                
                plt.show()

                #Cs = []
                #for s in sigs:
                #    C = np.diagflat(s)
                #    Cs.append(C)
                print('xlab',xlab)
                gmm.writeGaussians(hour,dim,m,means,Cs,
                                   dirInit+'init',sufx,
                                    xlabel=xlab,ylabel=ylab,zlabel=zlab)


    print('-------------------run gaussian mixture -------------------------')

    names = []
    if args.times == [-1, -2, -3]:
        names = gmm.getAq(datafile)
    else:
        names = args.times
        names.sort()
    print('file names',names)

    for name in names:
        if isinstance(name, int):
            print('---file: ',name,'---')
            data = dataframe[name]
            hour = name
            dim = int(args.dim)
            print('dim',dim)

            res0 = pd.read_csv(dirInit+'init-dim'+str(dim)+'-'+str(hour)+'h-'+sufx+'.csv')
            m = np.sum(['mean' in r for r in res0.columns])
            means,Cs,ws = gmm.runGM(hour,dim,m,data,res0,sufx,outf=True,show=False,cond=False,
            							xlabel=xlab,ylabel=ylab,zlabel=zlab)
            plt.show()
            for k in range(m):
                data.loc[:,'w'+str(k)] = ws[k]
            data.to_pickle(dirRes+'weighted'+str(name)+'h'+sufx+'.pkl',protocol=3)
            data.to_csv(dirRes+'weighted'+str(name)+'h'+sufx+'.csv', index=False)
