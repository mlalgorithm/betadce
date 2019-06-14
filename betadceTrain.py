import numpy as np
import argparse
from multiprocessing import Pool
import pandas as pd

def choosenk(n,k):
    kk=np.min([k,n-k])
    if kk<2:
        if kk==0:
            if k==n:
                x=np.arange(0,n)
            else:
                x=[]
        else:
            if k==1:
                x=np.arange(0,n).reshape((-1,1))
            else:
                x=np.arange(0,n)
                x=np.tile(np.arange(0,n),(n-1,1)).T.reshape((n-1,n)).T
    else:
        n1=n+1
        m=1
        for i in np.arange(1,kk+1):
            m=m*(n-kk+i)/i
        x=np.zeros((int(m),k),dtype=np.int)
        f=n1-k
        x[0:f,k-1]=np.arange(k-1,n).reshape((-1,))
        for a in np.arange(k-1,0,-1):
            d,h=f,f
            x[0:f,a-1]=a-1
            for b in np.arange(a+1,a+n-k+1):
                d=int(d*(n1+a-b-k)/(n1-b))
                e=f+1
                f=int(e+d-1)
                x[(e-1):f,a-1]=b-1
                x[(e-1):f,a:k]=x[(h-d):h,a:k]       
    return x

def bnnlearndisFaster(X,Y):
    fm=np.arange(1,X.shape[0]+1)+2
    loss=np.zeros(X.shape[0],)
    for i,ux in enumerate(X):
        t=np.sum((X-ux)**2,axis=1)
        t[i]=np.amax(t)+1;
        ti=t.argsort()
        s=(np.cumsum(Y[ti])+1)/fm
        smax=np.amax(s)
        smin=np.amin(s)
        if smax>=1-smin:
            loss[i]=-np.log(smax)
        else:
            loss[i]=-np.log(1-smin)  
    return np.mean(loss[Y==0])+np.mean(loss[Y==1])

def bnnlearndisFaster1d(X,Y):
    fm=np.arange(1,X.shape[0]+1)+2
    loss=np.zeros(X.shape[0],)
    for i,ux in enumerate(X):
        t=(X-ux)**2
        t[i]=np.amax(t)+1;
        ti=t.argsort()
        s=(np.cumsum(Y[ti])+1)/fm
        smax=np.amax(s)
        smin=np.amin(s)
        if smax>=1-smin:
            loss[i]=-np.log(smax)
        else:
            loss[i]=-np.log(1-smin)
    return np.mean(loss[Y==0])+np.mean(loss[Y==1])

def bnnlearndisFasterM(X,Y):
    Xc=np.unique(X[np.where(Y==1)[0],:],axis=0)
    fm=np.arange(1,X.shape[0]+1)+2
    smax0,dmax0,xmax0=0,0,0
    for ux in Xc:
        t=np.sum((X-ux)**2,axis=1)
        ti=t.argsort()
        s=(np.cumsum(Y[ti])+1)/fm
        t=t[ti]
        s[0:-1][t[1:]==t[0:-1]]=0
        smax=np.max(s)
        j=np.argmax(s)
        dmax=t[j]
        if smax>smax0:
            smax0=smax
            dmax0=dmax
            xmax0=ux
    return (smax0,dmax0,xmax0)

def bnnlearndisFasterM1d(X,Y):
    Xc=np.unique(X[np.where(Y==1)[0]])
    fm=np.arange(1,X.shape[0]+1)+2
    smax0,dmax0,xmax0=0,0,0
    for ux in Xc:
        t=(X-ux)**2
        ti=t.argsort()
        s=(np.cumsum(Y[ti])+1)/fm
        t=t[ti]
        s[0:-1][t[1:]==t[0:-1]]=0
        smax=np.max(s)
        j=np.argmax(s)
        dmax=t[j]
        if smax>smax0:
            smax0=smax
            dmax0=dmax
            xmax0=ux
    return (smax0,dmax0,xmax0)

def pmaxcal(rmax,nfeatures):
    # to fullfil: max return, s.t. C(return,nfeatures)<=rmax
    pm=rmax
    for i in np.arange(1,nfeatures+1):
        pm=pm*i
    p0=np.ceil(pm**(1/i))
    j0=p0
    jm=1
    for i in np.arange(1,nfeatures+1):
        jm=jm*j0
        j0=j0-1
    if jm>pm:
        return int(p0-1)
    for j in np.arange(1,nfeatures+1):
        jm=jm*(p0+j)/(p0+j-nfeatures)
        if jm>pm:
            return int(p0+j-1)
        
class Engine():
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    def __call__(self,c):
        return bnnlearndisFaster(self.X[:,c],self.Y)
    
def lbsmodel(X,Y,rmax,outfile,ncpu):
    itermax=50
    idx=np.where(np.std(X,axis=0)>0.00001)[0]
    X=X[:,idx]
    dmean=np.mean(X,axis=0)
    dstd=np.std(X,axis=0)
    X=(X-dmean)/dstd
    p=X.shape[1]
    print('epoch: '+str(1))
    smax=np.zeros(p)
    for i in np.arange(0,p):
        smax[i]=bnnlearndisFaster1d(X[:,i],Y) 
    sfr=np.argsort(smax,kind='stable')
    smaxk=smax[sfr[0]]
    print('loss = '+str(smaxk))
    loss=smaxk
    ind=np.where(smax==smaxk)[0][0]
    subset=idx[ind]
    print('subset= ',str(subset))
    #(a,radius,center)=bnnlearndisFasterM1d(X[:,ind],Y)
 
    pmax=pmaxcal(rmax,2)
    if pmax>p:
        pmax=p
    
    us=sfr[0:pmax]
    for k in np.arange(2,itermax):
        print('epoch: '+str(k))
        comb=us[choosenk(pmax,k)]
        if ncpu>=1:
            pool=Pool(ncpu)
        else:
            pool=Pool()
        engine=Engine(X,Y)
        smax=np.asarray(pool.map(engine,comb),dtype=np.float32)
        sfr=np.argsort(smax,kind='stable')
        smaxk=smax[sfr[0]]
        print('loss = '+str(smaxk))
        if smaxk<loss*0.95:
            loss=smaxk
            ind=np.where(smax==smaxk)[0][0]
            subset=idx[comb[ind]]
            print('subset= ',str(subset))
            #(a,radius,center)=bnnlearndisFasterM(X[:,comb[ind]],Y)

            if k==itermax-1:
                print("maximum iter reached!")
                print('subset= ',str(subset))
                print('Training finished. Result is saved in %s.\n\nResult:'%(outfile))
                print('loss= '+str(loss)+', subset= '+str(subset))
                np.savez(outfile,loss=loss,subset=subset,dmean=dmean,dstd=dstd,idx=idx,Xs=X[:,comb[ind]],Ys=Y)
                return loss
        else:
            loss=smaxk
            ind=np.where(smax==smaxk)[0][0]
            subset=idx[comb[ind]]
            print('subset= ',str(subset))
            print('Training finished. Result is saved in %s.\n\nResult:'%(outfile))
            print(' loss= '+str(loss)+',\n subset= '+str(subset))
            np.savez(outfile,loss=loss,subset=subset,dmean=dmean,dstd=dstd,idx=idx,Xs=X[:,comb[ind]],Ys=Y)
            return loss
        
        us,ind=np.unique(comb[sfr],return_index=True)
        us=us[np.argsort(ind)]
        pmax=pmaxcal(rmax,k+1)
        if pmax>p:
            pmax=p
        us=us[:pmax].astype(int)



def main():
    parser = argparse.ArgumentParser(description='Training by AKNN')
    parser.add_argument('--data', type=str, metavar='D', help='dataset to be trained, the last column is regarded as label (required)')
    parser.add_argument('--ne', type=int, default=2000000, metavar='N', help='maximum number of feature subsets to be evaluated in each epoch. the default is 2000000 (optional)')
    parser.add_argument('--out', type=str, default='model', metavar='O', help='filename to keep the output of training. the default is model.npz (optional)')
    parser.add_argument('--cpu', type=int, default=0, metavar='C', help='the number of CPUs to use. the default is to use all of CPUs available (optional)') 
    args = parser.parse_args()
    print('Reading the data...')
    #data=np.loadtxt(args.data,delimiter=',',dtype=np.float32)
    data=pd.read_csv(args.data,header=None,delimiter=',')
    #m={'Active':1,'Inactive':0}
    #m={'Active':1,'Inconc':0}
    #m={'Active':1,'Inconclusive':0}
    #data[data.columns[-1]]=data[data.columns[-1]].map(m)
    #data[data.columns[-1]]=(data[data.columns[-1]]=="Active").astype('int')
    data=data.values
    label=data[:,-1]
    #print(label)

    un=np.unique(label)
    if un[0]!=0 or un[1]!=1:
        print("label should be 0,1")
        print(un)
    #print(np.shape(np.unique(label)))

    #data=(data-dmean)/dstd
    #data=(data-dmin)/(dmax-dmin)
    #print(data.shape)
    print("Number of samples = %d" %(data.shape[0]))
    print("Number of variables = %d" %(data.shape[1]-1))
    #print(data)
    print('Start training:')
    if args.out[-4:]!='.npz':
        args.out=args.out+'.npz'
        #print("Result is saved in %s\n" %(args.out))
    lbsmodel(data[:,:-1],label,args.ne,args.out,args.cpu)


if __name__=='__main__':
    main()
