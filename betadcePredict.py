import numpy as np
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='prediction by trained AKNN')
    parser.add_argument('--data', type=str, metavar='D', help='data to be predicted (required)')
    parser.add_argument('--model', type=str, default='model', metavar='M', help='filename of the model trained. the default is model.npz (optional)')
    parser.add_argument('--result', type=str, default='Result', metavar='R', help='filename to keep the result of prediction. the default is Result.txt (optional)')
    args = parser.parse_args()
    if args.result[-4:]!='.txt':
        args.result=args.result+'.txt'
    if args.model[-4:]!='.npz':
        args.model=args.model+'.npz'
    model=np.load(args.model)
    subset=model['subset']
    dmean=model['dmean']
    dstd=model['dstd']
    idx=model['idx']
    Xs=model['Xs']
    Ys=model['Ys']

    #Xtest=np.loadtxt(args.data,delimiter=',',dtype=np.float32)
    Xtest=pd.read_csv(args.data,header=None,delimiter=',')
    #print(Xtest.shape)
    print("Number of samples = %d" %(Xtest.shape[0]))
    print("Number of variables = %d" %(Xtest.shape[1]-1))
    print("Index, Prediction, True label")
    #m={'Active':1,'Inactive':0}
    #Xtest[Xtest.columns[-1]]=Xtest[Xtest.columns[-1]].map(m)
    #Xtest[Xtest.columns[-1]]=(Xtest[Xtest.columns[-1]]=="Active").astype('int')
    Xtest=Xtest.values
    Ytest=Xtest[:,-1]
    Xtest=Xtest[:,idx]
    Xtest=(Xtest-dmean)/dstd
    #Xtest=(Xtest-dmin)/(dmax-dmin)
    Xtest=Xtest[:,subset]
    fm=np.arange(1,Xs.shape[0]+1)+2
    Ypred=np.zeros(Xtest.shape[0])
    for i,ux in enumerate(Xtest):
        t=np.sum((Xs-ux)**2,axis=1)
        t[i]=np.amax(t)+1;
        ti=t.argsort()
        s=(np.cumsum(Ys[ti])+1)/fm
        smax=np.amax(s)
        smin=np.amin(s)
        if smax>=1-smin:
            Ypred[i]=1
    #print(np.concatenate((Ypred,Ytest),axis=0))
    for i in np.arange(Ypred.shape[0]):
        print("%d, %d, %d" % (i+1,Ypred[i],Ytest[i]))
    cnum=(Ypred==Ytest).sum()
    n=Ypred.shape[0]
    print('Accuracy = %.4f [%d/%d]'% (cnum/n,cnum,n))
    print('Result is saved in %s.'%(args.result))
    
    with open(args.result,'w+') as f:
        for i in np.arange(Ypred.shape[0]):
            f.write("%d, %d, %d\n" % (i,Ypred[i],Ytest[i]))

if __name__=='__main__':
    main()
