"""Network algorithms; obtain case data separately. No case arrays embedded."""
import os
import numpy as np

with np.load(os.environ["SEGAN_NETWORK_CASE"], allow_pickle=False) as case:
    P_KW = case["P_KW"].copy()
    Q_KVAR = case["Q_KVAR"].copy()
    BRANCH = case["BRANCH"].copy()
    BASE_MVA = float(case["BASE_MVA"])
    BASE_KV = float(case["BASE_KV"])
if P_KW.shape != (33,) or Q_KVAR.shape != (33,) or BRANCH.shape != (32,4):
    raise ValueError("Expected the 33-bus radial case with open ties omitted")

def matrices(n=100):
    parent=BRANCH[:,0].astype(int)-1;child=BRANCH[:,1].astype(int)-1
    descendants=np.zeros((32,33));paths=np.zeros((33,32))
    for k,(i,j) in enumerate(zip(parent,child)):
        assert i<j;paths[j]=paths[i];paths[j,k]=1
    descendants[:]=paths.T
    mapping=np.zeros((33,n));nodes=np.resize(np.array([17,21,24,32]),n);mapping[nodes,np.arange(n)]=1
    z=(BRANCH[:,2]+1j*BRANCH[:,3])/(BASE_KV**2/BASE_MVA)
    ratings=np.maximum(.25,1.2*np.hypot(descendants@P_KW,descendants@Q_KVAR)/1000)
    return dict(parent=parent,child=child,D=descendants,A=paths,M=mapping,z=z,ratings_mva=ratings,pf_tan=float(np.tan(np.arccos(.95))))


def nodal(base,pv,cool,background=1.):
    m=matrices(len(base));P=background*P_KW+m['M']@(base+cool-pv);Q=background*Q_KVAR+m['pf_tan']*(m['M']@(base+cool))
    return P,Q


def linear(P,Q):
    m=matrices();pf=m['D']@P/1000;qf=m['D']@Q/1000
    v2=1-2*m['A']@(m['z'].real*pf/BASE_MVA+m['z'].imag*qf/BASE_MVA)
    return dict(voltage_squared=v2,branch_P_mw=pf,branch_Q_mvar=qf,branch_S_mva=np.hypot(pf,qf))


def ac_flow(P,Q):
    m=matrices();s=(np.array(P)+1j*np.array(Q))/(1000*BASE_MVA);V=np.ones(33,complex)
    for it in range(1,301):
        injection=np.conj(s/V);curr=injection.copy();I=np.zeros(32,complex)
        for k in range(31,-1,-1):
            i=m['parent'][k];j=m['child'][k];I[k]=curr[j];curr[i]+=I[k]
        new=np.ones(33,complex)
        for k in range(32):new[m['child'][k]]=new[m['parent'][k]]-m['z'][k]*I[k]
        err=float(abs(new-V).max());V=new
        if err<1e-12:break
    else:raise RuntimeError('AC forward/backward sweep did not converge')
    sending=V[m['parent']]*np.conj(I)*BASE_MVA;loss=m['z']*abs(I)**2*BASE_MVA*1000
    balance=1000*sending[0].real-np.sum(P)-loss.real.sum()
    assert abs(balance)<1e-6
    return dict(voltage_pu=abs(V),phasor=V,branch_S_mva=abs(sending),loss_kw=float(loss.real.sum()),reactive_loss_kvar=float(loss.imag.sum()),
        slack_P_kw=float(sending[0].real*1000),slack_Q_kvar=float(sending[0].imag*1000),power_balance_error_kw=float(balance),iterations=it)
