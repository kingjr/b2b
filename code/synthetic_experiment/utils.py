import numpy as np

def sonquist_morgan(x):
    z=np.sort(x)
    n=z.size
    m1=0
    m2=np.sum(z)
    mx=0
    best=-1
    for i in range(n-1):
        m1+=z[i]
        m2-=z[i]
        ind=(i+1)*(n-i-1)*(m1/(i+1)-m2/(n-i-1))**2
        if ind>mx :
            mx=ind
            best=z[i]
    res=[0 for i in range(n)]
    for i in range(n):
        if x[i]>best: res[i] = 1
    return np.array(res)
