"""Independent exact verifier; Python standard library only.

Uses the degree-one trigonometric nonnegativity criterion, rather than
circle-image centers, optimizer output, or sampled boundary points.
"""
import json,sys
from fractions import Fraction as F

def z(x): return tuple(F(t) for t in x)
def plus(x,y):return x[0]+y[0],x[1]+y[1]
def minus(x,y):return x[0]-y[0],x[1]-y[1]
def times(x,y):return x[0]*y[0]-x[1]*y[1],x[0]*y[1]+x[1]*y[0]
def conj(x):return x[0],-x[1]
def norm(x):return x[0]**2+x[1]**2
def scale(x,k):return x[0]*k,x[1]*k

def check_edge(a,b,c,R,g,e):
    v=minus(g,(F(1),F(0)))
    d=plus(times(v,c),minus(a,times(g,b)))
    A=minus(c,a)
    q=e*e*(R*R+norm(A))-R*R*norm(v)-norm(d)
    l=minus(scale(conj(A),e*e),times(v,conj(d)))
    return q>=0 and q*q>=4*R*R*norm(l)

def verify(data):
    total=0
    for case in data:
        iz=set(); jp=set(); product=F(1)
        for circ in case['circles']:
            c,R=z(circ['center']),F(circ['radius'])
            assert R>0
            for root in case['zeros']+case['poles']:
                assert norm(minus(z(root),c))!=R*R, 'root on protected circle'
        for edge in case['selected']:
            i,j=edge['i'],edge['j']
            assert i not in iz and j not in jp, 'not a matching'
            iz.add(i); jp.add(j)
            a,b=z(case['zeros'][i]),z(case['poles'][j])
            g,e=z(edge['gain']),F(edge['epsilon'])
            assert norm(g)>0 and 0<=e<1
            for circ in case['circles']:
                c,R=z(circ['center']),F(circ['radius'])
                assert R>0
                assert norm(minus(a,c))!=R*R and norm(minus(b,c))!=R*R
                assert check_edge(a,b,c,R,g,e), 'circle inequality failed'
            product*=1+e
            total+=1
        assert product-1<=F(case['budget']), 'budget exceeded'
    return len(data),total

if __name__=='__main__':
    with open(sys.argv[1] if len(sys.argv)>1 else 'results/certificates.json') as f: data=json.load(f)
    n,k=verify(data)
    print(f'PASS: {n} cases, {k} selected cancellations; exact continuous-circle inequalities and total budgets verified.')
