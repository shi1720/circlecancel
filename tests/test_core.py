import unittest,itertools
from fractions import Fraction as F
import numpy as np
from circlecancel import *

class CoreTests(unittest.TestCase):
 def test_circle_image(self):
  rng=np.random.default_rng(1720406)
  worst=0
  for _ in range(200):
   c=Circle(complex(*rng.normal(size=2)),rng.uniform(.2,2))
   a=c.center+c.radius*rng.choice([.3,2])*np.exp(1j*rng.uniform(0,6.28))
   b=c.center+c.radius*rng.choice([.6,3])*np.exp(1j*rng.uniform(0,6.28))
   m,s=image_disk(a,b,c); z=c.center+c.radius*np.exp(2j*np.pi*np.arange(257)/257)
   worst=max(worst,float(np.max(abs(abs((z-b)/(z-a)-m)-s))/(1+s)))
  self.assertLess(worst,1e-12)
 def test_exact_and_tamper(self):
  c=[Circle(0,1)]
  g,e=certified_edge(.2+.1j,.21+.11j,c)
  self.assertTrue(verify_pair(.2+.1j,.21+.11j,c,g,e))
  self.assertFalse(verify_pair(.2+.1j,.21+.11j,c,g,e/2))
 def test_topology_and_joint_obstruction(self):
  _,e,_=optimal_gain(.999,1.001,[Circle(0,1)])
  self.assertEqual(e,1)
  cs=[Circle(-.1,.01),Circle(.1,.01)]
  self.assertTrue(all(optimal_gain(0,1,[c])[1]<.12 for c in cs))
  self.assertGreaterEqual(optimal_gain(0,1,cs)[1],1-1e-8)
 def test_exact_flow_exhaustive(self):
  rng=np.random.default_rng(7729)
  for case in range(150):
   n=1+case%5
   edges={(i,j):F(int(rng.integers(1,50)),100) for i in range(n) for j in range(n) if rng.random()<.8}
   path=optimal_path(n,n,edges)
   exact={0:F(1)}
   for k in range(1,n+1):
    for iz in itertools.combinations(range(n),k):
     for jp in itertools.permutations(range(n),k):
      pairs=list(zip(iz,jp))
      if all(ij in edges for ij in pairs):
       p=np.prod([1+edges[ij] for ij in pairs])
       if k not in exact or p<exact[k]: exact[k]=p
   self.assertEqual({len(M):p for p,M in path},exact)
 def test_product(self):
  cs=[Circle(0,1),Circle(0,2)]
  aa=[.1+.1j,.2-.1j]; bb=[.101+.102j,.202-.101j]
  gains={}; edges={}
  for i in range(2):
   g,e=certified_edge(aa[i],bb[i],cs); gains[(i,i)]=g; edges[(i,i)]=e
  p,M=select(2,2,edges,F(1,10))
  for c in cs:
   z=c.center+c.radius*np.exp(2j*np.pi*np.arange(8192)/8192)
   self.assertLessEqual(max(abs(ratio(z,aa,bb,M,gains)-1)),float(p-1))


class IndependentReplayTests(unittest.TestCase):
 def test_certificate_replay_and_tamper(self):
  import json,copy
  from pathlib import Path
  from verify_certificate import verify
  cases=json.loads((Path(__file__).resolve().parents[1]/'results/certificates.json').read_text())
  self.assertEqual(verify(cases),(180,1379))
  case=next(copy.deepcopy(c) for c in cases if c['selected'])
  case['selected'][0]['epsilon']='0'
  with self.assertRaises(ValueError):verify([case])
 def test_reduced_representation(self):
  from circlecancel.core import reduced_factors
  aa=[.1,.2];bb=[.11,.8];M={(0,0)};g={(0,0):.99}
  az,bp,k=reduced_factors(aa,bb,M,g)
  z=np.exp(2j*np.pi*np.arange(40)/40)
  red=k*(z-az[0])/(z-bp[0]);original=(z-aa[0])*(z-aa[1])/(z-bb[0])/(z-bb[1])
  self.assertLess(np.max(abs(red/original-ratio(z,aa,bb,M,g))),1e-14)

if __name__=='__main__': unittest.main()
