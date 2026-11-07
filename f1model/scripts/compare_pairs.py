import pandas as pd,numpy as np,sys,warnings;warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from scripts.car_check import standings
base,others=sys.argv[1],sys.argv[2:]
ll=lambda x:-(x.y*np.log(x.p.clip(.001,.999))+(1-x.y)*np.log(1-x.p.clip(.001,.999)))
def car(m):
    c=pd.read_csv(f'outputs_v2/gap/{m}_cars.csv');c['car']=(c.car_q+c.car_r)/2;c['team']=c.ts.str.split('|').str[0];c['season']=c.ts.str.split('|').str[1].astype(int)
    d=c.merge(standings(),on=['season','team']).dropna(subset=['wcc'])
    per=d.groupby('season')[['car','wcc']].apply(lambda x:spearmanr(-x.car,x.wcc)[0] if len(x)>3 else np.nan)
    d['rk']=d.groupby('season').car.rank(ascending=False)
    return per.mean().round(3),per[per.index<=1975].mean().round(3),(d[d.rk==1].wcc<=2).mean().round(3)
print(base,'car',car(base))
a=pd.read_pickle(f'outputs_v2/gap/{base}_pairs.pkl');a['l']=ll(a);a['ac']=(a.p>.5)==(a.y==1)
for m in others:
    b=pd.read_pickle(f'outputs_v2/gap/{m}_pairs.pkl');a['l2']=ll(b).values;a['ac2']=(b.p.values>.5)==(a.y==1)
    print('==',m,'car',car(m))
    for s,g in a.groupby('session'):
        for lab,x in[('all',g),('teammates',g[g.same]),('cross',g[~g.same])]:
            t=x.groupby('season').apply(lambda t:(t.l2-t.l).sum());r=np.random.default_rng(0);N=len(x)
            bs=[t.sample(len(t),replace=True,random_state=int(r.integers(1e9))).sum()/N for _ in range(500)]
            print(f'{s:10s} {lab:9s} n={N:6d} {x.l.mean():.4f} {x.l2.mean():.4f} acc {x.ac.mean():.3f} {x.ac2.mean():.3f} delta90 {np.quantile(bs,.05):+.4f} {np.quantile(bs,.95):+.4f}')
