"""Full-data car check and named champion cars, across fits. args: fit names in outputs_v2/gap"""
import sys,pickle,numpy as np,pandas as pd
from scipy.stats import spearmanr
from scripts.car_check import standings
S=standings();NAM=[('mercedes',1954,'ferrari'),('mercedes',1955,'ferrari'),('lotus',1962,'ferrari'),('lotus',1963,'ferrari'),('lotus',1965,'ferrari'),('maserati',1957,'ferrari'),('mclaren',1988,'ferrari'),('williams',1992,'mclaren'),('ferrari',2002,'mclaren'),('red-bull',2023,'mclaren'),('mercedes',2016,'red-bull')]
for m in sys.argv[1:]:
    c=pd.read_csv(f'outputs_v2/gap/{m}_fullcars.csv');c['car']=(c.car_q+c.car_r)/2;c['team']=c.ts.str.split('|').str[0];c['season']=c.ts.str.split('|').str[1].astype(int)
    d=c.merge(S,on=['season','team']).dropna(subset=['wcc']);per=d.groupby('season')[['car','wcc']].apply(lambda x:spearmanr(-x.car,x.wcc)[0] if len(x)>3 else np.nan)
    d['rk']=d.groupby('season').car.rank(ascending=False)
    print(m,'all',round(per.mean(),3),'66-75',round(per[per.index<=1975].mean(),3),'top2',round((d[d.rk==1].wcc<=2).mean(),3))
    c['cen']=c.car-c.groupby('season').car.transform('mean');print(' largest edge over season mean: ',c.nlargest(6,'cen')[['ts','cen']].round(2).values.tolist())
    print(' best-to-second gap mean',round(c.groupby('season').car.apply(lambda x:(lambda v:v[0]-v[1])(np.sort(x.values)[::-1])).mean(),3))
    r=[]
    for a,y,b in NAM:
        k=c.set_index('ts');
        if f'{a}|{y}' in k.index and f'{b}|{y}' in k.index:r.append(f"{a[:4]}{y} {k.car[f'{a}|{y}']-k.car[f'{b}|{y}']:+.2f}")
    print(' ',r)
