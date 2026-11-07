"""Why rankings move with qualifying times: compare the head-to-head record against the size of the gap to teammates."""
import pandas as pd, numpy as np
r=pd.read_pickle('data/processed/results_v2_50.pkl');r=r[r.session=='gp'] if 'session' in r else r
g=pd.read_pickle('data/processed/qtimes_v2.pkl').drop_duplicates(['race_id','driver'])[['race_id','driver','qdef']]
r=r[r.quali_pos.notna()].merge(g,on=['race_id','driver']);r=r[r.qdef.notna()]; r['qdef']=r.qdef.clip(upper=.10)*100
tm=[c for c in ('team','constructor','entrant') if c in r][0]
m=r.merge(r,on=['race_id',tm],suffixes=('','_t'));m=m[m.driver!=m.driver_t]
m['win']=(m.quali_pos<m.quali_pos_t)*1.;m['gap']=m.qdef_t-m.qdef
b=pd.read_csv('outputs_v2/gap/base_long_fullscores.csv').set_index('driver');q=pd.read_csv('outputs_v2/gap/qt_long_fullscores.csv').set_index('driver')
def span(s):
    return tuple(map(int,s.split('-')))
rows=[]
for d in b.index:
    if d not in q.index or pd.isna(b.peak[d]) or pd.isna(q.peak[d]) or b.starts[d]<25: continue
    a,z=span(b.peak_span[d]); x=m[(m.driver==d)&(m.season.between(a,z))]
    if len(x)<15: continue
    rows.append(dict(driver=d,change=q.peak[d]-b.peak[d],h2h=x.win.mean(),gap=x.gap.median(),n=len(x),peak=b.peak[d]))
t=pd.DataFrame(rows).set_index('driver')
print(len(t),'corr change~h2h',round(t.change.corr(t.h2h),3),'change~gap',round(t.change.corr(t.gap),3))
f=np.polyfit(t.h2h,t.gap,1);t['excess']=t.gap-np.polyval(f,t.h2h)
print('change ~ excess gap given h2h',round(t.change.corr(t.excess),3))
X=np.c_[np.ones(len(t)),t.h2h,t.gap];c=np.linalg.lstsq(X,t.change,rcond=None)[0];print('regress change on h2h, gap',c.round(3),'R2',round(1-((t.change-X@c)**2).sum()/((t.change-t.change.mean())**2).sum(),3))
print(t.loc[[d for d in 'michael-schumacher niki-lauda ayrton-senna john-watson patrick-depailler mark-webber jenson-button harry-schell max-verstappen fernando-alonso'.split() if d in t.index]].round(2).to_string())
# Second test: does a driver's score move when his car's rating moves?
cb=pd.read_csv('outputs_v2/gap/base_long_cars.csv');cq=pd.read_csv('outputs_v2/gap/qt_long_cars.csv')
print(cb.head(2).to_string())
s=cb.ts.astype(str).str.extract(r'(?P<team>.*?)[_|\- ]*(?P<season>\d{4})$') if not {'team','season'}<=set(cb) else cb
c=pd.DataFrame(dict(team=s.team,season=s.season.astype(int),b=cb.car_q.values,q=cq.car_q.values))
c['b']-=c.groupby('season').b.transform('mean');c['q']-=c.groupby('season').q.transform('mean')
print('sd of car by season: base',round(c.groupby('season').b.std().mean(),3),'qt',round(c.groupby('season').q.std().mean(),3),'spread ratio qt/base by decade');
print((c.groupby(c.season//10*10).q.std()/c.groupby(c.season//10*10).b.std()).round(3).to_string())
c['dc']=c.q-c.b
sb=pd.read_csv('outputs_v2/gap/base_long_fullseasons.csv');sq=pd.read_csv('outputs_v2/gap/qt_long_fullseasons.csv')
j=sb.merge(sq,on=['driver','season','team'],suffixes=('_b','_q')).merge(c,on=['team','season'],how='left');j['ds']=j.score_q-j.score_b
print('n',j.dc.notna().sum(),'season score change ~ car change corr',round(j.ds.corr(j.dc),3))
for d,a,z in [('michael-schumacher',1994,1996),('niki-lauda',1976,1978),('ayrton-senna',1987,1989),('mark-webber',2009,2011),('jenson-button',2004,2006)]:
    print(d,j[(j.driver==d)&j.season.between(a,z)][['season','team','score_b','score_q','b','q']].round(2).to_string(index=False,header=False).replace('\n',' | '))
