"""Rankings with vs without qualifying times; same data, same fit length; also held-out by decade."""
import pandas as pd, numpy as np
b=pd.read_csv('outputs_v2/gap/base_long_fullscores.csv').set_index('driver'); q=pd.read_csv('outputs_v2/gap/qt_long_fullscores.csv').set_index('driver')
j=b.join(q[['peak','career','peak_span','peak_lo','peak_hi']],rsuffix='_q')
p=j[j.starts>=25].dropna(subset=['peak','peak_q'])
print('n',len(p),'peak spearman',round(p.peak.corr(p.peak_q,method='spearman'),3),'career',round(j.career.corr(j.career_q,method='spearman'),3))
print('mean abs peak change',round((p.peak-p.peak_q).abs().mean(),3),'median peak CI width',round((p.peak_hi-p.peak_lo).median(),2))
p=p.assign(rb=p.peak.rank(),rq=p.peak_q.rank(),d=p.peak_q-p.peak)
print(p.sort_values('peak_q').head(25)[['name','peak','peak_q','rb','rq','peak_span','peak_span_q']].round(2).to_string())
t=p[p.rb<=60]
print('moved most (top 60): better with times');print(t.sort_values('d').head(10)[['name','first','last','peak','peak_q','rb','rq']].round(2).to_string())
print('worse');print(t.sort_values('d').tail(10)[['name','first','last','peak','peak_q','rb','rq']].round(2).to_string())
print('by decade of debut, mean peak change');print(p.groupby(p['first']//10*10).d.agg(['mean','count']).round(3).to_string())
c=j.assign(d=j.career_q-j.career,rb=j.career.rank(ascending=False),rq=j.career_q.rank(ascending=False))
print(c.sort_values('career_q',ascending=False).head(15)[['name','career','career_q','rb','rq']].round(1).to_string())
a=pd.read_pickle('outputs_v2/gap/base_long_pairs.pkl');z=pd.read_pickle('outputs_v2/gap/qt_long_pairs.pkl')
def ll(g): pp=np.clip(g.p.values,1e-6,1-1e-6);return -(g.y.values*np.log(pp)+(1-g.y.values)*np.log(1-pp))
a['lb']=ll(a);a['lq']=ll(z);a['dec']=a.season//10*10
print(a.groupby(['session','dec','same'])[['lb','lq']].mean().round(4).assign(n=a.groupby(['session','dec','same']).size()).to_string())
