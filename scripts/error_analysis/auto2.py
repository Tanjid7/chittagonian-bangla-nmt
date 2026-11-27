from load import *
import sacrebleu, unicodedata as u, difflib, numpy as np, json, collections
nf=lambda s:u.normalize('NFC',s).strip()
S=[nf(s) for s in src]; Rf=[nf(r) for r in ref]; H={m:[nf(p) for p in P[m]] for m in MS}
out={}
# length buckets by source words
L=np.array([len(tok(s)) for s in S])
bk=lambda n:'1-3' if n<=3 else '4-5' if n<=5 else '6-7' if n<=7 else '8+'
B=np.array([bk(n) for n in L])
rows=[]
for b in ['1-3','4-5','6-7','8+']:
    idx=np.where(B==b)[0]; row={'bucket':b,'n':len(idx)}
    for m in MS:
        h=[H[m][i] for i in idx]; r=[Rf[i] for i in idx]
        row[m+'_BLEU']=round(sacrebleu.corpus_bleu(h,[r]).score,2); row[m+'_chrF']=round(sacrebleu.corpus_chrf(h,[r]).score,2)
        row[m+'_EM']=round(np.mean([a==c for a,c in zip(h,r)])*100,1)
    rows.append(row)
pd.DataFrame(rows).to_csv('len_buckets.csv',index=False); print(pd.DataFrame(rows).T)
# edit ops
prof=[]
for m in MS:
    om=ad=sb=reft=hypt=rep=copy=reord=lr_short=lr_long=0; copyw=collections.Counter(); omw=collections.Counter()
    for s,r,h in zip(S,Rf,H[m]):
        st,rt,ht=tok(s),tok(r),tok(h); reft+=len(rt); hypt+=len(ht)
        for op,i1,i2,j1,j2 in difflib.SequenceMatcher(a=rt,b=ht,autojunk=False).get_opcodes():
            if op=='delete': om+=i2-i1; omw.update(rt[i1:i2])
            elif op=='insert': ad+=j2-j1
            elif op=='replace':
                k=min(i2-i1,j2-j1); sb+=k; om+=max(0,(i2-i1)-k); ad+=max(0,(j2-j1)-k)
        if any(ht[i]==ht[i+1] and not (ht[i] in rt and rt.count(ht[i])>1) for i in range(len(ht)-1)): rep+=1
        c=[w for w in ht if w in st and w not in rt]
        if c: copy+=1; copyw.update(c)
        if ht!=rt and sorted(ht)==sorted(rt): reord+=1
        if len(ht)<len(rt): lr_short+=1
        if len(ht)>len(rt): lr_long+=1
    prof.append({'model':m,'ref_tokens':reft,'hyp_tokens':hypt,'omitted_per100':round(om/reft*100,1),'added_per100':round(ad/reft*100,1),
      'substituted_per100':round(sb/reft*100,1),'sent_shorter_than_ref':lr_short,'sent_longer_than_ref':lr_long,'sent_with_repetition':rep,
      'sent_with_copied_source_word':copy,'sent_pure_reorder':reord})
    out[m]={'top_copied':copyw.most_common(15),'top_omitted':omw.most_common(15)}
pf=pd.DataFrame(prof); pf.to_csv('edit_profile.csv',index=False); print(pf.T)
json.dump(out,open('topwords.json','w'),ensure_ascii=False,indent=0)
for m in MS: print(m,out[m])
