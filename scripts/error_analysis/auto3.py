from load import *
import sacrebleu, unicodedata as u, numpy as np, collections
nf=lambda s:u.normalize('NFC',s).strip()
S=[nf(s) for s in src]; Rf=[nf(r) for r in ref]; H={m:[nf(p) for p in P[m]] for m in MS}
RV=set(w for r in Rf for w in tok(r))
NEG={'না','নি','নেই','নাই','নয়','নাকি'}
NEG2={'না','নি','নেই','নাই','নয়'}
for m in MS:
    cp=0; cw=collections.Counter(); neg=0
    for s,r,h in zip(S,Rf,H[m]):
        st,rt,ht=tok(s),tok(r),tok(h)
        c=[w for w in ht if w in st and w not in RV]
        if c: cp+=1; cw.update(c)
        if (len(NEG2&set(rt))>0)!=(len(NEG2&set(ht))>0): neg+=1
    print(m,'sent w/ copied word never in any test ref:',cp,cw.most_common(12),'| negation mismatch sents:',neg)
# T5 vs NLLB
a,b=H['BanglaT5'],H['NLLB-200']
ca=np.array([sacrebleu.sentence_chrf(x,[r]).score for x,r in zip(a,Rf)]); cb=np.array([sacrebleu.sentence_chrf(x,[r]).score for x,r in zip(b,Rf)])
print('identical outputs',sum(x==y for x,y in zip(a,b)),'| T5 better',(ca>cb+1e-9).sum(),'NLLB better',(cb>ca+1e-9).sum(),'tie',(abs(ca-cb)<1e-9).sum())
print('both exact',sum(x==r and y==r for x,y,r in zip(a,b,Rf)),'only T5 exact',sum(x==r and y!=r for x,y,r in zip(a,b,Rf)),'only NLLB exact',sum(y==r and x!=r for x,y,r in zip(a,b,Rf)))
orc=[x if p>=q else y for x,y,p,q in zip(a,b,ca,cb)]
print('oracle BLEU',round(sacrebleu.corpus_bleu(orc,[Rf]).score,2),'chrF',round(sacrebleu.corpus_chrf(orc,[Rf]).score,2))
allbad=sum(all(H[m][i]!=Rf[i] for m in MS) for i in range(len(Rf))); print('no model exact',allbad,'any exact',len(Rf)-allbad)
print('all 4 identical to each other',sum(len({H[m][i] for m in MS})==1 for i in range(len(Rf))))
# paired bootstrap NFC
rng=np.random.default_rng(42); n=len(Rf); diffs={'BLEU':[], 'chrF':[]}
for _ in range(1000):
    ix=rng.integers(0,n,n); r=[Rf[i] for i in ix]; x=[a[i] for i in ix]; y=[b[i] for i in ix]
    diffs['BLEU'].append(sacrebleu.corpus_bleu(x,[r]).score-sacrebleu.corpus_bleu(y,[r]).score)
    diffs['chrF'].append(sacrebleu.corpus_chrf(x,[r]).score-sacrebleu.corpus_chrf(y,[r]).score)
for k,v in diffs.items(): v=np.array(v); print('NFC T5-NLLB',k,'mean',v.mean().round(2),'95%CI',np.percentile(v,[2.5,97.5]).round(2),'p(one-sided, diff<=0)',(v<=0).mean())
np.save('chrf_t5.npy',ca); np.save('chrf_nllb.npy',cb)
