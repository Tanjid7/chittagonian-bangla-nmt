from load import *
import sacrebleu, unicodedata as u
N=lambda L:[u.normalize('NFC',s) for s in L]
print('refs changed by NFC:',sum(u.normalize('NFC',r)!=r for r in ref),'/',len(ref))
for m in MS:
    print(m,'preds changed by NFC:',sum(u.normalize('NFC',p)!=p for p in P[m]))
    for tag,h,r in [('raw',P[m],ref),('NFC',N(P[m]),N(ref))]:
        b=sacrebleu.corpus_bleu(h,[r]).score; c=sacrebleu.corpus_chrf(h,[r]).score
        em=sum(x.strip()==y.strip() for x,y in zip(h,r))/len(r)
        print(f'  {tag}: BLEU {b:.2f} chrF {c:.2f} exact {em:.3f}')
