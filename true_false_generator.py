from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import tensorflow as tf
import scipy
text = "There is a lot of volcanic activity at divergent plate boundaries in the oceans. For example, many undersea volcanoes are found along the Mid-Atlantic Ridge. This is a divergent plate boundary that runs north-south through the middle of the Atlantic Ocean. As tectonic plates pull away from each other at a divergent plate boundary, they create deep fissures, or cracks, in the crust. Molten rock, called magma, erupts through these cracks onto Earth’s surface. At the surface, the molten rock is called lava. It cools and hardens, forming rock. Divergent plate boundaries also occur in the continental crust. Volcanoes form at these boundaries, but less often than in ocean crust. That’s because continental crust is thicker than oceanic crust. This makes it more difficult for molten rock to push up through the crust. Many volcanoes form along convergent plate boundaries where one tectonic plate is pulled down beneath another at a subduction zone. The leading edge of the plate melts as it is pulled into the mantle, forming magma that erupts as volcanoes. When a line of volcanoes forms along a subduction zone, they make up a volcanic arc. The edges of the Pacific plate are long subduction zones lined with volcanoes. This is why the Pacific rim is called the Pacific Ring of Fire."
from summa.summarizer import summarize
import string
import nltk
import re
import random
import benepar
import spacy
def traverse(sent,np,vp):
    end=sent[-1]
    if len(sent.leaves())==1:
        npf=None
        vpf=None
        if np is not None:
            npf=[" ".join([" ".join(leaf.leaves()) for leaf in list(np)])][0]
        if vp is not None:
            vpf=[" ".join([" ".join(leaf.leaves()) for leaf in list(vp)])][0]
        return npf,vpf
    elif end.label()=="NP":
        np=end
    elif end.label()=="VP":
        vp=end
    return traverse(end,np,vp)
def generate_true_false(text):
    nlp = spacy.load('en_core_web_sm')
    #nltk.download('punkt')
    #benepar.download('benepar_en3')
    benepar_parser = benepar.Parser("benepar_en3")
    text_tokens=nltk.tokenize.sent_tokenize(summarize(text,0.35))
    l=[]
    for i in text_tokens:
        string=re.split(r'[:;]+',i)[0]
        if len(string)>35 and len(string)<160:
            l.append(string)
    d={}
    for s in l:
        s=s.rstrip('.,;?:!')
        comp=benepar_parser.parse(s)
        a=None
        b=None
        lnp,lvp=traverse(comp,a,b)
        p=[]
        if lvp is not None:
            vps=None
            cvp=lvp.replace(" ","")
            msl=s.split()
            for i in range(len(msl)):
                cs=("".join(msl[i:])).replace(" ","")
                if cs==cvp:
                    vps=" ".join(msl[:i])
            p.append(vps)
        if lnp is not None:
            nps=None
            cnp=lnp.replace(" ","")
            msl=s.split()
            for i in range(len(msl)):
                cs=("".join(msl[i:])).replace(" ","")
                if cs==cnp:
                    nps=" ".join(msl[:i])
            p.append(nps)
        lp=sorted(p,key=len,reverse=True)
        if len(lp)==2 and (len(lp[0].split())-len(lp[1].split()))>4:
            del lp[1]
        if len(lp)>0:
            d[s]=lp
    tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2",pad_token_id=tokenizer.eos_token_id)
    BERT = SentenceTransformer('bert-base-nli-mean-tokens')
    dq={}
    torch.manual_seed(2023)
    for sent in d:
        for ps in d[sent]:
            ids = torch.tensor([tokenizer.encode(ps)])
            ml = len(s.split())+80
            so=model.generate(ids,do_sample=True,max_length=ml,top_p=0.90,top_k=50,repetition_penalty=10.0,num_return_sequences=10)
            fs=[]
            for i,o in enumerate(so):
                ds=nltk.tokenize.sent_tokenize(tokenizer.decode(o, skip_special_tokens=True))
                fs.append(ds[0])
            se=BERT.encode(fs)
            qe=BERT.encode([sent])
            ds=[]
            for q,qe in zip([sent],qe):
                dist=scipy.spatial.distance.cdist([qe], se, "cosine")[0]
                res=zip(range(len(dist)), dist)
                res=sorted(res, key=lambda x: x[1])
                for idx,dis in reversed(res[0:len(fs)]):
                    val=1-dis
                    if val<0.9:
                        ds.append(fs[idx].strip())
            sds=sorted(ds,key=len)
            sds=sds[:3]
            dq[sent]=sds
    return(dq)
