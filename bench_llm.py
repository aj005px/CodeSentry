from review_core import get_llm_reviewer
import time, os

code=open('samples/vulnerable.py').read()
r=get_llm_reviewer()
start=time.time()
r.load()
load_t=time.time()-start
print('load_t', load_t)
t0=time.time()
out=r.review(code)
t1=time.time()
gen_t=t1-t0
print('gen_t', gen_t)
print('chars', len(out))
