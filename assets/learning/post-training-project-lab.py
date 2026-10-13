"""CPU teaching checks only: attention, BPE, preference, LoRA, weighting and versions."""
import argparse,json,math
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
D=torch.float64

def multi_head_attention(x,wq,wk,wv,wo,heads):
    b,l,d=x.shape
    assert d%heads==0
    def split(w):return F.linear(x,w).reshape(b,l,heads,d//heads).transpose(1,2)
    q,k,v=split(wq),split(wk),split(wv)
    score=q@k.transpose(-1,-2)/math.sqrt(d//heads)
    future=torch.ones(l,l,dtype=torch.bool).triu(1)
    probs=score.masked_fill(future,float('-inf')).softmax(-1)
    return F.linear((probs@v).transpose(1,2).reshape(b,l,d),wo)

def bpe_merge(tokens,pair):
    out=[];i=0
    while i<len(tokens):
        if i+1<len(tokens) and (tokens[i],tokens[i+1])==pair:
            out.append(tokens[i]+tokens[i+1]);i+=2
        else:out.append(tokens[i]);i+=1
    return out

def run():
    torch.manual_seed(4)
    x=torch.randn(1,4,8,dtype=D);ws=[torch.randn(8,8,dtype=D) for _ in range(4)]
    a=multi_head_attention(x,*ws,heads=2);changed=x.clone();changed[:,3]+=20
    b=multi_head_attention(changed,*ws,heads=2)
    causal=(a[:,:3]-b[:,:3]).abs().max().item();assert causal<1e-12
    base=torch.randn(5,7,dtype=D);A=(torch.randn(2,7,dtype=D)*.02).requires_grad_();B=torch.zeros(5,2,dtype=D,requires_grad=True);inp=torch.randn(3,7,dtype=D);s=4.
    y=F.linear(inp,base)+s*F.linear(F.linear(inp,A),B)
    assert torch.allclose(y,F.linear(inp,base))
    y.square().mean().backward();assert A.grad.abs().max()==0 and B.grad.norm()>0
    first_A=A.grad.norm().item();first_B=B.grad.norm().item()
    with torch.no_grad():B-=.1*B.grad
    split=F.linear(inp,base)+s*F.linear(F.linear(inp,A),B);merged=F.linear(inp,base+s*B@A);err=(split-merged).abs().max().item();assert err<1e-12
    chosen=torch.tensor(-3.,dtype=D,requires_grad=True);rejected=torch.tensor(-5.,dtype=D,requires_grad=True);beta=.2
    margin=(chosen-(-3.))-(rejected-(-5.));loss=-F.logsigmoid(beta*margin);loss.backward()
    assert abs(loss.item()-math.log(2))<1e-12 and abs(chosen.grad.item()+beta/2)<1e-12
    def grad(weights):
        theta=torch.tensor(.4,dtype=D,requires_grad=True);ys=[torch.tensor([1.,2.],dtype=D),torch.tensor([0.,1.,2.,3.,4.,5.],dtype=D)]
        num=sum(w*(theta-y).square().sum() for w,y in zip(weights,ys));den=sum(w*len(y) for w,y in zip(weights,ys));(num/den).backward();return theta.grad.item()
    assert abs(grad([1.5,.5])-grad([3.,1.]))<1e-12
    lengths=[2,6];weights=[1.5,.5];den=sum(n*w for n,w in zip(lengths,weights));effective=[w*sum(lengths)/den for w in weights]
    assert abs(sum(n*w for n,w in zip(lengths,effective))/sum(lengths)-1)<1e-12
    single1=grad([1.,1.]);assert math.isfinite(single1)
    logp=torch.tensor(-2.,dtype=D,requires_grad=True);anchor=logp.detach();adv=torch.tensor(.7,dtype=D);ratio=(logp-anchor).exp();(-ratio*adv).backward();assert ratio.item()==1 and abs(logp.grad.item()+.7)<1e-12
    assert bpe_merge(['a','a','a'],('a','a'))==['aa','a']
    # Same-critic requirement: model drift alone may make a false state delta.
    before_same,after_same=.4,.45;before_old,after_new=.4,.65
    return {'validation_scope':'CPU toy mathematics only; not real-tokenizer, multi-GPU, environment or model benchmark',
        'causal_attention_future_difference':causal,'lora_initial_A_gradient':first_A,'lora_initial_B_gradient':first_B,'lora_merge_error':err,
        'DPO_initial_loss':loss.item(),'DPO_chosen_gradient':chosen.grad.item(),'DPO_rejected_gradient':rejected.grad.item(),
        'weighted_window_gradient':grad(weights),'scaled_weights_gradient':grad([3.,1.]),'effective_weights':effective,
        'single_forward_ratio':ratio.item(),'single_forward_logprob_gradient':logp.grad.item(),
        'BPE_nonoverlap_example':bpe_merge(['a','a','a'],('a','a')),'same_version_value_delta':after_same-before_same,'mixed_version_false_delta':after_new-before_old}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);args=p.parse_args();result=run();text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text,end='')
