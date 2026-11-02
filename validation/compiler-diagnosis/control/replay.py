import argparse, json, pathlib, time
import torch
from harness.model_gpt import GPT

p=argparse.ArgumentParser();p.add_argument('--backend',default='inductor');p.add_argument('--rank',type=int,default=0);p.add_argument('--microbatch',type=int,default=-1);p.add_argument('--tag',default=None);p.add_argument('--variant',default='default');args=p.parse_args()
root=pathlib.Path('/workspace/diagnosis')
tag=args.tag or f'{args.backend}-rank{args.rank}-{args.variant}'
if args.variant=='capture_mix':
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    original_run=CachingAutotuner.run
    capture_index=[0]
    def capture_run(kernel,*values,**kwargs):
        if not kernel.is_mix_order_reduction:
            return original_run(kernel,*values,**kwargs)
        idx=capture_index[0];capture_index[0]+=1
        label=f'mix-{idx:03d}'
        before=[v.cpu().clone() if isinstance(v,torch.Tensor) else v for v in values]
        result=original_run(kernel,*values,**kwargs)
        after=[v.cpu().clone() if isinstance(v,torch.Tensor) else v for v in values]
        torch.save(dict(before=before,after=after),root/f'{label}.pt')
        (root/f'{label}.py').write_text(kernel.fn.src)
        meta=dict(name=kernel.inductor_meta.get('kernel_name'),meta=kernel.inductor_meta,args=[dict(shape=list(v.shape),stride=list(v.stride()),dtype=str(v.dtype),finite=bool(v.isfinite().all())) if isinstance(v,torch.Tensor) else v for v in values])
        (root/f'{label}.json').write_text(json.dumps(meta,indent=2,default=str))
        print('CAPTURED',label,meta['name'],flush=True)
        return result
    CachingAutotuner.run=capture_run

model=GPT(vocab_size=50304,num_layers=12,model_dim=768).cuda()
model.load_state_dict(torch.load(root/'model.pt',weights_only=True,map_location='cpu',mmap=True))
if args.variant=='no_mix_order':
    import torch._inductor.config as config
    config.triton.mix_order_reduction=False
elif args.variant=='no_memory_planning':
    import torch._inductor.config as config
    config.allow_buffer_reuse=False
    config.inplace_buffers=False
elif args.variant=='no_fusion':
    import torch._inductor.config as config
    config.max_fusion_size=1
elif args.variant=='disable_proj':
    model.blocks[0].attn.proj.forward=torch.compiler.disable(model.blocks[0].attn.proj.forward)
elif args.variant=='anomaly':
    torch.autograd.set_detect_anomaly(True)
if args.backend!='none':
    model.compile(backend=args.backend,dynamic=False)
inputs,targets=[x.cuda() for x in torch.load(root/f'batch-rank{args.rank}.pt',weights_only=True)]
rows=[]
indices=range(len(inputs)//8) if args.microbatch<0 else [args.microbatch]
for i in indices:
    started=time.time();sl=slice(i*8,(i+1)*8)
    loss=model(inputs[sl],targets[sl]);loss.backward()
    named=list(model.named_parameters())
    flags=torch.stack([torch.isfinite(x.grad).all() for _,x in named]).tolist()
    bias=model.blocks[0].attn.proj.bias.grad
    row=dict(microbatch=i,loss=float(loss),seconds=time.time()-started,nonfinite=[n for (n,_),f in zip(named,flags) if not f],bias_nan=int(bias.isnan().sum()),bias_inf=int(bias.isinf().sum()),bias_max=float(bias.abs().max()))
    rows.append(row);print(tag,json.dumps(row),flush=True)
    (root/f'{tag}.json').write_text(json.dumps(dict(settings=vars(args),rows=rows),indent=2))
    torch.save(bias.cpu(),root/f'{tag}-bias.pt')

torch.save({n:p.grad.cpu() for n,p in model.named_parameters()},root/f'{tag}-gradients.pt')
