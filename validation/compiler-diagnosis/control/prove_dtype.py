import json,pathlib,torch
root=pathlib.Path('/workspace/diagnosis')
p=torch.load(root/'mix-024.pt',weights_only=True,mmap=True)
workspace=p['after'][11]
partial_sum=workspace.cuda().reshape(3,512,768)[0].sum(0).cpu()
bias=torch.load(root/'captured-bias.pt',weights_only=True)
misread=partial_sum.view(torch.bfloat16)[:768].float()
source=p['before'][2].float().reshape(8192,768).sum(0)
def same(x,y):return bool(((x==y)|(x.isnan()&y.isnan())).all())
result=dict(workspace_dtype=str(workspace.dtype),sum_dtype=str(partial_sum.dtype),actual_bias_dtype=str(bias.dtype),all_inputs_finite=all(bool(x.isfinite().all()) for x in p['before'][:10]),workspace_sum_finite=bool(partial_sum.isfinite().all()),source_sum_max=float(source.abs().max()),workspace_sum_max=float(partial_sum.abs().max()),workspace_vs_source_max_error=float((source-partial_sum).abs().max()),bias_matches_wrong_bf16_pointer_read=same(bias,misread),bias_max=float(bias.abs().max()),bad_dtype_values=misread[:12].tolist(),correct_values=partial_sum[:12].tolist(),actual_values=bias[:12].tolist())
(root/'dtype-proof.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
torch.save(dict(source_gradient=p['before'][2],workspace_sum=partial_sum,actual_bias=bias),root/'dtype-proof.pt')
