"""Scoped scheduling ablation: keep 1024-window groups and original arithmetic.

Optional group fences are redundant for in-order dependencies on this torch XPU
stream. Model stage fences and final caller synchronization remain untouched.
Validate full bytes and peak allocation before promoting any scheduling policy.
"""
from contextlib import contextmanager
import torch
import nr_backend.attention as attention
import nr_backend.multihead_block as multihead

class SwinScheduling:
    def __init__(self,*,enabled=False,sync_every_groups=0):
        if type(sync_every_groups) is not int or sync_every_groups<0:raise ValueError('Nonnegative integer fence interval required')
        self.enabled=enabled;self.interval=sync_every_groups;self.counts={}
        self.original=attention.swin_attention_windows
        assert multihead.swin_attention_windows is self.original

    def windows(self,query,key,value,bias):
        if not self.enabled or attention.current_arithmetic_backend()!='triton' or query.device.type!='xpu':
            return self.original(query,key,value,bias)
        if query.shape!=key.shape or query.shape!=value.shape or query.shape[-2:]!=(64,32) or bias.shape!=(64,64):
            raise ValueError('Expected matching 64x32 windows and one64x64 head bias')
        shape=query.shape;q=query.reshape(-1,64,32);k=key.reshape(-1,64,32);v=value.reshape(-1,64,32)
        result=torch.empty_like(q)
        for start in range(0,len(q),1024):
            sl=slice(start,start+1024);count=len(q[sl])
            scores=attention.sm89_f16_batched_dot(q[sl],k[sl].transpose(-1,-2),initial=bias.expand(count,64,64))
            weights=attention.normalize_attention_weights(attention.score_exponential(scores))
            result[sl]=attention.sm89_f16_batched_dot(weights,v[sl])
            self.counts['groups']=self.counts.get('groups',0)+1
            if self.interval and (start//1024+1)%self.interval==0:
                torch.xpu.synchronize(query.device)
                self.counts['group_fences']=self.counts.get('group_fences',0)+1
        self.counts['calls']=self.counts.get('calls',0)+1
        return result.reshape(shape)

    @contextmanager
    def installed(self):
        self.counts={}
        assert attention.swin_attention_windows is self.original and multihead.swin_attention_windows is self.original
        replacement=self.windows
        attention.swin_attention_windows=replacement;multihead.swin_attention_windows=replacement
        try:yield self
        finally:
            assert attention.swin_attention_windows is replacement and multihead.swin_attention_windows is replacement
            attention.swin_attention_windows=self.original;multihead.swin_attention_windows=self.original
