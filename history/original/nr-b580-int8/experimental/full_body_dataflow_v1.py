"""Full-body provenance after redundant-FP8 elimination, with scalar metadata."""
import torch
from fp8_graph_rewrite_v1 import RewritingDataflow,FP8GraphRewrite
from nr_backend.execution import current_arithmetic_backend


def constant(value):
    if isinstance(value,torch.Tensor):return {'tensor_argument':True}
    if value is None or isinstance(value,(str,bool,int,float)):return value
    if isinstance(value,(tuple,list)):return [constant(v) for v in value]
    if isinstance(value,dict):return {str(k):constant(v) for k,v in value.items()}
    if isinstance(value,(torch.dtype,torch.device,torch.memory_format,torch.layout)):return str(value)
    return {'unsupported_type':type(value).__name__}


class DetailedDataflow(RewritingDataflow):
    def __init__(self):
        super().__init__();self.kernel_counts={}

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        before=len(self.events)
        result=super().__torch_dispatch__(func,types,args,kwargs)
        if len(self.events)>before:
            event=self.events[-1]
            assert event['kind']=='aten' and event['name']==str(func)
            event['arguments']=constant(args);event['keyword_arguments']=constant(kwargs or {})
        return result

    def launch(self,kernel,grid,stream,args):
        super().launch(kernel,grid,stream,args)
        event=self.events[-1];name=event['name']
        event['kernel_ordinal']=self.kernel_counts.get(name,0)
        self.kernel_counts[name]=event['kernel_ordinal']+1
        event['constants']={n:constant(v) for n,v in zip(kernel.src.fn.arg_names,args) if not isinstance(v,torch.Tensor)}
        event['launch_options']={n:constant(getattr(kernel.metadata,n,None)) for n in ('num_warps','num_stages','enable_fp_fusion')}


class FullTraceGraph(FP8GraphRewrite):
    def __init__(self,model,provider):
        super().__init__(model,provider);self.records={}

    def apply(self,module,rgb,front,**options):
        if (module is not self.model or self.provider.mode!='fp16_xmx' or current_arithmetic_backend()!='triton'
                or rgb.device.type!='xpu' or tuple(rgb.shape)!=(256,256,3)
                or tuple(front.shape)!=(320,320,16) or options.get('return_float32',False)):
            return self.original(module,rgb,front,**options)
        analysis=DetailedDataflow()
        with analysis.installed():result=self.original(module,rgb,front,**options)
        summary=analysis.rewrite_summary();self.builds.append(summary)
        key='temporal' if options.get('previous') is not None else 'reset'
        if key not in self.records:
            self.records[key]=dict(events=analysis.events,summary=summary,result=analysis.ref(result))
        return result
