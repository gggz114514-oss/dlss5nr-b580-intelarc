"""480x864 adapter of the frozen residual filters; no crop of model input.

Only source geometry/filter tables and motion unit scales change. The existing
separable kernels and NR256 graph are reused. 142 active rows approximate the
source aspect ratio within 0.157%; output remains the original 864x480.
"""
import torch
import triton
from residual_scale_v1 import ResidualScale,filter_table,_pad


class Face480Scale(ResidualScale):
    def __init__(self):
        self.size,self.active_h,self.top,self.device=256,142,57,'xpu'
        self.tables={}
        for kind,axis,src,dst in [('lanczos2','y',480,142),('lanczos2','x',864,256),
                                  ('area','y',480,142),('area','x',864,256),
                                  ('catmull','x',256,864),('catmull','y',142,480)]:
            indices,weights=filter_table(src,dst,kind)
            self.tables[kind,axis]=torch.from_numpy(indices).to('xpu'),torch.from_numpy(weights).to('xpu')

    def prepare(self,rgb,motion):
        if rgb.shape!=(480,864,3) or motion.shape!=(480,864,2) or rgb.device!=motion.device:
            raise ValueError('Expected complete 480x864 RGB and pixel motion')
        rgb,motion=rgb.contiguous(),motion.contiguous()
        color=self.axis(self.axis(rgb,'lanczos2','y'),'lanczos2','x')
        flow=self.axis(self.axis(motion,'area','y',dtype=torch.float32),'area','x',dtype=torch.float32)
        canvas=torch.empty((256,256,3),dtype=torch.float32,device=rgb.device)
        low_motion=torch.empty((256,256,2),dtype=torch.float16,device=rgb.device)
        for source,out,channels,is_motion in [(color,canvas,3,False),(flow,low_motion,2,True)]:
            _pad[(triton.cdiv(out.numel(),256),)](source,out,256,142,57,channels,is_motion,256/864,142/480,256,enable_fp_fusion=False)
        return canvas,low_motion

    def metadata(self):
        return dict(super().metadata(),source=[480,864],active_rgb=[142,256],padding_top=57,
                    aspect_preserved='integer rounding of active height; relative ratio error -0.15625%',
                    display_face_crop_only=True,model_receives_complete_frame=True)
