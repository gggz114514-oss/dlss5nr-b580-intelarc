"""ctypes callback adapter. Pixel data never traverses a Python/CPU buffer."""
import ctypes as C
import json
import traceback
from pathlib import Path

P=C.c_void_p
Q=C.c_uint64
U=C.c_uint32
PROCESS=C.CFUNCTYPE(C.c_int,P,P,P,P,P,P,Q,U,U,Q,Q,U,C.POINTER(P),C.POINTER(P),C.POINTER(Q))
RETIRE=C.CFUNCTYPE(C.c_int,P,P,Q)
class Callbacks(C.Structure):
    _fields_=[('user',P),('process',PROCESS),('retire',RETIRE),('shader',C.c_char_p)]

class NativeCallback:
    def __init__(self,session,bridge_dll,shader):
        self.session=session
        self.bridge_dll=bridge_dll
        self.bridge=None
        self.error=None
        self.frames=[]
        self.retired=0
        self.shader=str(Path(shader).resolve()).encode('utf-8')
        self.process_callback=PROCESS(self._process)
        self.retire_callback=RETIRE(self._retire)
        self.callbacks=Callbacks(None,self.process_callback,self.retire_callback,self.shader)

    def _process(self,user,device,queue,color,motion,fence,value,w,h,frame_id,previous,reset,out,ready,ready_value):
        if self.error:return 1
        try:
            from nr_texture_bridge_v1 import TextureBridge,TextureNR,SourceFrame
            if self.bridge is None:
                self.bridge=TextureBridge(self.bridge_dll,w,h,native_device=device,native_queue=queue)
                self.adapter=TextureNR(self.bridge,self.session)
            result=self.adapter.process(SourceFrame(color,motion,w,h,frame_id,previous,bool(reset),fence,value))
            out[0]=result.resource;ready[0]=result.fence;ready_value[0]=result.value
            self.frames.append(dict(frame_id=frame_id,previous_id=previous,reset=bool(reset),width=w,height=h))
            return 0
        except BaseException:
            self.error=traceback.format_exc()
            print(self.error,flush=True)
            return 1

    def _retire(self,user,fence,value):
        if self.error:return 1
        try:
            self.bridge.retire(fence,value)
            self.retired+=1
            return 0
        except BaseException:
            self.error=traceback.format_exc()
            print(self.error,flush=True)
            return 1

    def run(self,dll_path,args,*,single=False):
        dll=C.CDLL(str(Path(dll_path).resolve()))
        run=dll.nr_worker_run_single_v1 if single else dll.nr_worker_run_v1
        run.argtypes=[C.c_int,C.POINTER(C.c_char_p),C.POINTER(Callbacks)]
        run.restype=C.c_int
        argv=(C.c_char_p*len(args))(*(arg.encode('utf-8') for arg in args))
        rc=run(len(args),argv,C.byref(self.callbacks))
        if self.error:raise RuntimeError(self.error)
        if rc:raise RuntimeError(f'Native NR worker failed: {rc}')
        if not self.frames or self.retired!=len(self.frames):raise RuntimeError('Incomplete NR consumption')
        return dict(passed=True,nr_frames=len(self.frames),retired=self.retired,frames=self.frames,
                    cpu_pixel_transport_bytes=0,motion_recomputed_by_nr=False)

    def close(self):
        if self.bridge:self.bridge.close()
        self.session.close()
