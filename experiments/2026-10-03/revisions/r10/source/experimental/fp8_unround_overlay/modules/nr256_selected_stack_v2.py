"""Construct the selected WindowBlocks v3 stack; explicit optional experiment rewrite.

Default behavior includes the already measured window layout. An explicitly
provided rewrite class is an experimental candidate, not a runtime promotion.
"""
from nr256_selected_stack_v1 import Stack as Base
from window_blocks_v3 import WindowBlocks


class Stack(Base):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.window_blocks=WindowBlocks(self.model,self.provider,self.components[2])
        self.components.insert(-1,self.window_blocks)

    def close(self):
        super().close()
        close=getattr(self.rewrite,'close',None)
        if close is not None:close()
