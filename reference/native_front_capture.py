"""Layouts derived from the instrumented SM89 producer and HMMA fragments."""
import numpy as np

def decode_front_dump(arena: bytes) -> np.ndarray:
    data=arena[0x16c00:0x16c00+40*40*2048]
    return np.frombuffer(data,dtype='<f2').reshape(40,40,2,64,8).transpose(0,1,3,2,4).reshape(
        40,40,8,8,16).transpose(0,2,1,3,4).reshape(320,320,16).copy()

def decode_projection_dump(arena: bytes) -> np.ndarray:
    data=arena[0x16c00:0x16c00+40*40*4096]
    physical=np.frombuffer(data,dtype='<f2').reshape(40,40,16,2,32,2)
    output=np.full((40,40,64,32),np.nan,dtype=np.float16)
    fragments=[(0,0),(0,1),(0,2),(0,3),(1,1),(1,3),(1,0),(2,1),
               (1,2),(2,0),(2,2),(2,3),(3,1),(3,3),(3,2),(3,0)]
    bases=(0,4,32,36)
    for frag,(a,b) in enumerate(fragments):
        for word in range(2):
            for lane in range(32):
                group,thread=lane//4,lane%4
                pixel=bases[a]+group%4+8*(group//4)+16*word
                for half in range(2):output[:,:,pixel,b*8+2*thread+half]=physical[:,:,frag,word,lane,half]
    return output.reshape(40,40,8,8,32).transpose(0,2,1,3,4).reshape(320,320,32).copy()

def decode_front_weights(blob: bytes) -> np.ndarray:
    output=np.zeros((16,32),dtype=np.float16)
    for tile,offset in enumerate((0x2010,0x2210)):
        lanes=np.frombuffer(blob[offset:offset+512],dtype='<f2').reshape(32,8)
        for lane in range(32):
            group,thread=lane//4,lane%4
            rows=(2*thread,2*thread+1,8+2*thread,9+2*thread)
            for fragment in range(2):
                for i,row in enumerate(rows):output[row,tile*16+fragment*8+group]=lanes[lane,fragment*4+i]
    return output

def decode_mlp1_all_dump(arena: bytes) -> tuple[np.ndarray,np.ndarray]:
    data=arena[0x16c00:0x16c00+40*40*8192]
    physical=np.frombuffer(data,dtype='<f2').reshape(40,40,2,16,2,32,2)
    output=np.empty((2,40,40,16,128),dtype=np.float16)
    for stage in range(2):
        for fragment in range(16):
            for word in range(2):
                for lane in range(32):
                    group,thread=divmod(lane,4)
                    output[stage,:,:,group+8*word,fragment*8+2*thread:fragment*8+2*thread+2]=physical[:,:,stage,fragment,word,lane,:]
    return output[0],output[1]

def select_a0_pixels(tensor: np.ndarray) -> np.ndarray:
    channels=tensor.shape[-1]
    pixels=[g%4+8*(g//4)+16*word for word in range(2) for g in range(8)]
    return tensor.reshape(40,8,40,8,channels).transpose(0,2,1,3,4).reshape(40,40,64,channels)[:,:,pixels,:].copy()

def decode_fp8_k32_n16_tiles(blob: bytes, k: int, n: int) -> np.ndarray:
    """Observed pre weight tiles: K32-major, N16-minor, native packed K permutation."""
    assert k%32==0 and n%16==0 and len(blob)==k*n
    physical=np.frombuffer(blob,dtype='u1').reshape(k//32,n//16,32,16)
    logical=np.zeros((k,n),dtype='u1')
    for kt in range(k//32):
        for nt in range(n//16):
            for lane in range(32):
                g,t=divmod(lane,4)
                for frag in range(2):
                    for reg in range(2):
                        for byte in range(4):
                            row=kt*32+2*t+byte%2+8*(byte//2)+16*reg
                            col=nt*16+g+8*frag
                            logical[row,col]=physical[kt,nt,lane,frag*8+reg*4+byte]
    return logical

def decode_mlp_output_dump(arena: bytes) -> np.ndarray:
    physical=np.frombuffer(arena[0x16c00:0x16c00+40*40*2048],dtype='u1').reshape(40,40,4,4,32,4)
    output=np.empty((40,40,64,32),dtype='u1')
    for a,base in enumerate((0,4,32,36)):
        for reg in range(4):
            for lane in range(32):
                group,thread=divmod(lane,4)
                pixel=base+group%4+8*(group//4)+16*(reg%2)
                for byte in range(4):
                    channel=2*thread+byte%2+8*(byte//2)+16*(reg//2)
                    output[:,:,pixel,channel]=physical[:,:,a,reg,lane,byte]
    return output.reshape(40,40,8,8,32).transpose(0,2,1,3,4).reshape(320,320,32).copy()

def decode_a0_half_fragments(arena: bytes, channels: int) -> np.ndarray:
    assert channels%8==0
    data=arena[0x16c00:0x16c00+40*40*16*channels*2]
    physical=np.frombuffer(data,dtype='<f2').reshape(40,40,channels//8,2,32,2)
    output=np.empty((40,40,16,channels),dtype=np.float16)
    for frag in range(channels//8):
        for word in range(2):
            for lane in range(32):
                g,t=divmod(lane,4)
                output[:,:,g+8*word,frag*8+2*t:frag*8+2*t+2]=physical[:,:,frag,word,lane,:]
    return output

def decode_score_dump(arena: bytes) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    data=arena[0x16c00:0x16c00+40*40*1280]
    words=np.frombuffer(data,dtype='<u4').reshape(40,40,10,32)
    halves=words.view('<f2').reshape(40,40,10,32,2)
    bytes_=words.view('u1').reshape(40,40,10,32,4)
    score=np.empty((40,40,16,8),dtype='f2');bias=np.empty_like(score)
    a=np.empty((40,40,16,32),dtype='u1');b=np.empty((40,40,32,8),dtype='u1')
    for lane in range(32):
        g,t=divmod(lane,4)
        for word in range(2):
            score[:,:,g+8*word,2*t:2*t+2]=halves[:,:,word,lane,:]
            bias[:,:,g+8*word,2*t:2*t+2]=halves[:,:,8+word,lane,:]
        for reg in range(4):
            for byte in range(4):a[:,:,g+8*(reg%2),2*t+byte%2+8*(byte//2)+16*(reg//2)]=bytes_[:,:,2+reg,lane,byte]
        for reg in range(2):
            for byte in range(4):b[:,:,2*t+byte%2+8*(byte//2)+16*reg,g]=bytes_[:,:,6+reg,lane,byte]
    return score,bias,a,b

def decode_c32_quad_bytes(data: bytes, height: int, width: int) -> np.ndarray:
    assert height%4==0 and width%4==0 and len(data)==height*width*32
    physical=np.frombuffer(data,dtype='u1').reshape(height//4,width//4,32,4,4)
    output=np.empty((height//4,width//4,4,4,32),dtype='u1')
    for lane in range(32):
        g,t=divmod(lane,4)
        for word in range(4):
            y,x=g//4+2*(word%2),g%4
            for byte in range(4):
                c=2*t+byte%2+8*(byte//2)+16*(word//2)
                output[:,:,y,x,c]=physical[:,:,lane,word,byte]
    return output.transpose(0,2,1,3,4).reshape(height,width,32).copy()

def encode_c32_quad_bytes(logical: np.ndarray) -> bytes:
    height,width,channels=logical.shape
    assert height%4==0 and width%4==0 and channels==32 and logical.dtype==np.uint8
    quads=logical.reshape(height//4,4,width//4,4,32).transpose(0,2,1,3,4)
    output=np.empty((height//4,width//4,32,4,4),dtype='u1')
    for lane in range(32):
        g,t=divmod(lane,4)
        for word in range(4):
            y,x=g//4+2*(word%2),g%4
            for byte in range(4):
                c=2*t+byte%2+8*(byte//2)+16*(word//2)
                output[:,:,lane,word,byte]=quads[:,:,y,x,c]
    return output.tobytes()

def decode_split_c16_bytes(data: bytes, height: int, width: int, channels: int=32) -> np.ndarray:
    assert channels%16==0 and len(data)==height*width*channels
    physical=np.frombuffer(data,dtype='u1').reshape(channels//16,height,width,4,4)
    output=np.empty((height,width,channels),dtype='u1')
    for c in range(channels//16):
        for t in range(4):
            for b in range(4):output[:,:,c*16+2*t+b%2+8*(b//2)]=physical[c,:,:,t,b]
    return output

def encode_split_c16_bytes(logical: np.ndarray) -> bytes:
    height,width,channels=logical.shape
    assert channels%16==0 and logical.dtype==np.uint8
    output=np.empty((channels//16,height,width,4,4),dtype='u1')
    for c in range(channels//16):
        for t in range(4):
            for b in range(4):output[c,:,:,t,b]=logical[:,:,c*16+2*t+b%2+8*(b//2)]
    return output.tobytes()


def encode_multihead_quad_bytes(logical: np.ndarray) -> bytes:
    height,width,channels=logical.shape
    assert channels%32==0
    planes=[np.frombuffer(encode_c32_quad_bytes(logical[:,:,i*32:(i+1)*32]),dtype='u1').reshape(height//4,width//4,512)for i in range(channels//32)]
    return np.stack(planes,axis=2).tobytes()


def decode_multihead_quad_bytes(data: bytes,height: int,width: int,channels: int) -> np.ndarray:
    assert channels%32==0 and len(data)==height*width*channels
    physical=np.frombuffer(data,dtype='u1').reshape(height//4,width//4,channels//32,512)
    return np.concatenate([decode_c32_quad_bytes(physical[:,:,i].tobytes(),height,width)for i in range(channels//32)],axis=-1)


def encode_vit_1d_bytes(logical: np.ndarray) -> bytes:
    tokens,channels=logical.shape
    assert tokens%16==0
    return encode_multihead_quad_bytes(logical.reshape(tokens//4,4,channels))


def decode_vit_1d_bytes(data: bytes,tokens: int,channels: int) -> np.ndarray:
    assert tokens%16==0
    return decode_multihead_quad_bytes(data,tokens//4,4,channels).reshape(tokens,channels)


def encode_vit_key_bytes(logical: np.ndarray) -> bytes:
    tokens,heads,channels=logical.shape
    assert channels==32 and tokens%16==0
    data=np.frombuffer(encode_vit_1d_bytes(logical.reshape(tokens,heads*32)),dtype='u1')
    return data.reshape(tokens//16,heads,32,2,2,4).transpose(0,1,2,4,3,5).tobytes()


def encode_vit_value_bytes(logical: np.ndarray) -> bytes:
    tokens,heads,channels=logical.shape
    assert channels==32 and tokens%32==0 and logical.dtype==np.uint8
    physical=np.empty((tokens//32,heads,2,32,16),dtype='u1')
    for tile in range(tokens//32):
        for nt in range(2):
            for lane in range(32):
                g,t=divmod(lane,4)
                for frag in range(2):
                    for reg in range(2):
                        for byte in range(4):physical[tile,:,nt,lane,frag*8+reg*4+byte]=logical[tile*32+2*t+byte%2+8*(byte//2)+16*reg,:,nt*16+g+8*frag]
    return physical.tobytes()
