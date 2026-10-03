#include <d3dcompiler.h>
#include <cstdio>
#include <initializer_list>

int wmain(int argc,wchar_t** argv) {
    if(argc!=2) return 2;
    for(const char* entry:{"prepare","composite"}) {
        ID3DBlob *code=nullptr,*errors=nullptr;
        const HRESULT hr=D3DCompileFromFile(argv[1],nullptr,
            D3D_COMPILE_STANDARD_FILE_INCLUDE,entry,"cs_5_0",
            D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_IEEE_STRICTNESS,
            0,&code,&errors);
        if(FAILED(hr)) {
            if(errors) std::fprintf(stderr,"%s: %s\n",entry,
                static_cast<const char*>(errors->GetBufferPointer()));
            else std::fprintf(stderr,"%s: HRESULT=0x%08x\n",entry,
                static_cast<unsigned>(hr));
            if(code) code->Release();
            if(errors) errors->Release();
            return 1;
        }
        std::printf("%s: %zu bytes\n",entry,code->GetBufferSize());
        code->Release();
        if(errors) errors->Release();
    }
    return 0;
}
