#include "nr_core.h"
#include "nr_diag.h"
#include "nr_contract.h"
#include "nr_identity.h"
#include <d3d12.h>
#include <nvsdk_ngx.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#define CHECK(x) do { if (!(x)) { std::cerr << "CHECK failed: " #x << "\n"; std::abort(); } } while(false)

class Parameters final : public NVSDK_NGX_Parameter {
    std::map<std::string, unsigned int> integers;
    std::map<std::string, float> floats;
    std::map<std::string, ID3D12Resource*> resources;
public:
    void Set(const char* k, unsigned long long v) override { integers[k] = unsigned(v); }
    void Set(const char* k, float v) override { floats[k] = v; }
    void Set(const char* k, double v) override { floats[k] = float(v); }
    void Set(const char* k, unsigned int v) override { integers[k] = v; }
    void Set(const char* k, int v) override { integers[k] = unsigned(v); }
    void Set(const char*, ID3D11Resource*) override {}
    void Set(const char* k, ID3D12Resource* v) override { resources[k] = v; }
    void Set(const char* k, void* v) override { resources[k] = static_cast<ID3D12Resource*>(v); }
    NVSDK_NGX_Result Get(const char* k, unsigned long long* v) const override {
        auto x=integers.find(k); if(x==integers.end()) return NVSDK_NGX_Result_Fail;
        *v=x->second; return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char* k, float* v) const override {
        auto x=floats.find(k); if(x==floats.end()) return NVSDK_NGX_Result_Fail;
        *v=x->second; return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char* k, double* v) const override {
        float t; auto r=Get(k,&t); if(r==NVSDK_NGX_Result_Success)*v=t; return r;
    }
    NVSDK_NGX_Result Get(const char* k, unsigned int* v) const override {
        auto x=integers.find(k); if(x==integers.end()) return NVSDK_NGX_Result_Fail;
        *v=x->second; return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char* k, int* v) const override {
        unsigned int t; auto r=Get(k,&t); if(r==NVSDK_NGX_Result_Success)*v=int(t); return r;
    }
    NVSDK_NGX_Result Get(const char*, ID3D11Resource**) const override {return NVSDK_NGX_Result_Fail;}
    NVSDK_NGX_Result Get(const char* k, ID3D12Resource** v) const override {
        auto x=resources.find(k); if(x==resources.end())return NVSDK_NGX_Result_Fail;
        *v=x->second; return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char* k, void** v) const override {
        auto x=resources.find(k); if(x==resources.end())return NVSDK_NGX_Result_Fail;
        *v=x->second; return NVSDK_NGX_Result_Success;
    }
    void Reset() override {integers.clear();floats.clear();resources.clear();}
};

int main() {
    CHECK(sizeof(NRB_StageTimes)==80);
    nrb::DiagnosticGate gate;
    for(auto route:{NRB_ROUTE_DLSS,NRB_ROUTE_FSR,NRB_ROUTE_XESS}) {
        for(uint32_t i=1;i<=nrb::kDiagnosticFramesPerRoute;i++) CHECK(gate.claim(route)==i);
        CHECK(gate.claim(route)==0);
    }
    NRB_Controls controls{sizeof(NRB_Controls), 1, 360, 0, NRB_HISTORY_FUSED,
        1, 0, 0, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f};
    CHECK(nrb::validate_controls(controls));
    controls.input_height=480; controls.style=1;
    CHECK(nrb::validate_controls(controls));
    controls.input_height=540; controls.style=2;
    CHECK(nrb::validate_controls(controls));
    controls.input_height=720;
    CHECK(nrb::validate_controls(controls));
    controls.input_height=721;
    CHECK(!nrb::validate_controls(controls));
    controls.input_height=720;
    controls.model_intensity=2.1f;
    CHECK(!nrb::validate_controls(controls));
    Parameters p;
    auto* color=reinterpret_cast<ID3D12Resource*>(0x1000);
    auto* motion=reinterpret_cast<ID3D12Resource*>(0x2000);
    auto* depth=reinterpret_cast<ID3D12Resource*>(0x3000);
    auto* output=reinterpret_cast<ID3D12Resource*>(0x4000);
    auto* list=reinterpret_cast<ID3D12GraphicsCommandList*>(0x5000);
    p.Set(NVSDK_NGX_Parameter_Color,color);
    p.Set(NVSDK_NGX_Parameter_MotionVectors,motion);
    p.Set(NVSDK_NGX_Parameter_Depth,depth);
    p.Set(NVSDK_NGX_Parameter_Output,output);
    p.Set(NVSDK_NGX_Parameter_Width,1280u);
    p.Set(NVSDK_NGX_Parameter_Height,720u);
    p.Set(NVSDK_NGX_Parameter_OutWidth,1920u);
    p.Set(NVSDK_NGX_Parameter_OutHeight,1080u);
    p.Set(NVSDK_NGX_Parameter_Jitter_Offset_X,-0.25f);
    p.Set(NVSDK_NGX_Parameter_Jitter_Offset_Y,0.5f);
    p.Set(NVSDK_NGX_Parameter_MV_Scale_X,1280.0f);
    p.Set(NVSDK_NGX_Parameter_MV_Scale_Y,-720.0f);
    p.Set(NVSDK_NGX_Parameter_Reset,1);
    p.Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure,0.75f);
    p.Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale,1.25f);
    p.Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
          int(NVSDK_NGX_DLSS_Feature_Flags_MVLowRes|NVSDK_NGX_DLSS_Feature_Flags_MVJittered));
    for(auto route:{NRB_ROUTE_DLSS,NRB_ROUTE_FSR,NRB_ROUTE_XESS}) {
        NRB_Frame f{};
        CHECK(nrb::extract_ngx(&p,route,list,f));
        CHECK(f.route==route && f.color==color && f.motion==motion &&
               f.depth==depth && f.output==output && f.render_width==1280 &&
               f.output_height==1080 && f.jitter_x==-0.25f &&
               f.motion_scale_y==-720.0f && f.reset_history &&
               f.motion_scale_origin==static_cast<NRB_MotionScaleOrigin>(route) &&
               f.low_resolution_motion && f.motion_jittered &&
               f.pre_exposure==0.75f && f.exposure_scale==1.25f);
        CHECK(nrb::validate_metadata(f)==NRB_OK);
    }
    p.Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,960u);
    p.Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height,540u);
    p.Set("FSR.upscaleSize.width",2560u);
    p.Set("FSR.upscaleSize.height",1440u);
    NRB_Frame dynamic{};
    nrb::extract_ngx(&p,NRB_ROUTE_FSR,list,dynamic);
    CHECK(dynamic.render_width==960 && dynamic.render_height==540 &&
           dynamic.output_width==2560 && dynamic.output_height==1440);
    NRB_Frame switched{};
    nrb::extract_ngx(&p,NRB_ROUTE_DLSS,list,switched);
    CHECK(switched.output_width==1920 && switched.output_height==1080);
    p.Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,int(NVSDK_NGX_DLSS_Feature_Flags_IsHDR));
    nrb::extract_ngx(&p,NRB_ROUTE_XESS,list,dynamic);
    CHECK(nrb::validate_metadata(dynamic)==NRB_OK);
    NRB_Frame cyberpunk=dynamic;
    cyberpunk.render_width=1280;cyberpunk.render_height=720;
    cyberpunk.motion_scale_x=1280;cyberpunk.motion_scale_y=720;
    cyberpunk.low_resolution_motion=1;
    cyberpunk.pre_exposure=cyberpunk.exposure_scale=1;
    const auto work=nrb::plan_re8_adaptations(cyberpunk,
        {DXGI_FORMAT_R11G11B10_FLOAT,DXGI_FORMAT_R16G16_FLOAT,1280,720,1280,720});
    CHECK((work & (nrb::RE8_COLOR_FORMAT|nrb::RE8_HDR_POLICY|
                   nrb::RE8_MOTION_FORMAT_OR_SCALE)) ==
          (nrb::RE8_COLOR_FORMAT|nrb::RE8_HDR_POLICY|
           nrb::RE8_MOTION_FORMAT_OR_SCALE));
    NRB_Frame re8=cyberpunk;
    re8.hdr_input=0;re8.motion_scale_x=re8.motion_scale_y=1;
    CHECK(nrb::plan_re8_adaptations(re8,
        {DXGI_FORMAT_B8G8R8A8_UNORM,DXGI_FORMAT_R16G16_FLOAT,1280,720,1280,720})==nrb::RE8_DIRECT);
    D3D12_RESOURCE_DESC packed{};
    packed.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    packed.Width=1280;packed.Height=720;packed.DepthOrArraySize=1;
    packed.MipLevels=1;packed.Format=DXGI_FORMAT_R11G11B10_FLOAT;
    packed.SampleDesc.Count=1;
    nrb::IdentityBoundary identity{};
    CHECK((nrb::plan_dlss_identity_copy(packed,packed,identity) &
           (nrb::IDENTITY_STATE_UNPROVEN|nrb::IDENTITY_LIFETIME_UNPROVEN|
            nrb::IDENTITY_DEST_STATE_UNPROVEN|
            nrb::IDENTITY_CURRENT_WRITE_UNPROVEN)) != 0);
    identity.current_source_state_proven=true;
    identity.destination_lifetime_proven=true;
    identity.destination_copy_dest_state_proven=true;
    identity.current_source_state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    identity.list_reset_generation=7;
    identity.last_color_write_generation=7;
    identity.last_color_write_sequence=20;
    identity.evaluate_entry_sequence=21;
    identity.consumer_retirement_tracking_id=99;
    CHECK(nrb::plan_dlss_identity_copy(packed,packed,identity)==nrb::IDENTITY_READY);
    identity.last_color_write_generation=6;
    CHECK(nrb::plan_dlss_identity_copy(packed,packed,identity)==nrb::IDENTITY_CURRENT_WRITE_UNPROVEN);
    identity.last_color_write_generation=7;
    auto wrong=packed;wrong.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
    CHECK(nrb::plan_dlss_identity_copy(packed,wrong,identity)==nrb::IDENTITY_LAYOUT_MISMATCH);
    p.Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,0);
    nrb::extract_ngx(&p,NRB_ROUTE_XESS,list,dynamic);
    nrb::QueueMeta queue{};
    queue.ordinal=2; queue.type=0; queue.same_device=true;
    auto line=nrb::format_diagnostic(1,dynamic,{87,960,540},{13,960,540},
        {45,960,540},{0,0,0},{87,1920,1080},queue,NRB_BYPASS_UNSYNCED);
    for(auto field:{"\"route\":\"XeSS\"","\"source\":[960,540]",
        "\"output\":[1920,1080]","\"color\":{\"format\":87",
        "\"motion\":{\"format\":13","\"depth\":{\"format\":45",
        "\"mv_scale\":[1280,-720]","\"jitter\":[-0.25,0.5]",
        "\"hdr\":0","\"reset\":1","\"ordinal\":2",
        "\"pre_exposure\":0.75","\"exposure_scale\":1.25",
        "\"motion_jittered\":0","\"low_resolution_motion\":0",
        "\"producer_status\":\"unproven\"",
        "\"bypass\":\"producer_or_consumer_order_unproven\""})
        CHECK(line.find(field)!=std::string::npos);
    std::cout << "three API maps and dynamic sizes passed\n";
}
