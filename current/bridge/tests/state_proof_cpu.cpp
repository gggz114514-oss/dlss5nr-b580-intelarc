#include "nr_sync_probe.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>

#define CHECK(x) do {if(!(x)) {std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";std::exit(1);}} while(false)
using Reason=nrb::ColorProofReason;
constexpr auto psr=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
constexpr auto uav=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
constexpr auto copy_dest=D3D12_RESOURCE_STATE_COPY_DEST;
struct Object {uint64_t cookie=0;bool candidate=true,known=true;int pins=0;};
struct Identities {
    int live=0,peak=0,calls=0;
    static nrb::ProofAcquire acquire(void* context,void* value,bool resource,nrb::ProofIdentity& out) noexcept {
        auto& self=*static_cast<Identities*>(context);++self.calls;
        auto& obj=*static_cast<Object*>(value);
        if(!obj.known) return nrb::ProofAcquire::unknown;
        if(resource && !obj.candidate) return nrb::ProofAcquire::not_candidate;
        ++obj.pins;++self.live;self.peak=std::max(self.peak,self.live);
        out={obj.cookie,&obj};return nrb::ProofAcquire::acquired;
    }
    static void release(void* context,nrb::ProofIdentity& identity) noexcept {
        auto& self=*static_cast<Identities*>(context);
        auto& obj=*static_cast<Object*>(identity.lease);--obj.pins;--self.live;
        identity={};
    }
    nrb::ProofIdentityOps ops() {return {this,acquire,release};}
};
struct Fixture {
    Identities ids{};
    std::array<Object,80> lists{};
    std::array<Object,8> resources{};
    Fixture() {
        uint64_t cookie=1;
        for(auto& o:lists) o.cookie=cookie++;
        for(auto& o:resources) o.cookie=cookie++;
    }
    ~Fixture() {CHECK(ids.live==0);}
    ID3D12GraphicsCommandList* list(size_t i=0) {return reinterpret_cast<ID3D12GraphicsCommandList*>(&lists[i]);}
    ID3D12Resource* resource(size_t i=0) {return reinterpret_cast<ID3D12Resource*>(&resources[i]);}
    D3D12_RESOURCE_BARRIER transition(size_t r=0,D3D12_RESOURCE_STATES before=copy_dest,
            D3D12_RESOURCE_STATES after=psr,UINT sub=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            D3D12_RESOURCE_BARRIER_FLAGS flags=D3D12_RESOURCE_BARRIER_FLAG_NONE) {
        D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;b.Flags=flags;
        b.Transition={resource(r),sub,before,after};return b;
    }
    void barrier(nrb::SyncProbe& p,size_t l=0,size_t r=0,
            D3D12_RESOURCE_STATES before=copy_dest,D3D12_RESOURCE_STATES after=psr) {
        const auto b=transition(r,before,after);p.barriers(list(l),1,&b);
    }
    nrb::SyncObservation observe(nrb::SyncProbe& p,size_t l=0,size_t r=0,bool dlss=true) {
        return p.evaluate(list(l),resource(r),resource(7),dlss);
    }
    void begin(nrb::SyncProbe& p) {p.reset(list(),true);barrier(p);}
};
bool allowed(const nrb::SyncObservation& s) {return nrb::dlss_color_source(s,true);}
void rejected(const nrb::SyncObservation& s,Reason why) {
    CHECK(s.color_state.enabled);CHECK(!allowed(s));CHECK(s.color_state.reason==why);
}
void disabled_contract() {
    Fixture f;nrb::SyncProbe p(f.ids.ops(),false);f.begin(p);
    CHECK(allowed(f.observe(p)));CHECK(f.ids.calls==0);
    for(int i=0;i<64;++i) f.observe(p);
    CHECK(!allowed(f.observe(p)));CHECK(f.ids.calls==0);
    p.unknown_recording(f.list());CHECK(f.ids.calls==0);
    CHECK(nrb::dlss_color_state_scope("legacy-direct-reviewed-v1"));
    CHECK(!nrb::dlss_color_state_scope(""));CHECK(!nrb::dlss_color_state_scope("1"));
    CHECK(!nrb::dlss_color_state_scope("legacy-direct-reviewed-v1 "));
    D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Format=DXGI_FORMAT_R11G11B10_FLOAT;d.Width=1280;d.Height=720;
    d.DepthOrArraySize=1;d.MipLevels=1;d.SampleDesc.Count=1;
    CHECK(nrb::dlss_color_candidate(d));d.MipLevels=2;CHECK(!nrb::dlss_color_candidate(d));
    d.MipLevels=1;d.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;CHECK(!nrb::dlss_color_candidate(d));
    d.Format=DXGI_FORMAT_R11G11B10_FLOAT;d.Width=256;CHECK(!nrb::dlss_color_candidate(d));
}
void normal_launch_profile_and_override() {
    using Mode=nrb::ColorStateScopeMode;
    constexpr auto cp=L"G:/epic/Cyberpunk2077/bin/x64/Cyberpunk2077.exe";
    CHECK(nrb::dlss_color_select_scope(false,"",cp)==Mode::reviewed);
    CHECK(nrb::dlss_color_select_scope(false,"",L"C:\\GAME\\CYBERPUNK2077.EXE")==Mode::reviewed);
    CHECK(nrb::dlss_color_select_scope(false,"",L"re8.exe")==Mode::disabled);
    CHECK(nrb::dlss_color_select_scope(false,"",L"")==Mode::disabled);
    CHECK(!nrb::dlss_color_default_profile(L"Cyberpunk2077.exe.bak"));
    CHECK(!nrb::dlss_color_default_profile(L"Cyberpunk2077.exe/re8.exe"));
    for(const auto value:{"", "0", "disabled", "unexpected", "legacy-direct-reviewed-v1 "})
        CHECK(nrb::dlss_color_select_scope(true,value,cp)==Mode::disabled);
    CHECK(nrb::dlss_color_select_scope(true,nrb::kDlssColorStateShadowScope,cp)==Mode::shadow);
    CHECK(nrb::dlss_color_select_scope(true,nrb::kDlssColorStateScope,cp)==Mode::reviewed);
    // Explicit reviewed opt-in for other executables retains the old opt-in contract.
    CHECK(nrb::dlss_color_select_scope(true,nrb::kDlssColorStateScope,L"re8.exe")==Mode::reviewed);
}
void shadow_gate_contract() {
    CHECK(nrb::dlss_color_state_scope("legacy-direct-shadow-v1"));
    CHECK(nrb::dlss_color_state_shadow_scope("legacy-direct-shadow-v1"));
    CHECK(!nrb::dlss_color_state_shadow_scope("legacy-direct-reviewed-v1"));
    CHECK(!nrb::dlss_color_state_shadow_scope("legacy-direct-shadow-v1 "));
    Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
    p.unknown_recording(f.list());auto s=f.observe(p);
    rejected(s,Reason::implementation);
    CHECK(nrb::legacy_dlss_color_source(s));
    CHECK(nrb::dlss_color_source(s,false,true));
    s.color.after=uav;CHECK(!nrb::dlss_color_source(s,true,true));
    s.color.after=psr;s.closed=true;CHECK(!nrb::dlss_color_source(s,true,true));
    s.closed=false;s.evaluate_sequence=s.color.sequence+65;
    CHECK(!nrb::dlss_color_source(s,true,true));
}
void processing_gate_extension() {
    Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
    p.unknown_recording(f.list());auto s=f.observe(p);
    CHECK(!allowed(s));CHECK(nrb::legacy_dlss_color_source(s));
    CHECK(nrb::dlss_color_processing_source(s,false));
    s.evaluate_sequence=s.color.sequence+65;
    CHECK(!nrb::dlss_color_processing_source(s,true));
    p.reset(f.list(),true);f.barrier(p);s=f.observe(p);
    s.evaluate_sequence=s.color.sequence+65;
    CHECK(!nrb::legacy_dlss_color_source(s));CHECK(allowed(s));
    CHECK(nrb::dlss_color_processing_source(s,true));
    CHECK(!nrb::dlss_color_processing_source(s,true,true));
    CHECK(!nrb::dlss_color_processing_source(s,false));
    s.closed=true;CHECK(!nrb::dlss_color_processing_source(s,true));
    s.closed=false;s.reset_observed=false;
    CHECK(!nrb::dlss_color_processing_source(s,true));
    s.reset_observed=true;s.color.after=uav;s.color_state.transition.after=uav;
    CHECK(!nrb::dlss_color_processing_source(s,true));
}
void unrelated_churn_and_ring_loss() {
    Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
    const auto initial=f.observe(p);CHECK(allowed(initial));
    p.reset(f.list(1),true);f.resources[1].candidate=false;
    for(int i=0;i<600;++i) {
        f.barrier(p,1,1);f.observe(p,1,1,false);
        p.submit(reinterpret_cast<ID3D12CommandList*>(f.list(1)),uint64_t(i)+1);
    }
    const auto after=f.observe(p);CHECK(allowed(after));CHECK(after.incomplete);
    CHECK(after.color.sequence==initial.color.sequence);
    CHECK(after.color_state.local_ordinal==initial.color_state.local_ordinal);
    CHECK(after.evaluate_sequence-after.color.sequence>256);
    CHECK(!nrb::legacy_dlss_color_source(after));
    for(int i=0;i<300;++i) CHECK(allowed(f.observe(p)));
}
void known_candidate_noise() {
    Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);p.reset(f.list(1),true);
    f.barrier(p,1,1);
    for(int i=0;i<247;++i) f.observe(p,1,1,false);
    const auto s=f.observe(p);CHECK(allowed(s));CHECK(s.evaluate_sequence-s.color.sequence>64);
    CHECK(!f.observe(p,0,0,false).color_state.enabled); // other native routes use the legacy observation
}
void latest_state_and_batch_ordinal() {
    Fixture f;nrb::SyncProbe p(f.ids.ops(),true);p.reset(f.list(),true);
    auto batch=std::array{f.transition(),f.transition(1)};p.barriers(f.list(),2,batch.data());
    const auto s=f.observe(p);CHECK(allowed(s));
    CHECK(s.color.sequence+1==f.observe(p,0,1).color.sequence);
    f.barrier(p,0,0,psr,uav);rejected(f.observe(p),Reason::not_psr);
    f.barrier(p,0,0,uav,psr);CHECK(allowed(f.observe(p)));
    f.barrier(p,0,0,psr,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    rejected(f.observe(p),Reason::not_psr);
}
void inconsistent_chain_sticky() {
    Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
    f.barrier(p,0,0,uav,psr);rejected(f.observe(p),Reason::state_chain);
    f.barrier(p,0,0,psr,psr);rejected(f.observe(p),Reason::state_chain);
    p.reset(f.list(),true);CHECK(!allowed(f.observe(p)));f.barrier(p);CHECK(allowed(f.observe(p)));
}
void partial_and_split_sticky() {
    for(int variant=0;variant<3;++variant) {
        Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
        auto b=f.transition(0,psr,psr,variant==0?0:D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            variant==1?D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY:
            variant==2?D3D12_RESOURCE_BARRIER_FLAG_END_ONLY:D3D12_RESOURCE_BARRIER_FLAG_NONE);
        p.barriers(f.list(),1,&b);rejected(f.observe(p),Reason::partial_split);
        f.barrier(p,0,0,psr,psr);rejected(f.observe(p),Reason::partial_split);
    }
}
void alias_and_unknown_batch() {
    for(int variant=0;variant<5;++variant) {
        Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
        D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        if(variant==1) b.Aliasing={f.resource(1),f.resource(2)};
        if(variant<2) p.barriers(f.list(1),1,&b); // alias alone does not replace recorded access state
        else if(variant==2) p.barriers(f.list(1),65,&b); // bound checked before dereference
        else if(variant==3) p.barriers(f.list(1),1,nullptr);
        else {b.Type=static_cast<D3D12_RESOURCE_BARRIER_TYPE>(99);p.barriers(f.list(1),1,&b);}
        if(variant<2) {
            CHECK(allowed(f.observe(p)));
            f.barrier(p,0,0,psr,uav);rejected(f.observe(p),Reason::not_psr);
            f.barrier(p,0,0,uav,psr);CHECK(allowed(f.observe(p)));
        } else {
            rejected(f.observe(p),Reason::unknown_event);
            f.barrier(p,0,0,psr,psr);rejected(f.observe(p),Reason::unknown_event);
        }
        p.reset(f.list(),true);CHECK(!allowed(f.observe(p)));f.barrier(p);CHECK(allowed(f.observe(p)));
    }
}
void same_resource_cross_list() {
    for(int variant=0;variant<3;++variant) {
        Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
        if(variant==0) p.reset(f.list(1),true);
        if(variant<2) f.barrier(p,1,0,psr,uav);
        else {D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;
            b.UAV.pResource=f.resource();p.barriers(f.list(1),1,&b);}
        CHECK(allowed(f.observe(p))); // another recording is not this list's state
        if(variant==0) {
            rejected(f.observe(p,1),Reason::not_psr);
            f.barrier(p,1,0,uav,psr);CHECK(allowed(f.observe(p,1)));
        }
        f.barrier(p,0,0,psr,uav);rejected(f.observe(p),Reason::not_psr);
        if(variant==0) CHECK(allowed(f.observe(p,1)));
        f.barrier(p,0,0,uav,psr);CHECK(allowed(f.observe(p)));
    }
}
void cross_list_is_resource_local_and_can_extend_age() {
    Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
    f.barrier(p,0,1);p.reset(f.list(1),true);
    for(int i=0;i<80;++i) {
        f.barrier(p,1,0,psr,uav);f.barrier(p,1,0,uav,psr);
    }
    auto s=f.observe(p);
    CHECK(s.incomplete);CHECK(allowed(s));CHECK(allowed(f.observe(p,0,1)));
    CHECK(!nrb::legacy_dlss_color_source(s));
    CHECK(nrb::dlss_color_processing_source(s,true));
    CHECK(!nrb::dlss_color_processing_source(s,false));
    CHECK(!nrb::dlss_color_processing_source(s,true,true));
}
void cross_list_cannot_erase_permanent_rejections() {
    for(int variant=0;variant<2;++variant) {
        Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
        const auto bad=(variant==0?f.transition(0,psr,uav,0):f.transition(0,copy_dest,uav));
        p.barriers(f.list(),1,&bad);
        const auto reason=variant==0?Reason::partial_split:Reason::state_chain;
        rejected(f.observe(p),reason);
        p.reset(f.list(1),true);f.barrier(p,1,0,uav,psr);
        rejected(f.observe(p),reason);
        f.barrier(p,0,0,psr,uav);rejected(f.observe(p),reason);
        f.barrier(p,0,0,uav,psr);rejected(f.observe(p),reason);
    }
}
void capacity_sticky() {
    Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
    for(size_t i=1;i<=4;++i) f.barrier(p,0,i);
    rejected(f.observe(p),Reason::capacity);f.barrier(p,0,0,psr,psr);
    rejected(f.observe(p),Reason::capacity);CHECK(f.ids.live==5); // one list plus four pinned resources
    p.reset(f.list(),true);CHECK(!allowed(f.observe(p)));f.barrier(p);CHECK(allowed(f.observe(p)));
}
void lifecycle_and_generation() {
    Fixture f;nrb::SyncProbe p(f.ids.ops(),true);
    CHECK(!allowed(f.observe(p)));p.reset(f.list(),false);CHECK(!allowed(f.observe(p)));
    f.begin(p);const auto previous=f.observe(p);p.close(f.list(),true);
    rejected(f.observe(p),Reason::closed);CHECK(f.ids.live==0);
    p.reset(f.list(),true);const auto fresh=f.observe(p);
    CHECK(fresh.generation!=previous.generation);CHECK(!allowed(fresh));f.barrier(p);
    p.close(f.list(),false);rejected(f.observe(p),Reason::lifecycle);
    f.barrier(p,0,0,psr,psr);rejected(f.observe(p),Reason::lifecycle);
    p.reset(f.list(),true);f.barrier(p);p.reset(f.list(),false);
    rejected(f.observe(p),Reason::lifecycle);
}
void identity_reuse_and_provenance() {
    for(int variant=0;variant<4;++variant) {
        Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);CHECK(allowed(f.observe(p)));
        if(variant==0) ++f.resources[0].cookie;
        else if(variant==1) ++f.lists[0].cookie;
        else if(variant==2) f.resources[0].known=false;
        else f.resources[0].candidate=false;
        rejected(f.observe(p),Reason::identity);
        f.resources[0].known=true;f.resources[0].candidate=true;
        f.barrier(p,0,0,psr,psr);rejected(f.observe(p),Reason::identity);
        p.reset(f.list(),true);CHECK(!allowed(f.observe(p)));f.barrier(p);CHECK(allowed(f.observe(p)));
    }
    Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
    ++f.resources[0].cookie;f.barrier(p,0,0,psr,psr); // reuse observed at the next barrier, before Evaluate
    rejected(f.observe(p),Reason::identity);
}
void implementation_gate_and_lru() {
    Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
    const auto good=f.observe(p);CHECK(allowed(good));CHECK(!nrb::dlss_color_source(good,false));
    auto wrong=good;++wrong.color_state.generation;CHECK(!allowed(wrong));
    p.unknown_recording(f.list());rejected(f.observe(p),Reason::implementation);
    f.barrier(p,0,0,psr,psr);rejected(f.observe(p),Reason::implementation);
    p.reset(f.list(),true);f.barrier(p);CHECK(allowed(f.observe(p)));
    for(size_t i=1;i<=64;++i) p.reset(f.list(i),true);
    rejected(f.observe(p),Reason::no_reset);f.barrier(p);rejected(f.observe(p),Reason::no_reset);
    CHECK(f.ids.live==64);CHECK(f.ids.peak<=65);
}
void identity_budget_and_uav() {
    Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
    D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource=f.resource();p.barriers(f.list(),1,&b);CHECK(allowed(f.observe(p)));
    b.UAV.pResource=nullptr;p.barriers(f.list(1),1,&b);CHECK(allowed(f.observe(p)));
    CHECK(f.ids.live==2);CHECK(f.ids.peak<=3);
    p.reset(f.list(),true);CHECK(f.ids.live==1);CHECK(!allowed(f.observe(p)));
}
int main() {
    struct Test {const char* name;void(*run)();};
    const Test tests[]={
        {"default_off_and_exact_enable_scope",disabled_contract},
        {"normal_launch_exact_cp_profile_and_explicit_override",normal_launch_profile_and_override},
        {"shadow_retains_legacy_gate_and_reports_rejection",shadow_gate_contract},
        {"processing_gate_preserves_working_path_and_extends_age",processing_gate_extension},
        {"unrelated_churn_and_global_ring_overwrite",unrelated_churn_and_ring_loss},
        {"different_candidate_list_and_native_route_scope",known_candidate_noise},
        {"latest_state_and_exact_barrier_ordinal",latest_state_and_batch_ordinal},
        {"contradictory_state_chain_no_revival",inconsistent_chain_sticky},
        {"partial_begin_end_split_no_revival",partial_and_split_sticky},
        {"alias_access_state_unchanged_unknown_batch_still_rejected",alias_and_unknown_batch},
        {"other_recording_does_not_rewrite_this_lists_declaration",same_resource_cross_list},
        {"same_resource_other_recording_churn_preserves_strict_age_extension",cross_list_is_resource_local_and_can_extend_age},
        {"cross_list_cannot_erase_partial_or_bad_chain",cross_list_cannot_erase_permanent_rejections},
        {"four_resource_capacity_no_revival",capacity_sticky},
        {"reset_close_generation_failed_lifecycle",lifecycle_and_generation},
        {"list_resource_identity_reuse_unknown_no_revival",identity_reuse_and_provenance},
        {"implementation_gate_and_list_eviction",implementation_gate_and_lru},
        {"bounded_identity_pins_and_known_uav",identity_budget_and_uav},
    };
    for(const auto& t:tests) {t.run();std::cout<<"PASS "<<t.name<<"\n";}
}
