#include "nr_hdr_proxy.h"
#include "nr_hdr_resource_pool_api.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <thread>
#include <type_traits>
#include <vector>

using namespace nrb;
using Microsoft::WRL::ComPtr;
namespace {
unsigned checks=0;
void require(bool success,const char* text) {
    ++checks;
    if(!success) {std::fprintf(stderr,"FAIL: %s\n",text);std::exit(1);}
}
struct FakeCanonical final : IUnknown {
    std::atomic<ULONG> refs{1};
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid,void** out) override {
        if(!out) return E_POINTER;*out=nullptr;
        if(iid!=__uuidof(IUnknown)) return E_NOINTERFACE;
        *out=static_cast<IUnknown*>(this);AddRef();return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override {return ++refs;}
    ULONG STDMETHODCALLTYPE Release() override {auto value=--refs;if(!value) delete this;return value;}
};
struct FakeAlias final : IUnknown {
    ComPtr<IUnknown> canonical;ULONG refs=1;
    explicit FakeAlias(IUnknown* identity):canonical(identity) {}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid,void** out) override {return canonical->QueryInterface(iid,out);}
    ULONG STDMETHODCALLTYPE AddRef() override {return ++refs;}
    ULONG STDMETHODCALLTYPE Release() override {auto value=--refs;if(!value) delete this;return value;}
};
struct ObservedOwner {std::atomic<unsigned> destroyed{0},clears{0};};
struct FakeBundle {
    ComPtr<IUnknown> identity;
    std::shared_ptr<ObservedOwner> observed;
    uint64_t texture_bytes=18481152,descriptor_bytes_estimate=160,id=0,descriptor_source=0;
    static std::atomic<uint64_t> ids;
    FakeBundle(IUnknown* device,std::shared_ptr<ObservedOwner> owner):identity(device),observed(std::move(owner)),id(++ids) {}
    ~FakeBundle() {++observed->destroyed;}
    void clear_bindings() {descriptor_source=0;++observed->clears;}
};
std::atomic<uint64_t> FakeBundle::ids{0};
using Pool=BoundedHdrResourcePool<FakeBundle>;
static_assert(!std::is_copy_constructible_v<Pool::Lease>);
static_assert(Pool::capacity==HdrProxyResourcePool::capacity && Pool::capacity==2);
using ProductInit=bool(HdrProxy::*)(ID3D12Device*,ID3D12Resource*,ID3D12Resource*,ID3D12Resource*,
    const std::wstring&,HdrProxyCache*,HdrProxyResourcePool*);
static_assert(std::is_same_v<decltype(static_cast<ProductInit>(&HdrProxy::initialize)),ProductInit>);
#ifndef NRB_POOL_HEADER_ONLY_CPU
ProductInit product_init=static_cast<ProductInit>(&HdrProxy::initialize);
#endif

HdrResourceKey key(IUnknown* device) {
    HdrResourceKey k;k.device_identity=device;
    k.source.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    k.source.Width=1280;k.source.Height=720;k.source.DepthOrArraySize=1;
    k.source.MipLevels=1;k.source.SampleDesc.Count=1;
    k.source.Format=DXGI_FORMAT_R11G11B10_FLOAT;
    k.xess=k.source;k.xess.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    k.proxy=k.xess;k.proxy.Format=DXGI_FORMAT_R32G32B32A32_FLOAT;
    k.nr_motion=k.proxy;k.nr_motion.Format=DXGI_FORMAT_R16G16_FLOAT;
    k.heap={D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,5,D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,0};
    return k;
}
auto factory(IUnknown* device,const std::shared_ptr<ObservedOwner>& owner) {
    return [device,owner] {return std::make_unique<FakeBundle>(device,owner);};
}
void submitted(Pool::Lease& lease,bool processor) {
    auto& proof=lease.proof();proof.prepare_recorded();proof.private_records_closed();proof.prepare_submitted();
    if(processor) {proof.prepare_completed(true);proof.processor_begin();proof.processor_ready(true);
        proof.composite_recorded();proof.composite_closed();proof.composite_submitted();}
}
void terminal(Pool::Lease& lease,bool processor) {
    lease.proof().deferred_submitted();lease.proof().completion_signal(true);lease.proof().consumer_completed(1,true);
    if(processor) lease.proof().processor_retirement(true);
}
void busy_no_alias(IUnknown* device) {
    auto observed=std::make_shared<ObservedOwner>();Pool pool;const auto k=key(device);
    unsigned attempts=0;
    auto make=[&] {++attempts;return std::make_unique<FakeBundle>(device,observed);};
    require(!pool.acquire(k,make) && !attempts,"default OFF never creates");
    pool.set_enabled(true);auto a=pool.acquire(k,make),b=pool.acquire(k,make);
    require(a && b && a->bundle().id!=b->bundle().id,"two exclusive bundles/heaps");
    a->bundle().descriptor_source=11;b->bundle().descriptor_source=22;
    require(!pool.acquire(k,make) && attempts==2,"busy bounded miss has no wait or third allocation");
    require(a->bundle().descriptor_source==11 && b->bundle().descriptor_source==22,"busy miss cannot rebind either heap");
    submitted(*a,true);require(!a->finish(),"queue exposed frame cannot return before exact terminal");
    a->proof().deferred_submitted();a->proof().completion_signal(true);a->proof().consumer_completed(1,true);
    require(!a->finish(),"D3D12 completion alone cannot retire processor input");
    a->proof().processor_retirement(true);const auto id=a->bundle().id;
    require(a->finish(),"consumer AND processor retirement returns");
    require(!a->proof().pristine_unsubmitted(),"submitted frame never becomes pristine");
    bool threw=false;try {(void)a->bundle();} catch(const std::logic_error&) {threw=true;}
    require(threw,"returned handle cannot mutate rebound heap");
    auto c=pool.acquire(k,make);
    require(c && c->bundle().id==id && c->bundle().descriptor_source==0,"hit uses safely retired slot only");
    require(c->initial_state()==D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,"hit tracks executed terminal read state");
    auto* proxy=reinterpret_cast<ID3D12Resource*>(uintptr_t(0x1000));
    auto* motion=reinterpret_cast<ID3D12Resource*>(uintptr_t(0x2000));
    const auto transitions=hdr_reuse_transitions(proxy,motion,c->initial_state());
    require(transitions.count==2,"both reused textures transition before prepare");
    for(unsigned i=0;i<2;++i) require(transitions.barriers[i].Transition.StateBefore==D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE &&
        transitions.barriers[i].Transition.StateAfter==D3D12_RESOURCE_STATE_UNORDERED_ACCESS &&
        transitions.barriers[i].Transition.Subresource==D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES &&
        transitions.barriers[i].Transition.pResource==(i?motion:proxy),"actual product barrier helper has exact before/after/resources");
    require(!hdr_reuse_transitions(proxy,motion,D3D12_RESOURCE_STATE_UNORDERED_ACCESS).count,"fresh UAV set adds no false barrier");
    c->proof().prepare_recorded();require(c->finish(),"recorded but private unsubmitted lists may be destroyed");
    auto d=pool.acquire(k,make);require(d->initial_state()==D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,"unexecuted transition cannot alter physical state");
    require(b->finish() && d->finish(),"pristine leases return safely");
    const auto stats=pool.stats();require(stats.resident_slots==2 && stats.free_slots==2 && stats.hits==2 &&
        stats.texture_bytes==2*18481152 && stats.descriptor_bytes_estimate==320,"bounded bytes and hit/free counters");
    require(pool.shutdown() && observed->destroyed==2,"shutdown evicts only safe free slots");
}
void canonical_and_key(IUnknown* device) {
    ComPtr<IUnknown> alias1,alias2;alias1.Attach(new FakeAlias(device));alias2.Attach(new FakeAlias(device));
    ComPtr<IUnknown> canonical1,canonical2;
    require(SUCCEEDED(alias1.As(&canonical1)) && SUCCEEDED(alias2.As(&canonical2)) &&
        alias1.Get()!=alias2.Get() && canonical1.Get()==canonical2.Get(),"IUnknown canonical identity unifies interface aliases");
    auto k=key(canonical1.Get());require(k==key(canonical2.Get()),"full keys use canonical identity");
    auto other=ComPtr<IUnknown>();other.Attach(new FakeCanonical);
    require(!(k==key(other.Get())),"different device never aliases");
    std::vector<HdrResourceKey> variants;
    const auto changed=[&](auto mutation) {auto v=k;mutation(v);variants.push_back(v);};
    changed([](auto& v){v.source.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE3D;});
    changed([](auto& v){v.source.Alignment=65536;});changed([](auto& v){++v.source.Width;});
    changed([](auto& v){++v.source.Height;});changed([](auto& v){++v.source.DepthOrArraySize;});
    changed([](auto& v){++v.source.MipLevels;});changed([](auto& v){v.source.Format=DXGI_FORMAT_R16G16_FLOAT;});
    changed([](auto& v){++v.source.SampleDesc.Count;});changed([](auto& v){++v.source.SampleDesc.Quality;});
    changed([](auto& v){v.source.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;});
    changed([](auto& v){v.source.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;});
    changed([](auto& v){++v.xess.Height;});changed([](auto& v){++v.proxy.Width;});
    changed([](auto& v){++v.nr_motion.Height;});changed([](auto& v){v.motion_available=true;});
    changed([](auto& v){++v.game_motion.Width;});changed([](auto& v){++v.heap.NumDescriptors;});
    changed([](auto& v){v.heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV;});
    changed([](auto& v){v.heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_NONE;});
    changed([](auto& v){++v.heap.NodeMask;});changed([](auto& v){++v.layout_abi;});
    changed([](auto& v){v.initial=D3D12_RESOURCE_STATE_COMMON;});
    changed([](auto& v){v.terminal=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;});
    for(auto& v:variants) require(!(k==v),"geometry/layout/format/heap/state differences reject compatibility");
    auto observed=std::make_shared<ObservedOwner>();Pool pool;pool.set_enabled(true);
    auto a=pool.acquire(k,factory(device,observed));auto b=pool.acquire(variants.front(),factory(device,observed));
    require(a && b && a->bundle().id!=b->bundle().id,"incompatible geometry cannot reuse live bundle");
    require(a->finish() && b->finish(),"independent canonical keys return");
}
void generation_shutdown(IUnknown* device) {
    auto observed=std::make_shared<ObservedOwner>();Pool pool;pool.set_enabled(true);
    auto a=pool.acquire(key(device),factory(device,observed));submitted(*a,true);
    pool.invalidate();require(pool.stats().generation==2 && !observed->destroyed,"invalidation retains active old generation");
    auto b=pool.acquire(key(device),factory(device,observed));require(b && a->bundle().id!=b->bundle().id,"old generation never rebound");
    terminal(*a,true);require(a->finish() && observed->destroyed==1,"old generation evicted only after exact terminal");
    submitted(*b,false);require(!pool.shutdown() && observed->destroyed==1,"shutdown preserves GPU-exposed ownership");
    require(!pool.acquire(key(device),factory(device,observed)),"shutdown admits no new leases");
    terminal(*b,false);require(b->finish() && observed->destroyed==2,"post-shutdown retirement safely drops references");
}
void failures(IUnknown* device) {
    for(unsigned failure=0;failure<10;++failure) {
        auto observed=std::make_shared<ObservedOwner>();Pool pool;pool.set_enabled(true);
        auto a=pool.acquire(key(device),factory(device,observed));submitted(*a,failure!=8);
        if(failure!=9) a->proof().deferred_submitted();
        switch(failure) {
        case 0:a->proof().completion_signal(false);break;
        case 1:a->proof().completion_signal(true);a->proof().consumer_completed(UINT64_MAX,true);break;
        case 2:a->proof().completion_signal(true);a->proof().consumer_completed(0,true);break;
        case 3:a->proof().completion_signal(true);a->proof().consumer_completed(1,false);break;
        case 4:a->proof().completion_signal(true);a->proof().consumer_completed(1,true);a->proof().processor_retirement(false);break;
        case 5:a->proof().processor_ready(false);break;
        case 6:a->proof().poison();break; // exception/device loss/unproven record
        case 7:a->proof().prepare_submitted();break; // duplicate exposure
        case 8:a->proof().processor_begin();break; // processor without prepare proof
        case 9:a->proof().completion_signal(true);a->proof().consumer_completed(1,true);break; // missing deferred execution
        }
        require(!a->finish(),"failed/unknown lease cannot return");a->quarantine();a.reset();
        require(pool.stats().quarantined_slots==1 && !observed->destroyed && !observed->clears,
            "quarantine retains resources and bound descriptor references");
        pool.invalidate();
        auto b=pool.acquire(key(device),factory(device,observed));
        require(b && pool.stats().resident_slots==2,"poison occupies one of the bounded slots across generations");
        submitted(*b,false);
        require(!pool.acquire(key(device),factory(device,observed)),"poison plus busy slot falls back uniquely");
        terminal(*b,false);require(b->finish(),"healthy independent slot retires");
        require(!pool.shutdown() && observed->destroyed==1,"shutdown releases only healthy retired bundle");
    }
}
void cold_failures_and_toggle(IUnknown* device) {
    auto observed=std::make_shared<ObservedOwner>();Pool pool;pool.set_enabled(true);
    require(!pool.acquire(key(device),[]()->std::unique_ptr<FakeBundle>{return {};}),"creation failure returns a miss");
    require(!pool.acquire(key(device),[]()->std::unique_ptr<FakeBundle>{throw std::bad_alloc();}),"factory exception leaves no ownerless slot");
    auto too_big=[&] {auto result=std::make_unique<FakeBundle>(device,observed);
        result->texture_bytes=Pool::max_texture_bytes+1;return result;};
    require(!pool.acquire(key(device),too_big) && pool.stats().resident_slots==0,"byte cap rejects oversize bundle");
    auto a=pool.acquire(key(device),factory(device,observed));submitted(*a,true);
    const auto id=a->bundle().id;pool.set_enabled(false);
    require(!pool.acquire(key(device),factory(device,observed)) && a->bundle().id==id,"OFF toggle leaves active lease untouched");
    terminal(*a,true);require(a->finish(),"toggle OFF does not invalidate safe retirement");
    pool.set_enabled(true);auto b=pool.acquire(key(device),factory(device,observed));
    require(b && b->bundle().id==id && b->finish(),"independent resource toggle can reuse retired residency");
}
void concurrent_exclusive(IUnknown* device) {
    auto observed=std::make_shared<ObservedOwner>();Pool pool;pool.set_enabled(true);
    std::mutex active_mutex;std::set<uint64_t> active;
    std::atomic<unsigned> aliases{0},successes{0};
    std::vector<std::thread> threads;
    for(unsigned t=0;t<4;++t) threads.emplace_back([&] {
        for(unsigned i=0;i<256;++i) {
            auto lease=pool.acquire(key(device),factory(device,observed));if(!lease) continue;
            const auto id=lease->bundle().id;
            {std::lock_guard lock(active_mutex);if(!active.insert(id).second) ++aliases;}
            submitted(*lease,true);terminal(*lease,true);
            {std::lock_guard lock(active_mutex);active.erase(id);}
            if(!lease->finish()) ++aliases;else ++successes;
        }
    });
    for(auto& thread:threads) thread.join();
    require(!aliases && successes>0 && active.empty() && pool.stats().resident_slots<=2,
        "concurrent acquire/return never aliases a live bundle");
}
}

int main(int argc,char** argv) {
    ComPtr<IUnknown> device;device.Attach(new FakeCanonical);
    busy_no_alias(device.Get());canonical_and_key(device.Get());generation_shutdown(device.Get());failures(device.Get());
    cold_failures_and_toggle(device.Get());concurrent_exclusive(device.Get());
#ifndef NRB_POOL_HEADER_ONLY_CPU
    require(product_init!=nullptr,"actual HdrProxy pool overload links");
#endif
    require(sizeof(NRB_HdrResourcePoolStats)==168,"C ABI size is pinned");
#ifndef NRB_POOL_HEADER_ONLY_CPU
    if(argc>1) {
        HdrProxyCache cache;
        const auto size=MultiByteToWideChar(CP_UTF8,0,argv[1],-1,nullptr,0);
        std::wstring path(size,L'\0');MultiByteToWideChar(CP_UTF8,0,argv[1],-1,path.data(),size);path.pop_back();
        require(!cache.enabled() && cache.warmup_shader(path),"actual pipeline cache CPU shader warmup OFF");
        const auto first=cache.stats();cache.set_enabled(true);
        require(cache.warmup_shader(path) && cache.stats().shader_compile_calls==first.shader_compile_calls,
            "resource pool does not bypass shader generation cache semantics");
        cache.invalidate_shader();require(cache.warmup_shader(path) && cache.stats().shader_generation>first.shader_generation &&
            cache.stats().shader_compile_calls==first.shader_compile_calls+2,"actual shader invalidation recompiles new generation");
        cache.set_enabled(false);require(!cache.enabled(),"cache toggle remains independent");
    }
#else
    (void)argc;(void)argv;
#endif
    std::printf("PASS: %u CPU lease/key/ownership/state/product checks; GPU device/queue not created\n",checks);
}
