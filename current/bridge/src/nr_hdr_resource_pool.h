#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <limits>
#include <utility>
#include <stdexcept>

namespace nrb {
inline bool hdr_same_desc(const D3D12_RESOURCE_DESC& a,const D3D12_RESOURCE_DESC& b) {
    // Compare fields, never padding. Canonical device identity is held strongly
    // in the bundle for the entire residency, including quarantine.
    return a.Dimension==b.Dimension && a.Alignment==b.Alignment &&
        a.Width==b.Width && a.Height==b.Height && a.DepthOrArraySize==b.DepthOrArraySize &&
        a.MipLevels==b.MipLevels && a.Format==b.Format &&
        a.SampleDesc.Count==b.SampleDesc.Count && a.SampleDesc.Quality==b.SampleDesc.Quality &&
        a.Layout==b.Layout && a.Flags==b.Flags;
}
struct HdrResourceKey {
    IUnknown* device_identity=nullptr;
    D3D12_RESOURCE_DESC source{},xess{},game_motion{},proxy{},nr_motion{};
    D3D12_DESCRIPTOR_HEAP_DESC heap{};
    bool motion_available=false;
    uint32_t layout_abi=1;
    D3D12_RESOURCE_STATES initial=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES terminal=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    bool operator==(const HdrResourceKey& b) const {
        return device_identity==b.device_identity && layout_abi==b.layout_abi &&
            motion_available==b.motion_available && initial==b.initial && terminal==b.terminal &&
            hdr_same_desc(source,b.source) && hdr_same_desc(xess,b.xess) &&
            hdr_same_desc(game_motion,b.game_motion) && hdr_same_desc(proxy,b.proxy) &&
            hdr_same_desc(nr_motion,b.nr_motion) && heap.Type==b.heap.Type &&
            heap.NumDescriptors==b.heap.NumDescriptors && heap.Flags==b.heap.Flags &&
            heap.NodeMask==b.heap.NodeMask;
    }
};

// This same proof object drives the product and CPU fake-lease tests. All lists
// touching the bundle are private to Pending. Record destruction is safe only
// before queue exposure, or after the exact consumer fence and processor proof.
class HdrLeaseProof {
public:
    bool pristine_unsubmitted() const {return !queue_started_ && !processor_called_ && !poisoned_;}
    bool terminal_safe() const {
        return !poisoned_ && private_records_closed_ && prepare_recorded_ && prepare_submitted_ && deferred_submitted_ &&
            completion_signaled_ && completion_succeeded_ && records_safe_ &&
            (!composite_recorded_ || composite_submitted_) &&
            (!processor_called_ || (processor_safe_ && processor_retired_));
    }
    bool can_rewrite_descriptors() const {
        return !poisoned_ && prepare_completed_ && processor_safe_ && !composite_recorded_;
    }
    // GPU ordering authorizes the callback, never rewriting a prepare heap
    // that the GPU may still be reading. Composite uses a separate private heap.
    bool can_record_private_composite() const {
        return !poisoned_ && prepare_gpu_ordered_ && processor_safe_ && !composite_recorded_;
    }
    bool processor_called() const {return processor_called_;}
    bool processor_safe() const {return processor_safe_;}
    bool processor_retired() const {return processor_retired_;}
    bool poisoned() const {return poisoned_;}
    bool queue_started() const {return queue_started_;}
    void prepare_recorded() {if(prepare_recorded_ || queue_started_) poison();else prepare_recorded_=true;}
    void private_records_closed() {
        if(!prepare_recorded_ || queue_started_ || private_records_closed_) poison();
        else private_records_closed_=true;
    }
    // Call BEFORE original_execute, including exceptional/reentrant paths.
    void prepare_submitted() {
        queue_started_=true;
        if(!private_records_closed_ || !prepare_recorded_ || prepare_submitted_) poison();
        prepare_submitted_=true;
    }
    void prepare_completed(bool exact_success) {
        if(!exact_success || !prepare_submitted_) poison();else prepare_completed_=true;
    }
    void prepare_gpu_ordered(bool exact_signaled_and_qualified) {
        if(!exact_signaled_and_qualified || !prepare_submitted_) poison();else prepare_gpu_ordered_=true;
    }
    void processor_begin() {
        if((!prepare_completed_ && !prepare_gpu_ordered_) || processor_called_) poison();
        processor_called_=true;
    }
    // Success means a valid processor result AND a successful queue Wait on
    // its ready fence. A failed/throwing callback has unknown XPU use: poison.
    void processor_ready(bool safe) {
        if(!processor_called_ || !safe) poison();else processor_safe_=true;
    }
    void composite_recorded(bool private_descriptors=false) {
        if(!(private_descriptors?can_record_private_composite():can_rewrite_descriptors())) poison();
        else composite_recorded_=true;
    }
    void composite_submitted() {
        if(!composite_recorded_ || !composite_closed_ || composite_submitted_) poison();
        composite_submitted_=true;queue_started_=true;
    }
    void composite_closed() {if(!composite_recorded_ || composite_closed_) poison();else composite_closed_=true;}
    void deferred_submitted() {
        if(!private_records_closed_ || !prepare_submitted_ || deferred_submitted_) poison();
        deferred_submitted_=true;queue_started_=true;
    }
    void completion_signal(bool success) {
        if(!deferred_submitted_ || !queue_started_ || !success) poison();else completion_signaled_=true;
    }
    // A completed value alone cannot repair a failed Signal, device removal,
    // callback error or unknown record. UINT64_MAX is never completion success.
    void consumer_completed(uint64_t value,bool all_private_records_safe) {
        if(!completion_signaled_ || value<1 || value==UINT64_MAX ||
           !all_private_records_safe) {poison();return;}
        completion_succeeded_=true;records_safe_=true;prepare_completed_=true;
    }
    void processor_retirement(bool success) {
        if(!completion_succeeded_ || !processor_safe_ || !success) poison();
        else processor_retired_=true;
    }
    void poison() {poisoned_=true;}
private:
    bool prepare_recorded_=false,private_records_closed_=false,prepare_submitted_=false,prepare_completed_=false;
    bool queue_started_=false,processor_called_=false,processor_safe_=false,processor_retired_=false;
    bool prepare_gpu_ordered_=false;
    bool composite_recorded_=false,composite_closed_=false,composite_submitted_=false,deferred_submitted_=false;
    bool completion_signaled_=false,completion_succeeded_=false,records_safe_=false,poisoned_=false;
};

struct HdrResourcePoolStats {
    uint64_t acquires=0,hits=0,creates=0,disabled_fallbacks=0,busy_fallbacks=0;
    uint64_t creation_failures=0,retire_returns=0,unsubmitted_returns=0,quarantines=0,evictions=0;
    uint64_t generation=1,resident_slots=0,free_slots=0,leased_slots=0,quarantined_slots=0;
    uint64_t texture_bytes=0,peak_texture_bytes=0,descriptor_bytes_estimate=0;
    bool enabled=false,shutdown=false;
};

struct HdrReuseTransitions {
    UINT count=0;
    std::array<D3D12_RESOURCE_BARRIER,2> barriers{};
};
inline HdrReuseTransitions hdr_reuse_transitions(ID3D12Resource* proxy,ID3D12Resource* motion,
                                                D3D12_RESOURCE_STATES before) {
    HdrReuseTransitions result;
    if(before==D3D12_RESOURCE_STATE_UNORDERED_ACCESS) return result;
    result.count=2;
    ID3D12Resource* resources[]={proxy,motion};
    for(size_t i=0;i<2;++i) {
        auto& barrier=result.barriers[i];barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition={resources[i],D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            before,D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
    }
    return result;
}

// Bundle supplies texture_bytes, descriptor_bytes_estimate and clear_bindings().
// The production specialization below owns COM objects; tests use fake owners.
// A lease is exclusive, serial tagged and not copyable. No implicit unsafe
// return, spin, wait, dynamic capacity or eviction of an active/poisoned slot.
template<class Bundle> class BoundedHdrResourcePool {
    enum class Phase {empty,free,leased,quarantined};
    struct Slot {
        HdrResourceKey key;
        std::unique_ptr<Bundle> bundle;
        Phase phase=Phase::empty;
        uint64_t serial=0,generation=0;
        D3D12_RESOURCE_STATES state=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    };
    struct State {
        std::mutex mutex;
        std::array<Slot,2> slots;
        HdrResourcePoolStats counters;
        uint64_t serial=0;
        ~State() {
            // Deliberate, bounded process-lifetime retention if the caller
            // destroys a session without retirement. Never Release GPU use.
            for(auto& slot:slots) if(slot.phase==Phase::leased || slot.phase==Phase::quarantined)
                (void)slot.bundle.release();
        }
        void evict(Slot& slot) {
            counters.texture_bytes-=slot.bundle->texture_bytes;
            counters.descriptor_bytes_estimate-=slot.bundle->descriptor_bytes_estimate;
            slot.bundle.reset();slot.phase=Phase::empty;++counters.evictions;
        }
    };
public:
    static constexpr size_t capacity=2;
    static constexpr uint64_t max_texture_bytes=64ull*1024*1024;
    class Lease {
        friend class BoundedHdrResourcePool;
        std::shared_ptr<State> owner_;
        size_t index_=0;
        uint64_t serial_=0,generation_=0;
        D3D12_RESOURCE_STATES initial_;
        HdrLeaseProof proof_;
        Lease(std::shared_ptr<State> owner,size_t index):owner_(std::move(owner)),index_(index) {
            auto& slot=owner_->slots[index];generation_=slot.generation;initial_=slot.state;
        }
        Slot* matching() const {
            auto& slot=owner_->slots[index_];
            return slot.serial==serial_ && slot.generation==generation_?&slot:nullptr;
        }
    public:
        Lease(const Lease&)=delete;
        Lease& operator=(const Lease&)=delete;
        ~Lease() {if(owner_ && serial_) {if(!finish()) quarantine();}}
        Bundle& bundle() const {
            if(!owner_ || !serial_) throw std::logic_error("returned HDR resource lease");
            auto* slot=matching();
            if(!slot || slot->phase!=Phase::leased) throw std::logic_error("quarantined HDR resource lease");
            return *slot->bundle;
        }
        HdrLeaseProof& proof() {return proof_;}
        const HdrLeaseProof& proof() const {return proof_;}
        D3D12_RESOURCE_STATES initial_state() const {return initial_;}
        // Caller must destroy private unsubmitted records, or retire Pending,
        // before allowing any subsequent acquire/rebind after this return.
        bool finish() {
            if(!owner_ || !serial_) return true;
            std::lock_guard lock(owner_->mutex);
            auto* slot=matching();
            if(!slot || slot->phase!=Phase::leased ||
               (!proof_.terminal_safe() && !proof_.pristine_unsubmitted())) return false;
            if(proof_.terminal_safe()) {slot->state=slot->key.terminal;++owner_->counters.retire_returns;}
            else ++owner_->counters.unsubmitted_returns;
            slot->bundle->clear_bindings();slot->phase=Phase::free;
            if(owner_->counters.shutdown || slot->generation!=owner_->counters.generation)
                owner_->evict(*slot);
            // Keep State alive until lock_guard is destroyed.
            serial_=0;return true;
        }
        void quarantine() {
            if(!owner_ || !serial_) return;
            std::lock_guard lock(owner_->mutex);
            if(auto* slot=matching();slot && slot->phase==Phase::leased) {
                slot->phase=Phase::quarantined;++owner_->counters.quarantines;
            }
            proof_.poison();
        }
        bool returned() const {return !serial_;}
    };
    BoundedHdrResourcePool():state_(std::make_shared<State>()) {}
    BoundedHdrResourcePool(const BoundedHdrResourcePool&)=delete;
    BoundedHdrResourcePool& operator=(const BoundedHdrResourcePool&)=delete;
    ~BoundedHdrResourcePool() {shutdown();}
    void set_enabled(bool enabled) {std::lock_guard lock(state_->mutex);if(!state_->counters.shutdown) state_->counters.enabled=enabled;}
    bool enabled() const {std::lock_guard lock(state_->mutex);return state_->counters.enabled;}
    void invalidate() {
        std::lock_guard lock(state_->mutex);++state_->counters.generation;
        for(auto& slot:state_->slots) if(slot.phase==Phase::free) state_->evict(slot);
    }
    bool shutdown() {
        std::lock_guard lock(state_->mutex);state_->counters.enabled=false;state_->counters.shutdown=true;
        bool safe=true;
        for(auto& slot:state_->slots) {
            if(slot.phase==Phase::free) state_->evict(slot);
            else if(slot.phase!=Phase::empty) safe=false;
        }
        return safe;
    }
    HdrResourcePoolStats stats() const {
        std::lock_guard lock(state_->mutex);auto result=state_->counters;
        for(const auto& slot:state_->slots) {
            result.resident_slots+=slot.phase!=Phase::empty;
            result.free_slots+=slot.phase==Phase::free;result.leased_slots+=slot.phase==Phase::leased;
            result.quarantined_slots+=slot.phase==Phase::quarantined;
        }
        return result;
    }
    template<class Factory> std::unique_ptr<Lease> acquire(const HdrResourceKey& key,Factory&& factory) {
        std::lock_guard lock(state_->mutex);auto& stats=state_->counters;++stats.acquires;
        if(!stats.enabled || stats.shutdown) {++stats.disabled_fallbacks;return {};}
        size_t index=capacity;
        for(size_t i=0;i<capacity;++i) if(state_->slots[i].phase==Phase::free &&
            state_->slots[i].key==key && state_->slots[i].generation==stats.generation) {index=i;break;}
        const bool hit=index!=capacity;
        if(!hit) {
            for(size_t i=0;i<capacity;++i) if(state_->slots[i].phase==Phase::empty) {index=i;break;}
            if(index==capacity) for(size_t i=0;i<capacity;++i) if(state_->slots[i].phase==Phase::free) {index=i;break;}
            if(index==capacity) {++stats.busy_fallbacks;return {};}
        }
        auto& slot=state_->slots[index];
        try {
            // Allocate the handle before mutating an idle slot. An allocation
            // failure must not leave an ownerless leased slot.
            auto lease=std::unique_ptr<Lease>(new Lease(state_,index));
            if(!hit) {
                if(slot.bundle) state_->evict(slot); // only a proven free victim
                auto bundle=factory();
                if(!bundle || !bundle->texture_bytes || bundle->texture_bytes>max_texture_bytes ||
                   stats.texture_bytes>max_texture_bytes-bundle->texture_bytes) {
                    ++stats.creation_failures;lease->serial_=0;return {};
                }
                slot.bundle=std::move(bundle);slot.key=key;slot.state=key.initial;
                stats.texture_bytes+=slot.bundle->texture_bytes;
                stats.descriptor_bytes_estimate+=slot.bundle->descriptor_bytes_estimate;
                if(stats.texture_bytes>stats.peak_texture_bytes) stats.peak_texture_bytes=stats.texture_bytes;
                ++stats.creates;
            } else ++stats.hits;
            slot.phase=Phase::leased;slot.generation=stats.generation;slot.serial=++state_->serial;
            lease->serial_=slot.serial;lease->generation_=slot.generation;lease->initial_=slot.state;
            return lease;
        } catch(...) {++stats.creation_failures;return {};}
    }
private:
    std::shared_ptr<State> state_;
};

struct HdrResourceBundle {
    Microsoft::WRL::ComPtr<IUnknown> identity;
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    Microsoft::WRL::ComPtr<ID3D12Resource> proxy,motion;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
    // Pending also owns these originals. Retain descriptor references inside
    // quarantine even if a standalone caller mistakenly drops its HdrProxy.
    Microsoft::WRL::ComPtr<ID3D12Resource> bound_source,bound_motion,bound_xess,bound_nr_result;
    UINT stride=0;
    uint64_t texture_bytes=0,descriptor_bytes_estimate=0;
    void clear_bindings() {bound_source.Reset();bound_motion.Reset();bound_xess.Reset();bound_nr_result.Reset();}
};
using HdrProxyResourcePool=BoundedHdrResourcePool<HdrResourceBundle>;
// Direct C++ controls for the deferred session; implemented only in
// deferred_identity.cpp. These and the C ABI do not imply a wired web control.
bool deferred_hdr_set_resource_pool_enabled(bool enabled);
bool deferred_hdr_resource_pool_stats(HdrResourcePoolStats& stats);
}
