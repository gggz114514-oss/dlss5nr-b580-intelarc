#pragma once

#include <d3d12.h>
#include <array>
#include <cstdint>
#include <mutex>
#include <string_view>

namespace nrb {
// The original diagnostic ring holds raw pointers. The optional scoped proof
// below pins identities separately; a successful observed Reset is required.
struct TransitionObservation {
    bool seen=false;
    uint64_t sequence=0;
    D3D12_RESOURCE_STATES before{};
    D3D12_RESOURCE_STATES after{};
    D3D12_RESOURCE_BARRIER_FLAGS flags{};
    UINT subresource=0;
    bool full_non_pixel_read=false;
};
// Scoped recorded-state prototype, NOT GPU state/completion verification.
// Identity acquisition must pin the canonical COM identity from the barrier.
enum class ColorProofReason : uint32_t {
    ready, disabled, no_reset, closed, identity, capacity, partial_split,
    aliasing, cross_list, unknown_event, lifecycle, state_chain, not_psr,
    implementation
};
struct ProofIdentity {uint64_t cookie=0;void* lease=nullptr;};
enum class ProofAcquire {not_candidate, acquired, unknown};
struct ProofIdentityOps {
    void* context=nullptr;
    ProofAcquire (*acquire)(void*,void*,bool,ProofIdentity&) noexcept=nullptr;
    void (*release)(void*,ProofIdentity&) noexcept=nullptr;
};
struct ColorStateObservation {
    bool enabled=false,known=false;
    ColorProofReason reason=ColorProofReason::disabled;
    uint64_t generation=0,resource_cookie=0,local_ordinal=0;
    TransitionObservation transition{};
};
inline constexpr char kDlssColorStateScope[]="legacy-direct-reviewed-v1";
inline constexpr char kDlssColorStateShadowScope[]="legacy-direct-shadow-v1";
inline bool dlss_color_state_shadow_scope(std::string_view value) noexcept {
    return value==kDlssColorStateShadowScope;
}

inline bool dlss_color_state_scope(std::string_view value) noexcept {
    return value==kDlssColorStateScope || dlss_color_state_shadow_scope(value);
}
// Only the reviewed Cyberpunk profile is enabled on an ordinary launch.
// An explicit environment selection takes precedence, including disable/unknown.
enum class ColorStateScopeMode {disabled, reviewed, shadow};
inline bool dlss_color_default_profile(std::wstring_view executable) noexcept {
    const auto separator=executable.find_last_of(L"\\/");
    const auto name=separator==std::wstring_view::npos?executable:executable.substr(separator+1);
    constexpr std::wstring_view expected=L"cyberpunk2077.exe";
    if(name.size()!=expected.size()) return false;
    for(size_t i=0;i<name.size();++i) {
        const auto ch=name[i]>=L'A' && name[i]<=L'Z'?name[i]+(L'a'-L'A'):name[i];
        if(ch!=expected[i]) return false;
    }
    return true;
}
inline ColorStateScopeMode dlss_color_select_scope(bool override_present,
        std::string_view override_value,std::wstring_view executable) noexcept {
    if(override_present) {
        if(override_value==kDlssColorStateScope) return ColorStateScopeMode::reviewed;
        if(dlss_color_state_shadow_scope(override_value)) return ColorStateScopeMode::shadow;
        return ColorStateScopeMode::disabled;
    }
    return dlss_color_default_profile(executable)?ColorStateScopeMode::reviewed:
        ColorStateScopeMode::disabled;
}
inline bool dlss_color_candidate(const D3D12_RESOURCE_DESC& d) noexcept {
    return d.Dimension==D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
        d.Format==DXGI_FORMAT_R11G11B10_FLOAT && d.Width==1280 && d.Height==720 &&
        d.DepthOrArraySize==1 && d.MipLevels==1 && d.SampleDesc.Count==1;
}
struct SyncObservation {
    uint64_t generation=0, evaluate_sequence=0;
    uint64_t submit_batch_before=0;
    bool reset_observed=false, closed=false, incomplete=false;
    bool oversized_barrier=false;
    TransitionObservation color{}, motion{};
    ColorStateObservation color_state{};
};
inline bool legacy_dlss_color_source(const SyncObservation& s) noexcept {
    return s.reset_observed && !s.closed && !s.oversized_barrier && s.color.seen &&
        s.color.flags==D3D12_RESOURCE_BARRIER_FLAG_NONE &&
        s.color.subresource==D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES &&
        s.color.after==D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE &&
        s.evaluate_sequence>s.color.sequence && s.evaluate_sequence-s.color.sequence<=64;
}
inline bool dlss_color_source(const SyncObservation& s,bool implementation_known,
                              bool shadow_only=false) noexcept {
    // Shadow diagnostics retain the previously working gate, including its age bound.
    if(shadow_only || !s.color_state.enabled) return legacy_dlss_color_source(s);
    const auto& p=s.color_state;const auto& t=p.transition;
    return implementation_known && s.reset_observed && !s.closed && !s.oversized_barrier &&
        p.known && p.reason==ColorProofReason::ready && p.generation==s.generation &&
        p.resource_cookie && p.local_ordinal &&
        t.seen && t.flags==D3D12_RESOURCE_BARRIER_FLAG_NONE &&
        t.subresource==D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES &&
        t.after==D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE && s.evaluate_sequence>t.sequence;
}
// Extend the existing accepted source path; a diagnostic tracking miss must
// not make all previously valid frames disappear. The new proof is needed
// only where unrelated command traffic aged the old observation out.
inline bool dlss_color_processing_source(const SyncObservation& s,
        bool implementation_known,bool shadow_only=false) noexcept {
    return legacy_dlss_color_source(s) ||
        (!shadow_only && dlss_color_source(s,implementation_known));
}
struct SubmitObservation {
    uint64_t generation=0, batch=0, latest_evaluate_sequence=0;
    uint32_t evaluations=0;
    bool correlated=false;
};

class SyncProbe {
public:
    explicit SyncProbe(ProofIdentityOps ops={},bool color_state_enabled=false) noexcept;
    ~SyncProbe();
    bool color_state_enabled() const noexcept {return color_state_enabled_;}
    void unknown_recording(ID3D12GraphicsCommandList* list);
    void reset(ID3D12GraphicsCommandList* list, bool succeeded);
    void close(ID3D12GraphicsCommandList* list, bool succeeded);
    void barriers(ID3D12GraphicsCommandList* list, UINT count,
                  const D3D12_RESOURCE_BARRIER* barriers);
    SyncObservation evaluate(ID3D12GraphicsCommandList* list,
                             ID3D12Resource* color, ID3D12Resource* motion,
                             bool dlss_scope=true);
    SubmitObservation submit(ID3D12CommandList* list, uint64_t batch);
private:
    static constexpr size_t kLists=64, kTransitions=256, kPending=16;
    static constexpr size_t kColorStates=4;
    struct ColorState {
        ID3D12Resource* object=nullptr;
        ProofIdentity identity{};
        TransitionObservation transition{};
        uint64_t local_ordinal=0;
        ColorProofReason rejected=ColorProofReason::ready;
    };
    struct List {
        ID3D12CommandList* object=nullptr;
        uint64_t generation=0, touched=0, reset_sequence=0, submit_batch=0;
        bool closed=false, incomplete=false, oversized_barrier=false;
        ProofIdentity identity{};
        uint64_t local_ordinal=0;
        ColorProofReason rejected=ColorProofReason::ready;
        std::array<ColorState,kColorStates> colors{};
    };
    struct Transition {
        ID3D12CommandList* list=nullptr;
        ID3D12Resource* resource=nullptr;
        uint64_t generation=0, sequence=0;
        D3D12_RESOURCE_STATES before{}, after{};
        D3D12_RESOURCE_BARRIER_FLAGS flags{};
        UINT subresource=0;
    };
    struct Pending {
        ID3D12CommandList* list=nullptr;
        uint64_t generation=0, evaluate_sequence=0;
    };
    List* find(ID3D12CommandList* list);
    List& allocate(ID3D12CommandList* list);
    TransitionObservation latest(const List&, ID3D12Resource*) const;
    ProofAcquire acquire(void*,bool,ProofIdentity&) noexcept;
    void release(ProofIdentity&) noexcept;
    void clear_proof(List&) noexcept;
    void invalidate_all(ColorProofReason) noexcept;
    void observe_color_barrier(List*,const D3D12_RESOURCE_BARRIER&,uint64_t);
    ColorStateObservation color_state(List*,ID3D12Resource*);
    std::mutex mutex_;
    std::array<List,kLists> lists_{};
    std::array<Transition,kTransitions> transitions_{};
    std::array<Pending,kPending> pending_{};
    uint64_t sequence_=0, generation_=0;
    ProofIdentityOps identities_{};
    bool color_state_enabled_=false;
};
}
