#include "../src/nr_tail_probe.h"

#include <cstddef>
#include <cstdio>

namespace {

// The probe only compares these addresses; it never dereferences them as COM
// objects. This keeps the test offline and independent of a D3D12 device.
alignas(16) unsigned char list_identity[16]{};
alignas(16) unsigned char color_identity[16]{};
alignas(16) unsigned char output_identity[16]{};

ID3D12GraphicsCommandList* test_list() {
    return reinterpret_cast<ID3D12GraphicsCommandList*>(list_identity);
}

ID3D12Resource* test_color() {
    return reinterpret_cast<ID3D12Resource*>(color_identity);
}

ID3D12Resource* test_output() {
    return reinterpret_cast<ID3D12Resource*>(output_identity);
}

bool expect(const char* label, unsigned actual, unsigned wanted) {
    if (actual == wanted) return true;
    std::printf("FAIL %-42s expected %u, got %u\n", label, wanted, actual);
    return false;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource,
                                  D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after,
                                  D3D12_RESOURCE_BARRIER_FLAGS flags =
                                      D3D12_RESOURCE_BARRIER_FLAG_NONE) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = flags;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

bool test_evaluate_phases_and_ordinary_srv() {
    auto* list = test_list();
    nrb::tail_reset(list);

    nrb::tail_action(list);
    nrb::tail_action(list);
    nrb::tail_copy(list, test_color());
    nrb::tail_before_evaluate(list, test_color(), test_output());
    nrb::tail_action(list);
    nrb::tail_action(list);
    nrb::tail_copy(list, test_color());
    nrb::tail_after_evaluate(list);
    nrb::tail_action(list);

    auto color_barrier = transition(
        test_color(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    auto output_write_barrier = transition(
        test_output(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto output_read_barrier = transition(
        test_output(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    nrb::tail_barriers(list, 1, &color_barrier);
    nrb::tail_barriers(list, 1, &output_write_barrier);
    nrb::tail_barriers(list, 1, &output_read_barrier);
    nrb::tail_action(list);
    nrb::tail_copy(list, test_output());

    const auto result = nrb::tail_close(list);
    bool ok = true;
    if (!result.valid) {
        std::puts("FAIL ordinary transition sample was not valid");
        return false;
    }
    ok &= expect("pre-Evaluate actions including copy", result.pre_actions, 3);
    ok &= expect("Evaluate actions including copy", result.in_evaluate_actions, 3);
    ok &= expect("post-Evaluate actions including copy", result.after_actions, 3);
    ok &= expect("color transitions after Evaluate", result.after_color_barriers, 1);
    ok &= expect("output transitions after Evaluate", result.after_output_barriers, 2);
    ok &= expect("completed output-to-SRV transitions", result.output_read_barriers, 1);
    ok &= expect("actions after output-to-SRV", result.actions_after_output_read, 2);
    ok &= expect("copies sourced from output", result.after_output_copies, 1);
    ok &= expect("ordinary output transition flags", result.output_flags,
                 D3D12_RESOURCE_BARRIER_FLAG_NONE);
    ok &= expect("ordinary output subresource", result.output_subresource,
                 D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
    if (ok) std::puts("PASS: Evaluate phases, ordinary SRV transition, and copy counts.");
    return ok;
}

bool test_split_srv_transition() {
    auto* list = test_list();
    nrb::tail_reset(list);
    nrb::tail_before_evaluate(list, test_color(), test_output());
    nrb::tail_after_evaluate(list);

    // For a split transition, BEGIN_ONLY schedules the transition. The
    // resource does not reach PIXEL_SHADER_RESOURCE until END_ONLY executes.
    auto begin = transition(
        test_output(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY);
    auto end = transition(
        test_output(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_BARRIER_FLAG_END_ONLY);

    nrb::tail_barriers(list, 1, &begin);
    nrb::tail_action(list); // Must not count as after the completed SRV transition.
    nrb::tail_barriers(list, 1, &end);
    nrb::tail_action(list); // Must count.

    const auto result = nrb::tail_close(list);
    bool ok = true;
    if (!result.valid) {
        std::puts("FAIL split-transition sample was not valid");
        return false;
    }
    ok &= expect("split output barriers", result.after_output_barriers, 2);
    ok &= expect("completed split output-to-SRV transitions",
                 result.output_read_barriers, 1);
    ok &= expect("actions after completed split transition",
                 result.actions_after_output_read, 1);
    ok &= expect("completed split transition flags", result.output_flags,
                 D3D12_RESOURCE_BARRIER_FLAG_END_ONLY);
    if (ok) std::puts("PASS: split SRV transition ordering.");
    return ok;
}

bool test_route_switch_discards_old_tail() {
    auto* list=test_list();
    nrb::tail_probe_rearm(NRB_ROUTE_XESS);
    nrb::tail_reset(list);
    nrb::tail_before_evaluate(list,test_color(),test_output(),NRB_ROUTE_XESS);
    nrb::tail_after_evaluate(list);
    auto result=nrb::tail_close(list);
    if(!result.valid || result.route!=NRB_ROUTE_XESS) return false;
    nrb::tail_probe_rearm(NRB_ROUTE_FSR);
    nrb::tail_reset(list);
    nrb::tail_before_evaluate(list,test_color(),test_output(),NRB_ROUTE_XESS);
    if(nrb::tail_close(list).valid) return false;
    nrb::tail_before_evaluate(list,test_color(),test_output(),NRB_ROUTE_FSR);
    nrb::tail_after_evaluate(list);
    result=nrb::tail_close(list);
    if(!result.valid || result.route!=NRB_ROUTE_FSR) return false;
    for(unsigned i=1;i<8;++i) {
        nrb::tail_reset(list);
        nrb::tail_before_evaluate(list,test_color(),test_output(),NRB_ROUTE_FSR);
        nrb::tail_after_evaluate(list);
        if(!nrb::tail_close(list).valid) return false;
    }
    if(nrb::tail_probe_active()) return false;
    nrb::tail_probe_rearm(NRB_ROUTE_XESS);
    if(!nrb::tail_probe_active()) return false;
    nrb::tail_reset(list);
    nrb::tail_before_evaluate(list,test_color(),test_output(),NRB_ROUTE_XESS);
    nrb::tail_after_evaluate(list);
    result=nrb::tail_close(list);
    return result.valid && result.route==NRB_ROUTE_XESS;
}

bool test_xess_copy_tail_gate() {
    nrb::TailSummary sample{};
    sample.valid=true;
    sample.route=NRB_ROUTE_XESS;
    sample.after_actions=1;
    sample.after_copy_from_color=1;
    sample.after_color_barriers=1;
    sample.after_output_barriers=1;
    sample.color_before=2240;
    sample.color_after=2112;
    sample.output_before=8;
    sample.output_after=192;
    sample.color_subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    sample.output_subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    sample.copy_source_format=DXGI_FORMAT_R11G11B10_FLOAT;
    sample.copy_dest_format=DXGI_FORMAT_R11G11B10_FLOAT;
    sample.copy_source_width=sample.copy_dest_width=1280;
    sample.copy_source_height=sample.copy_dest_height=720;
    if(!nrb::tail_safe_for_deferred_identity(sample)) return false;
    sample.after_copy_to_output=1;
    if(nrb::tail_safe_for_deferred_identity(sample)) return false;
    sample.after_copy_to_output=0;
    sample.after_dispatches=1;
    if(nrb::tail_safe_for_deferred_identity(sample)) return false;
    sample.after_dispatches=0;
    sample.copy_dest_width=1920;
    if(nrb::tail_safe_for_deferred_identity(sample)) return false;
    sample.copy_dest_width=1280;
    sample.route=NRB_ROUTE_FSR;
    return !nrb::tail_safe_for_deferred_identity(sample);
}

} // namespace

int main() {
    nrb::tail_probe_rearm(NRB_ROUTE_DLSS);
    const bool ordinary_ok = test_evaluate_phases_and_ordinary_srv();
    const bool split_ok = test_split_srv_transition();
    const bool route_ok = test_route_switch_discards_old_tail();
    const bool xess_gate_ok = test_xess_copy_tail_gate();
    if (ordinary_ok && split_ok && route_ok && xess_gate_ok) {
        std::puts("PASS: tail probe action counters match both transition cases.");
        return 0;
    }
    std::puts("FAIL: tail probe counters do not match the D3D12 transition semantics.");
    return 1;
}
