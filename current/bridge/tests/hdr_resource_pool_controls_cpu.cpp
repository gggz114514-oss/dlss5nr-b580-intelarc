#include "deferred_identity.h"
#include "nr_hdr_resource_pool.h"
#include "nr_hdr_resource_pool_api.h"
#include <cstdio>
#include <cstdlib>

void require(bool success,const char* text) {
    if(!success) {std::fprintf(stderr,"FAIL: %s\n",text);std::exit(1);}
}
int main() {
    nrb::HdrResourcePoolStats cpp;
    require(nrb::deferred_hdr_resource_pool_stats(cpp) && !cpp.enabled && !cpp.resident_slots,
        "actual deferred pool session defaults OFF without GPU creation");
    NRB_HdrCacheStats cache{};cache.abi_size=sizeof(cache);
    require(nrb::deferred_hdr_cache_stats(&cache) && cache.enabled && !cache.shader_compile_calls,
        "frozen cache+timing session default remains ON");
    require(nrb::deferred_hdr_set_resource_pool_enabled(true) && nrb::deferred_hdr_resource_pool_stats(cpp) && cpp.enabled,
        "direct C++ session setter/stats share ownership");
    nrb::deferred_hdr_set_cache_enabled(false);
    require(nrb::deferred_hdr_cache_stats(&cache) && !cache.enabled && nrb::deferred_hdr_resource_pool_stats(cpp) && cpp.enabled,
        "pipeline cache OFF does not disable resource pool");
    NRB_HdrResourcePoolStats abi{};abi.abi_size=sizeof(abi);
    require(NRB_GetHdrResourcePoolStats(&abi)==1 && abi.enabled && abi.capacity==2 &&
        abi.max_texture_bytes==64ull*1024*1024 && !abi.texture_bytes && !abi.resident_slots,
        "C ABI stats reflect the same bounded C++ session");
    require(NRB_SetHdrResourcePoolEnabled(0)==1 && nrb::deferred_hdr_resource_pool_stats(cpp) && !cpp.enabled,
        "C ABI OFF restores the direct C++ session");
    nrb::deferred_hdr_set_cache_enabled(true);
    require(nrb::deferred_hdr_cache_stats(&cache) && cache.enabled && NRB_GetHdrResourcePoolStats(&abi)==1 && !abi.enabled,
        "pipeline cache ON does not enable resource pool");
    abi.abi_size=0;
    require(NRB_GetHdrResourcePoolStats(&abi)==0 && NRB_GetHdrResourcePoolStats(nullptr)==0,
        "C ABI rejects missing/mismatched stat layout");
    std::puts("PASS: 8 actual deferred-session C++/C ABI control checks; cache ON and pool OFF restored; no GPU creation");
}
