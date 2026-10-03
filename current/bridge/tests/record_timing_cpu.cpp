#include "../src/nr_record_timing.h"
#include <limits>
#include <cstdio>

int main() {
    unsigned checks=0;
    const auto check=[&](bool condition) {++checks;if(!condition) {std::fprintf(stderr,"check %u failed\n",checks);return false;}return true;};
    nrb::RecordTimingWindow window;
    NRB_RecordTimes row{};row.abi_size=sizeof(row);
    if(!check(!window.snapshot(&row) && row.frames==0)) return 1;
    if(!check(!window.snapshot(nullptr))) return 1;
    row.abi_size=0;
    if(!check(!window.snapshot(&row) && row.abi_size==0)) return 1;
    row.abi_size=sizeof(row);
    if(!check(window.append(1,{10,2,3,1,7,60}))) return 1;
    if(!check(window.snapshot(&row) && row.frames==1 && row.last_nr_frame_id==1 && row.record_average_ms==10)) return 1;
    if(!check(!window.append(1,{10,2,3,1,7,60}))) return 1;
    if(!check(!window.append(2,{10,2,-1,1,7,60}))) return 1;
    if(!check(!window.append(2,{10,2,3,1,7,std::numeric_limits<double>::quiet_NaN()}))) return 1;
    if(!check(!window.append(2,{10,8,3,1,7,60}))) return 1;
    if(!check(!window.append(2,{10,2,3,1,7,12}))) return 1;
    if(!check(window.snapshot(&row) && row.frames==1 && row.last_nr_frame_id==1)) return 1;
    window.reset();
    if(!check(!window.snapshot(&row) && row.frames==0 && row.record_average_ms==0)) return 1;
    for(unsigned i=1;i<=40;i++) if(!check(window.append(i,{double(i),0,0,0,2,double(i)+40}))) return 1;
    if(!check(window.snapshot(&row) && row.frames==40 && row.record_last_ms==40 && row.record_average_ms==24.5)) return 1;
    if(!check(row.record_to_submit_gap_average_ms==2 && row.record_to_retire_span_average_ms==64.5)) return 1;
    window.reset();
    if(!check(window.append(1,{1,0,0,0,1,5}))) return 1;
    std::printf("record_timing_cpu: %u checks passed; no GPU\n",checks);
    return 0;
}
