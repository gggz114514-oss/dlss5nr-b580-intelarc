#pragma once
#include "nr_bridge.h"
#include <array>
#include <algorithm>
#include <cmath>
namespace nrb {
class RecordTimingWindow {
    std::array<std::array<double,32>,6> rows_{};
    std::array<double,6> sums_{},last_{};
    uint64_t count_=0, frame_=0;
public:
    void reset() {rows_={}; sums_={}; last_={}; count_=frame_=0;}
    bool append(uint64_t frame,const std::array<double,6>& values) {
        if(!frame || frame<=frame_) return false;
        for(double v:values) if(!std::isfinite(v) || v<0) return false;
        if(values[1]+values[2]+values[3]>values[0]+0.001 ||
           values[4]+values[0]>values[5]+0.001) return false;
        const size_t slot=count_%32;
        for(size_t i=0;i<6;i++) {
            if(count_>=32) sums_[i]-=rows_[i][slot];
            rows_[i][slot]=values[i]; sums_[i]+=values[i]; last_[i]=values[i];
        }
        ++count_; frame_=frame; return true;
    }
    bool snapshot(NRB_RecordTimes* times) const {
        if(!times || times->abi_size!=sizeof(NRB_RecordTimes)) return false;
        NRB_RecordTimes out{};out.abi_size=sizeof(out);
        out.frames=count_;out.last_nr_frame_id=frame_;
        if(count_) {
            const double samples=double(std::min<uint64_t>(32,count_));
        out.record_last_ms=last_[0]; out.record_average_ms=sums_[0]/samples;
        out.resources_last_ms=last_[1]; out.resources_average_ms=sums_[1]/samples;
        out.hdr_initialize_last_ms=last_[2]; out.hdr_initialize_average_ms=sums_[2]/samples;
        out.xess_record_last_ms=last_[3]; out.xess_record_average_ms=sums_[3]/samples;
        out.record_to_submit_gap_last_ms=last_[4]; out.record_to_submit_gap_average_ms=sums_[4]/samples;
        out.record_to_retire_span_last_ms=last_[5]; out.record_to_retire_span_average_ms=sums_[5]/samples;
        }
        *times=out;return count_>0;
    }
};
}
