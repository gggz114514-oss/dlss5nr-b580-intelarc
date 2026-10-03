// Candidate for the Cyberpunk R11G11B10 -> SDR NR -> R11G11B10 path.
// The original scene-linear source is never clamped in the final composite.
Texture2D<float3> GameColor : register(t0);
// t1 is the game's RG16F motion during prepare and the NR color during composite.
// The descriptor changes only after the prepare fence has completed.
Texture2D<float4> MotionOrNR : register(t1);
RWTexture2D<float4> Proxy : register(u0);
RWTexture2D<float2> NRMotion : register(u1);
RWTexture2D<float3> XeSSInput : register(u2);
cbuffer Constants : register(b0) {
    uint Width; uint Height; uint UseGameMotion;
    float MotionScaleX; float MotionScaleY;
};

[numthreads(8,8,1)]
void prepare(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= Width || thread.y >= Height) return;
    uint2 xy = thread.xy;
    float3 scene = GameColor.Load(int3(xy, 0));
    Proxy[xy] = float4(saturate(scene), 1.0f);
    float2 pixels = float2(0.0f, 0.0f);
    if (UseGameMotion != 0) {
        pixels = MotionOrNR.Load(int3(xy, 0)).xy *
            float2(MotionScaleX, MotionScaleY);
        // The shared bridge consumes RG16F. Reject invalid values rather
        // than writing NaNs or infinities into its temporal reprojection.
        if (!all(isfinite(pixels))) pixels = float2(0.0f, 0.0f);
        pixels = clamp(pixels, -65504.0f, 65504.0f);
    }
    NRMotion[xy] = pixels;
}

[numthreads(8,8,1)]
void composite(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= Width || thread.y >= Height) return;
    uint2 xy = thread.xy;
    float3 scene = GameColor.Load(int3(xy, 0));
    float3 sdr = saturate(scene);
    float3 nr = MotionOrNR.Load(int3(xy, 0)).rgb;
    float peak = max(scene.r, max(scene.g, scene.b));
    float weight = saturate((1.2f - peak) / 0.4f);
    XeSSInput[xy] = max(scene + weight * (nr - sdr), 0.0f);
}
