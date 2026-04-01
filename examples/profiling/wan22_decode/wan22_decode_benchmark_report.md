# Wan2.2 Decode Benchmark Report

GPU: `NVIDIA RTX PRO 6000 Blackwell 96GB`  
Driver / CUDA: `570.195.03 / 12.8`

Workloads used in this report:

- Full streaming benchmark: saved `Wan2.2 TI2V-5B` latent with shape `48 x 31 x 44 x 80` and output `121` frames at `1280 x 704 @ 24 fps`
- Nsight representative clip: first `3` latent steps from the same sample, which decode to `9` output frames

## 1. lighttaew2_2 streaming decode timing

Source: [`lighttaew2_2_stream_benchmark.json`](/root/LatentsCompress/examples/streaming/lighttaew2_2_stream_benchmark.json)

Median run on the full `31`-step latent:

- first frame latency: `0.00581 s`
- steady 4-frame decode time: `0.00724 s` mean, `0.00717 s` median
- steady 4-frame decode jitter (CV): `0.0402`
- steady throughput: `552.63 fps`
- end-to-end throughput over the whole `121`-frame clip: `96.25 fps`

Interpretation:

- After warmup, `lighttaew2_2` is fast enough for a real streaming preview path.
- The first chunk emits `1` frame, then each additional latent step emits `4` frames.

## 2. Nsight Systems summary

Representative harness:

- [`profile_wan22_decode.py`](/root/LatentsCompress/scripts/profile_wan22_decode.py)
- `wan_vae_wrapper`, `3` latent steps, `warmup=1`, `profile=1`
- `lighttae_stream`, `3` latent steps, `warmup=1`, `profile=1`

Tracked profiler artifacts in this repository:

- exported decode-range JSON summaries from `nsys stats`

Note:

- the raw `.nsys-rep` captures were intentionally left out of git because the binary profiler dump embedded environment-specific sensitive strings
- the repository keeps the lightweight exported JSON summaries that support the conclusions below

Filtered decode-range stats:

- [`wan_vae_wrapper_3step_nsys_stats_run1_nvtx_sum_nvtx=decode-1.json`](/root/LatentsCompress/examples/profiling/wan22_decode/wan_vae_wrapper_3step_nsys_stats_run1_nvtx_sum_nvtx=decode-1.json)
- [`wan_vae_wrapper_3step_nsys_stats_run1_cuda_api_sum_nvtx=decode-1.json`](/root/LatentsCompress/examples/profiling/wan22_decode/wan_vae_wrapper_3step_nsys_stats_run1_cuda_api_sum_nvtx=decode-1.json)
- [`wan_vae_wrapper_3step_nsys_stats_run1_cuda_gpu_kern_sum_nvtx=decode-1.json`](/root/LatentsCompress/examples/profiling/wan22_decode/wan_vae_wrapper_3step_nsys_stats_run1_cuda_gpu_kern_sum_nvtx=decode-1.json)
- [`lighttae_stream_3step_nsys_stats_run1_nvtx_sum_nvtx=decode-1.json`](/root/LatentsCompress/examples/profiling/wan22_decode/lighttae_stream_3step_nsys_stats_run1_nvtx_sum_nvtx=decode-1.json)
- [`lighttae_stream_3step_nsys_stats_run1_cuda_api_sum_nvtx=decode-1.json`](/root/LatentsCompress/examples/profiling/wan22_decode/lighttae_stream_3step_nsys_stats_run1_cuda_api_sum_nvtx=decode-1.json)
- [`lighttae_stream_3step_nsys_stats_run1_cuda_gpu_kern_sum_nvtx=decode-1.json`](/root/LatentsCompress/examples/profiling/wan22_decode/lighttae_stream_3step_nsys_stats_run1_cuda_gpu_kern_sum_nvtx=decode-1.json)

Key readout:

- `Wan VAE` decode range: `1.312 s`
- `lighttae2 stream` decode range: `0.0180 s`
- The dominant GPU kernel in both cases is the same cuDNN BF16 Tensor Core implicit-GEMM kernel:
  `sm80_xmma_fprop_implicit_gemm_bf16bf16...cudnn`
- That kernel accounts for `57.2%` of VAE GPU kernel time and `61.1%` of TAE GPU kernel time.

Process-level interpretation from `nsys`:

- `Wan VAE` spends most of its decode wall time inside large Tensor Core convolution/GEMM kernels, with many launches (`2018` kernel instances in the decode range).
- `lighttae` streaming has the same dominant compute kernel, but much shorter total GPU work (`12.30 ms` summed kernel time in the decode range) and far fewer launches (`954` kernel instances).
- `lighttae` streaming also shows meaningful API-side `cudaMemcpyAsync` time, so the actual streaming path is a mix of compute and transfer/launch overhead, not pure compute.

## 3. Nsight Compute status

`ncu` could not collect hardware counter metrics in this environment.

Observed error:

- `ERR_NVGPUCTRPERM`

Observed driver setting:

- [`/proc/driver/nvidia/params`](/proc/driver/nvidia/params) reports `RmProfilingAdminOnly: 1`

Impact:

- I could not extract counter-based roofline metrics such as exact arithmetic intensity.
- I therefore cannot honestly claim a counter-verified `compute bound` or `memory bound` label from `ncu` on this machine.

Best-effort inference from `nsys`, clearly marked as inference:

- The dominant cuDNN BF16 Tensor Core implicit-GEMM kernel is likely compute-heavy.
- The overall `Wan VAE` decode path behaves more like a compute-dominated convolution stack.
- The overall `lighttae` streaming path is more mixed, because transfer and launch overhead are visible enough to matter at process level.

## 4. VAE batch scaling

Representative clip: first `3` latent steps only.

Observed median latencies:

| batch | wrapper decode (s) | direct batched decode (s) | wrapper vs B1 | direct vs B1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | `1.2739` | `1.2754` | `1.00x` | `1.00x` |
| 2 | `2.5474` | `2.5469` | `2.00x` | `2.00x` |
| 4 | `5.0892` | `5.0616` | `3.99x` | `3.97x` |

Interpretation:

- The official wrapper path is expected to scale almost linearly because it literally loops over the list of samples in Python, see [`vae2_2.py`](/root/Wan2.2/wan/modules/vae2_2.py#L1038).
- On this machine and this clip, even the direct batched call to `vae.model.decode()` was still almost linear in latency.
- So for the current `Wan2.2_VAE` implementation, batching multiple decode jobs should be treated as near-linear in latency unless we later prove otherwise with a more specialized fused path.
