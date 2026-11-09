from data import *
def build(R):
    R.h1("10. The Ara Vector Kernel: Results")
    R.p("The vector-processor part of the system is a C kernel (`ara/kv_kernels.c`) that computes attention scores directly on 4-bit packed keys. It exists in two variants that share data layout and algorithm: a scalar path that executes on any host, and an RVV 1.0 path written with the standard vector intrinsics, compiled for a RISC-V vector target.", align="j")
    R.h2("10.1 Algorithm in detail")
    R.p("For a block of T tokens and d channels with per-channel scale s_c and zero z_c, the key of token t is reconstructed as k_{t,c} = Q_{t,c}·s_c + z_c. The attention logit is therefore:", align="j")
    R.eq("q·k_t = Σ_c q_c (Q_{t,c} s_c + z_c) = Σ_c (q_c s_c) Q_{t,c} + Σ_c q_c z_c = Σ_c q̃_c Q_{t,c} + b")
    R.p("The pre-scaled query q̃_c = q_c·s_c and the bias b = Σ q_c z_c are computed once per block (`kv_qk_prepare`), and q̃ is de-interleaved into even-channel and odd-channel halves so that each packed byte (low nibble = even channel, high nibble = odd channel) multiplies two contiguous vectors. The per-token loop then does only integer-to-float conversion and fused multiply–add:", align="j")
    R.table(["Step per token", "Scalar C", "RVV instruction (observed in the compiled output)"],
            [["Load packed bytes", "`row[j]`", "`vle8.v` (byte load with `vsetvli`)"],
             ["Extract low / high nibble", "`& 0x0F`, `>> 4`", "`vand.vx`, `vsrl.vi`"],
             ["Widen to 32 bits", "implicit", "`vzext.vf4`"],
             ["Convert to float", "cast", "`vfcvt.f.xu.v`"],
             ["Load pre-scaled query halves", "`qs_even[j]`, `qs_odd[j]`", "`vle32.v`"],
             ["Accumulate", "`acc += …`", "`vfmacc.vv` with tail-undisturbed policy"],
             ["Reduce across lanes", "sequential sum", "`vfredusum.vs`"],
             ["Add bias", "`+ bias`", "scalar add after `vfmv.f.s`"]],
            caption="Mapping from the scalar algorithm to the RVV instructions.", widths=[1.8, 1.9, 2.9])
    R.p("The tail-undisturbed (`_tu`) form of the multiply–add matters for correctness: when the last strip of a row has fewer than a full vector of elements, the accumulator lanes beyond the active length must keep their previous values until the final reduction. Using the default tail-agnostic form could let those lanes be overwritten with arbitrary values.", align="j")
    R.h2("10.2 Numerical agreement (host)")
    R.fig("t07_ara_check.png", "Scalar C kernel versus NumPy dequantise-then-dot on random blocks, and the RVV compile check.", 6.4)
    R.p("On four block shapes the maximum absolute difference between the C kernel and the straightforward NumPy computation (dequantise the whole block, then take the dot product) is between 4.8 × 10⁻⁶ and 2.0 × 10⁻⁵, which is floating-point rounding for sums of 64–128 products of magnitude about one. The test suite also checks the pre-scaling algebra and the nibble order.", align="j")
    R.h2("10.3 RISC-V Vector compile check")
    R.p("The same source was compiled with clang for `riscv64-unknown-elf` with `-march=rv64gcv -O2` (object file only). The inner loop of the compiled code is shown below; it contains the expected vector instructions listed in the mapping table above.", align="j")
    R.fig("t05_rvv_asm.png", "RVV assembly for the 4-bit key kernel (excerpt of the compiler output).", 6.4)
    R.p("This demonstrates that the intrinsics lower to the intended instruction sequence for a vector target. **It does not demonstrate that the code runs correctly or fast on Ara:** no Ara, Spike or QEMU execution has been performed (Chapter 12).", align="j")
    R.h2("10.4 Why a fused kernel: bandwidth arithmetic")
    R.p("An unfused implementation would first dequantise the key block into a floating-point buffer and then compute dot products. If that buffer is staged in memory, each 4-bit key element costs 0.5 bytes to read, 4 bytes to write and 4 bytes to read back (8.5 bytes of traffic) versus 0.5 bytes in the fused kernel, a factor of 17 in traffic for the key stream. If the dequantised tile is kept entirely in registers the memory traffic is the same as the fused kernel's but the kernel then executes the extra dequantisation instructions (a multiply and add per element) before each multiply–add. The fused form removes both. These are analytical counts from the algorithm, not measurements.", align="j")
    R.callout("What would make this a measured result", "Run the kernel on Spike for functional verification, then on the Ara RTL simulation or an FPGA build to obtain cycles per token for block sizes of 16–64 tokens and head dimensions of 64–128; compare with a dequantise-then-dot baseline written with the same intrinsics, on the same data.")
