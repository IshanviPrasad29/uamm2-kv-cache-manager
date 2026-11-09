from data import *
def build(R):
    R.h1("14. Conclusion")
    R.p("Running language models on edge processors is limited by memory: the KV cache grows with every token, must be read in full by standard attention, and competes with the weights for a few gigabytes of shared DRAM. The research community has produced strong but partial answers: block paging, attention-based eviction, query-aware selection, low-bit quantisation, offloading and compression engines. Each leaves a dimension of the problem open.", align="j")
    R.p("UAMM-2 combines them around one idea: a block's placement is a (tier, codec) pair chosen together from measured importance, measured error and a priced cost, with a precision ratchet, hysteresis and query-aware reads. It adds three verified RTL blocks, a fused vector kernel for the Ara vector processor and a runtime seam for compiler-based runtimes, and keeps the codec interface open for external compression engines.", align="j")
    R.p("On a controlled synthetic workload with assumed memory costs, the combined design keeps every token within a budget that makes the earlier threshold policy delete 63 % of them, retains 0.94 of the attention mass instead of 0.76, and reduces modelled fetch time by about 8× at a cost of bounded quantisation error. It degrades gracefully as the budget is stretched, and its sensitivity to each knob has been measured. The RTL is bit-exact against golden models and every injected fault was caught.", align="j")
    R.p("What remains is to replace the assumptions with measurements: real model activations, real memory costs, execution of the vector kernel on Ara or a simulator and integration with a deployed runtime. Chapter 12 lists these steps with a proposed order. The design, the code and the experiments are organised so that each of those substitutions is a local change.", align="j")
    import os
    R.h2("Code availability")
    R.p("The complete source code of UAMM-1 and UAMM-2, the experiment scripts and raw results, the RTL with its testbenches and waveform dumps, and the scripts that regenerate every figure and table in this report are available at " + os.environ.get("REPO_URL", "https://github.com/<username>/<repository>") + ".", align="j")
    R.h1("References")
    refs = [
     "Kwon, W. et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023. arXiv:2309.06180.",
     "Zhang, Z. et al. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. NeurIPS 2023. arXiv:2306.14048.",
     "Liu, Z. et al. Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time. NeurIPS 2023. arXiv:2305.17118.",
     "Xiao, G. et al. Efficient Streaming Language Models with Attention Sinks. ICLR 2024. arXiv:2309.17453.",
     "Li, Y. et al. SnapKV: LLM Knows What You are Looking for Before Generation. 2024. arXiv:2404.14469.",
     "Ge, S. et al. Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs (FastGen). 2023. arXiv:2310.01801.",
     "Cai, Z. et al. PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling. 2024. arXiv:2406.02069.",
     "Tang, J. et al. Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference. ICML 2024. arXiv:2406.10774.",
     "Lee, W. et al. InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management. OSDI 2024. arXiv:2406.19707.",
     "Liu, Z. et al. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. ICML 2024. arXiv:2402.02750.",
     "Hooper, C. et al. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization. NeurIPS 2024. arXiv:2401.18079.",
     "Chang, C.-C. et al. Palu: Compressing KV-Cache with Low-Rank Projection. 2024. arXiv:2407.21118.",
     "Sheng, Y. et al. FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU. ICML 2023. arXiv:2303.06865.",
     "Alizadeh, K. et al. LLM in a flash: Efficient Large Language Model Inference with Limited Memory. 2023. arXiv:2312.11514.",
     "Shazeer, N. Fast Transformer Decoding: One Write-Head is All You Need. 2019. arXiv:1911.02150.",
     "Ainslie, J. et al. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. 2023. arXiv:2305.13245.",
     "DeepSeek-AI. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. 2024. arXiv:2405.04434.",
     "Dao, T. et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022. arXiv:2205.14135.",
     "Xiao, G. et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML 2023. arXiv:2211.10438.",
     "Cavalcante, M. et al. Ara: A 1-GHz+ Scalable and Energy-Efficient RISC-V Vector Processor with Multiprecision Floating-Point Support in 22-nm FD-SOI. IEEE TVLSI, 2020.",
     "Perotti, M. et al. Ara2: Exploring Single- and Multi-Core Vector Processing with an Efficient RVV 1.0 Compliant Open-Source Processor. 2023. arXiv:2311.07493.",
     "Chen, T. et al. TVM: An Automated End-to-End Optimizing Compiler for Deep Learning. OSDI 2018. arXiv:1802.04799.",
     "EvicPress (arXiv:2512.14946); Ye et al., KV-Compress (arXiv:2410.00161); PagedEviction (arXiv:2509.04377); DiffKV (arXiv:2412.03131); TensorRT-LLM KV-cache compression framework; CXL-SpecKV; HiKV (arXiv:2607.22389); Titanus (arXiv:2505.17787); Kelle (arXiv:2510.16040). Recent preprints; details to be re-verified against the latest versions before formal citation.",
     "RISC-V International. RISC-V Vector Extension, version 1.0.",
     "Verilator, Yosys and GTKWave open-source tools used for simulation, synthesis checking and waveform viewing."]
    R.numbered(refs)
