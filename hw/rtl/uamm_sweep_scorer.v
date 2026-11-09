// UAMM-2 sweep scorer: hardware "heat classifier" for the block table.
//
// Replaces the single-shot uamm_controller of UAMM-1:
//   * no divider: weights are integers summing to 256 (154/64/38 ~= 60%/25%/15%), score = sum >> 8
//   * block metadata lives in a small SRAM; one sweep streams every block through a 2-stage pipeline
//     (1 block/cycle) so thousands of blocks are scored per planner interval
//   * `protected` (sink / recent window) is honoured in hardware, matching the Python planner
//
// Metadata word (25 bits): {protected, frequency[7:0], recency[7:0], importance[7:0]}
// Output tier code: 2'b00 hot (T0/T1), 2'b01 warm (T2), 2'b10 cold (T3).
// Synthesizable Verilog-2001.
module uamm_sweep_scorer #(
    parameter NBLK = 64,
    parameter AW   = 6            // AW = log2(NBLK)
) (
    input  wire          clk,
    input  wire          rst_n,
    // metadata write port
    input  wire          we,
    input  wire [AW-1:0] waddr,
    input  wire [24:0]   wdata,
    // sweep control
    input  wire          start,       // ignored while busy
    input  wire [7:0]    pressure,    // memory pressure, latched at start
    output reg           busy,
    output reg           done,        // pulses with the last out_valid of a sweep
    // result stream
    output reg           out_valid,
    output reg  [AW-1:0] out_idx,
    output reg  [7:0]    out_score,
    output reg  [1:0]    out_tier,
    output reg           out_evict
);
    localparam [7:0] W_IMP   = 8'd154;
    localparam [7:0] W_REC   = 8'd64;
    localparam [7:0] W_FRQ   = 8'd38;
    localparam [7:0] TH_HOT  = 8'd174;
    localparam [7:0] TH_WARM = 8'd89;
    localparam [7:0] EVICT_P = 8'd243;

    reg [24:0] mem [0:NBLK-1];
    always @(posedge clk) if (we) mem[waddr] <= wdata;

    reg [AW:0]    rd_cnt;
    reg [7:0]     pres;
    reg           s0_v;
    reg [AW-1:0]  s0_idx;
    reg [24:0]    s0_data;

    // synchronous SRAM read, address = issue counter
    always @(posedge clk) s0_data <= mem[rd_cnt[AW-1:0]];

    wire [7:0]  imp   = s0_data[7:0];
    wire [7:0]  rec   = s0_data[15:8];
    wire [7:0]  frq   = s0_data[23:16];
    wire        prot  = s0_data[24];
    /* verilator lint_off UNUSEDSIGNAL */
    wire [15:0] wsum  = (imp * W_IMP) + (rec * W_REC) + (frq * W_FRQ);  // <= 255*256
    /* verilator lint_on UNUSEDSIGNAL */
    wire [7:0]  score = wsum[15:8];   // wsum[7:0] intentionally dropped (>> 8)
    wire [1:0]  tier  = prot ? 2'b00 :
                        (score >= TH_HOT)  ? 2'b00 :
                        (score >= TH_WARM) ? 2'b01 : 2'b10;
    wire        evict = (!prot) && (pres >= EVICT_P) && (score < TH_WARM);
    localparam [31:0] LAST_IDX = NBLK - 1;
    wire        last  = s0_v && ({{(32-AW){1'b0}}, s0_idx} == LAST_IDX);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;  done <= 1'b0;  rd_cnt <= {(AW+1){1'b0}};  pres <= 8'd0;
            s0_v <= 1'b0;  s0_idx <= {AW{1'b0}};
            out_valid <= 1'b0;  out_idx <= {AW{1'b0}};  out_score <= 8'd0;
            out_tier <= 2'b10;  out_evict <= 1'b0;
        end else begin
            // ---- stage 1: score the block read last cycle
            out_valid <= s0_v;
            done      <= last;
            if (s0_v) begin
                out_idx   <= s0_idx;
                out_score <= score;
                out_tier  <= tier;
                out_evict <= evict;
            end
            // ---- stage 0: issue reads
            if (start && !busy) begin
                busy <= 1'b1;  rd_cnt <= {(AW+1){1'b0}};  pres <= pressure;  s0_v <= 1'b0;
            end else if (busy && (rd_cnt < NBLK)) begin
                s0_v   <= 1'b1;
                s0_idx <= rd_cnt[AW-1:0];
                rd_cnt <= rd_cnt + 1'b1;
            end else begin
                s0_v <= 1'b0;
            end
            if (last) busy <= 1'b0;
        end
    end
endmodule
