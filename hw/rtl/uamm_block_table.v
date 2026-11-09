// UAMM-2 block-table translation unit (logical block -> tier / codec / physical slot).
// Inspired by the address-translation stage of FPGA KV-cache engines; deliberately minimal:
// direct-mapped table, 1-cycle synchronous lookup, no TLB.
//   entry = {tier[1:0], codec[1:0], phys[PW-1:0]} plus a separate valid bit (reset to 0).
// A read issued in the same cycle as a write to the same address returns the OLD entry.
// Synthesizable Verilog-2001.
module uamm_block_table #(
    parameter AW = 8,
    parameter PW = 12
) (
    input  wire          clk,
    input  wire          rst_n,
    // update port
    input  wire          we,
    input  wire [AW-1:0] waddr,
    input  wire          w_valid,
    input  wire [1:0]    w_tier,
    input  wire [1:0]    w_codec,
    input  wire [PW-1:0] w_phys,
    // lookup port
    input  wire          rd_en,
    input  wire [AW-1:0] raddr,
    output reg           rd_valid,
    output reg           hit,
    output reg  [1:0]    tier,
    output reg  [1:0]    codec,
    output reg  [PW-1:0] phys
);
    localparam DEPTH = (1 << AW);
    reg [PW+3:0]    mem   [0:DEPTH-1];
    reg [DEPTH-1:0] vbits;

    always @(posedge clk) if (we) mem[waddr] <= {w_tier, w_codec, w_phys};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vbits    <= {DEPTH{1'b0}};
            rd_valid <= 1'b0;
            hit      <= 1'b0;
            tier     <= 2'd0;
            codec    <= 2'd0;
            phys     <= {PW{1'b0}};
        end else begin
            if (we) vbits[waddr] <= w_valid;
            rd_valid <= rd_en;
            if (rd_en) begin
                hit   <= vbits[raddr];
                tier  <= mem[raddr][PW+3:PW+2];
                codec <= mem[raddr][PW+1:PW];
                phys  <= mem[raddr][PW-1:0];
            end
        end
    end
endmodule
