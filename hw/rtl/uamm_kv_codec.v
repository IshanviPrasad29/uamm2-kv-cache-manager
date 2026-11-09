// UAMM-2 KV codec datapath: 8-bit -> 4-bit group quantizer and the inverse dequantizer.
// One beat = LANES values. Intended to sit on the AXI-stream path between DRAM/flash and Ara's
// vector load/store unit so (de)quantization happens in the load path, not in vector instructions.
//
// Per-group parameters are computed by software/firmware from the group's min/max (range = max-min):
//   zp       = min                       (8-bit)
//   inv_q12  = round(15*4096 / range)    (Q4.12, 16-bit; 0 when range == 0)
//   step_q8  = round(range*256 / 15)     (Q8.8,  16-bit; 0 when range == 0)
// Quantize:   q  = clamp( ((x - zp) * inv_q12 + 2048) >> 12 , 0, 15 )   (x < zp clamps to q = 0)
// Dequantize: x' = min( 255, zp + ((q * step_q8 + 128) >> 8) )
// Nibble i of the packed word is lane i (bits [4i+3:4i]) - the same packing as uamm2/codecs.py.
// Synthesizable Verilog-2001, one pipeline register.

module uamm_kv_quant4 #(parameter LANES = 16) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    input  wire [LANES*8-1:0]   x_flat,
    input  wire [7:0]           zp,
    input  wire [15:0]          inv_q12,
    output reg                  out_valid,
    output reg  [LANES*4-1:0]   q_flat
);
    reg [LANES*4-1:0] q_next;
    reg [7:0]  xi;
    reg [7:0]  dx;
    reg [31:0] prod;
    reg [31:0] r;
    integer i;
    always @* begin
        q_next = {(LANES*4){1'b0}};
        for (i = 0; i < LANES; i = i + 1) begin
            xi   = x_flat[i*8 +: 8];
            dx   = (xi >= zp) ? (xi - zp) : 8'd0;
            prod = dx * inv_q12;
            r    = (prod + 32'd2048) >> 12;
            q_next[i*4 +: 4] = (r > 32'd15) ? 4'd15 : r[3:0];
        end
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            q_flat    <= {(LANES*4){1'b0}};
        end else begin
            out_valid <= in_valid;
            if (in_valid) q_flat <= q_next;
        end
    end
endmodule

module uamm_kv_dequant4 #(parameter LANES = 16) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    input  wire [LANES*4-1:0]   q_flat,
    input  wire [7:0]           zp,
    input  wire [15:0]          step_q8,
    output reg                  out_valid,
    output reg  [LANES*8-1:0]   x_flat
);
    reg [LANES*8-1:0] x_next;
    reg [3:0]  qi;
    reg [31:0] prod;
    reg [31:0] sum;
    integer i;
    always @* begin
        x_next = {(LANES*8){1'b0}};
        for (i = 0; i < LANES; i = i + 1) begin
            qi   = q_flat[i*4 +: 4];
            prod = qi * step_q8;
            sum  = {24'd0, zp} + ((prod + 32'd128) >> 8);
            x_next[i*8 +: 8] = (sum > 32'd255) ? 8'd255 : sum[7:0];
        end
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            x_flat    <= {(LANES*8){1'b0}};
        end else begin
            out_valid <= in_valid;
            if (in_valid) x_flat <= x_next;
        end
    end
endmodule
