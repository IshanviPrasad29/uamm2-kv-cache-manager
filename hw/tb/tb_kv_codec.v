`timescale 1ns/1ps
// Bit-exact check of uamm_kv_quant4 -> uamm_kv_dequant4 against uamm2/hwmodel.py vectors.
module tb_kv_codec;
    localparam LANES = 16, N = 400;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0, in_valid = 0;
    reg [LANES*8-1:0] x_in = 0;
    reg [7:0]  zp = 0;
    reg [15:0] inv = 0, step = 0;
    wire q_valid, x_valid;
    wire [LANES*4-1:0] q_flat;
    wire [LANES*8-1:0] x_out;

    uamm_kv_quant4 #(.LANES(LANES)) uq (.clk(clk), .rst_n(rst_n), .in_valid(in_valid),
        .x_flat(x_in), .zp(zp), .inv_q12(inv), .out_valid(q_valid), .q_flat(q_flat));
    uamm_kv_dequant4 #(.LANES(LANES)) ud (.clk(clk), .rst_n(rst_n), .in_valid(q_valid),
        .q_flat(q_flat), .zp(zp), .step_q8(step), .out_valid(x_valid), .x_flat(x_out));

    reg [167:0] vin  [0:N-1];
    reg [191:0] vexp [0:N-1];
    integer i, errors = 0, sawq, sawx;

    initial begin
        $readmemh("vectors/codec_in.hex", vin);
        $readmemh("vectors/codec_exp.hex", vexp);
        repeat (3) @(posedge clk);
        rst_n = 1;
        for (i = 0; i < N; i = i + 1) begin
            @(negedge clk);
            {zp, inv, step, x_in} = vin[i];
            in_valid = 1;
            @(negedge clk);
            in_valid = 0;
            @(negedge clk);
            if ({x_out, q_flat} !== vexp[i]) begin
                errors = errors + 1;
                if (errors <= 10) $display("MISMATCH vec %0d got=%048h exp=%048h", i, {x_out, q_flat}, vexp[i]);
            end
        end
        rst_n = 0;
        #2;
        if (q_valid !== 1'b0 || x_valid !== 1'b0 || q_flat !== 0 || x_out !== 0) begin
            errors = errors + 1; $display("reset did not clear codec outputs");
        end
        if (errors == 0) $display("TB_KV_CODEC PASS vectors=%0d", N);
        else             $display("TB_KV_CODEC FAIL errors=%0d", errors);
        $finish;
    end
    initial if ($test$plusargs("trace")) begin
        $dumpfile("trace_kv_codec.vcd");
        $dumpvars(0, tb_kv_codec);
    end
endmodule
