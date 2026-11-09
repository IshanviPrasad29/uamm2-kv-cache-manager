`timescale 1ns/1ps
// Bit-exact check of uamm_sweep_scorer against vectors from uamm2/hwmodel.py.
// Run from the hw/ directory (vectors are read from vectors/).
module tb_sweep_scorer;
    localparam NBLK = 64, AW = 6, DS = 4, NP = 4;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0, we = 0, start = 0;
    reg [AW-1:0] waddr = 0;
    reg [24:0]   wdata = 0;
    reg [7:0]    pressure = 0;
    wire busy, done, out_valid, out_evict;
    wire [AW-1:0] out_idx;
    wire [7:0] out_score;
    wire [1:0] out_tier;

    uamm_sweep_scorer #(.NBLK(NBLK), .AW(AW)) dut (
        .clk(clk), .rst_n(rst_n), .we(we), .waddr(waddr), .wdata(wdata),
        .start(start), .pressure(pressure), .busy(busy), .done(done),
        .out_valid(out_valid), .out_idx(out_idx), .out_score(out_score),
        .out_tier(out_tier), .out_evict(out_evict));

    reg [24:0] meta [0:DS*NBLK-1];
    reg [10:0] expv [0:NP*DS*NBLK-1];
    reg [7:0]  plist [0:NP-1];
    integer errors = 0, got = 0, base = 0, checks = 0;
    reg checking = 0;

    always @(posedge clk) begin
        if (checking && out_valid) begin
            checks = checks + 1;
            got = got + 1;
            if ({out_evict, out_tier, out_score} !== expv[base + out_idx]) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("MISMATCH idx=%0d got evict=%b tier=%b score=%0d  exp=%03h",
                             out_idx, out_evict, out_tier, out_score, expv[base + out_idx]);
            end
        end
    end

    task load_dataset(input integer ds);
        integer i;
        begin
            for (i = 0; i < NBLK; i = i + 1) begin
                @(negedge clk); we = 1; waddr = i; wdata = meta[ds*NBLK + i];
            end
            @(negedge clk); we = 0;
        end
    endtask

    task sweep(input integer ds, input integer pi, input integer poke);
        begin
            base = pi*DS*NBLK + ds*NBLK; got = 0; checking = 1;
            @(negedge clk); pressure = plist[pi]; start = 1;
            @(negedge clk); start = 0;
            if (poke) begin
                repeat (10) @(negedge clk);
                pressure = 8'd255; start = 1;       // must be ignored while busy
                @(negedge clk); start = 0;
            end
            wait (done === 1'b1);
            repeat (3) @(posedge clk);
            checking = 0;
            if (got != NBLK) begin
                errors = errors + 1;
                $display("COUNT ERROR ds=%0d pi=%0d got=%0d", ds, pi, got);
            end
            if (busy !== 1'b0) begin
                errors = errors + 1;
                $display("busy still high after sweep");
            end
        end
    endtask

    integer ds, pi;
    initial begin
        $readmemh("vectors/scorer_meta.hex", meta);
        $readmemh("vectors/scorer_exp.hex", expv);
        plist[0] = 0; plist[1] = 242; plist[2] = 243; plist[3] = 255;
        repeat (3) @(posedge clk);
        rst_n = 1;
        for (ds = 0; ds < DS; ds = ds + 1) begin
            load_dataset(ds);
            for (pi = 0; pi < NP; pi = pi + 1)
                sweep(ds, pi, (ds == 0 && pi == 0));
        end
        // reset in the middle of a sweep, then verify clean recovery
        @(negedge clk); pressure = 8'd255; start = 1;
        @(negedge clk); start = 0;
        repeat (10) @(negedge clk);
        rst_n = 0;
        #2;
        if (busy !== 1'b0 || out_valid !== 1'b0 || done !== 1'b0) begin
            errors = errors + 1; $display("reset did not clear control state");
        end
        repeat (2) @(negedge clk);
        rst_n = 1;
        sweep(3, 3, 0);
        if (errors == 0) $display("TB_SWEEP_SCORER PASS checks=%0d", checks);
        else             $display("TB_SWEEP_SCORER FAIL errors=%0d", errors);
        $finish;
    end

    initial begin
        #5000000;
        $display("TB_SWEEP_SCORER FAIL timeout");
        $finish;
    end
    initial if ($test$plusargs("trace")) begin
        $dumpfile("trace_sweep_scorer.vcd");
        $dumpvars(0, tb_sweep_scorer);
    end
endmodule
