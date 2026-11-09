`timescale 1ns/1ps
// Randomized check of uamm_block_table against a behavioural reference model.
module tb_block_table;
    localparam AW = 8, PW = 12, NOPS = 6000;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0, we = 0, w_valid = 0, rd_en = 0;
    reg [AW-1:0] waddr = 0, raddr = 0;
    reg [1:0] w_tier = 0, w_codec = 0;
    reg [PW-1:0] w_phys = 0;
    wire rd_valid, hit;
    wire [1:0] tier, codec;
    wire [PW-1:0] phys;

    uamm_block_table #(.AW(AW), .PW(PW)) dut (.clk(clk), .rst_n(rst_n), .we(we), .waddr(waddr),
        .w_valid(w_valid), .w_tier(w_tier), .w_codec(w_codec), .w_phys(w_phys), .rd_en(rd_en),
        .raddr(raddr), .rd_valid(rd_valid), .hit(hit), .tier(tier), .codec(codec), .phys(phys));

    reg [PW+3:0] rmem [0:(1<<AW)-1];
    reg          rv   [0:(1<<AW)-1];
    integer n, i, a, errors = 0, reads = 0, writes = 0, hits = 0;
    reg [1:0] t, c;
    reg [PW-1:0] ph;
    reg vv;

    task do_read(input integer addr);
        begin
            @(negedge clk); rd_en = 1; raddr = addr; we = 0;
            @(posedge clk); #1;
            reads = reads + 1;
            if (rd_valid !== 1'b1) begin errors = errors + 1; $display("rd_valid low"); end
            if (hit !== rv[addr]) begin
                errors = errors + 1; $display("HIT MISMATCH addr=%0d got=%b exp=%b", addr, hit, rv[addr]);
            end else if (rv[addr]) begin
                hits = hits + 1;
                if ({tier, codec, phys} !== rmem[addr]) begin
                    errors = errors + 1;
                    $display("DATA MISMATCH addr=%0d got=%0h exp=%0h", addr, {tier, codec, phys}, rmem[addr]);
                end
            end
            @(negedge clk); rd_en = 0;
            @(posedge clk); #1;
            if (rd_valid !== 1'b0) begin errors = errors + 1; $display("rd_valid stuck high"); end
        end
    endtask

    initial begin
        for (i = 0; i < (1<<AW); i = i + 1) begin rv[i] = 0; rmem[i] = 0; end
        repeat (3) @(posedge clk);
        rst_n = 1;
        for (i = 0; i < 20; i = i + 1) do_read($urandom % (1<<AW));    // all must miss after reset
        for (n = 0; n < NOPS; n = n + 1) begin
            if (($urandom % 4) < 2) begin
                a = $urandom % (1<<AW);
                t = $urandom; c = $urandom; ph = $urandom; vv = (($urandom % 8) != 0);
                @(negedge clk);
                we = 1; waddr = a; w_valid = vv; w_tier = t; w_codec = c; w_phys = ph; rd_en = 0;
                rmem[a] = {t, c, ph}; rv[a] = vv; writes = writes + 1;
                @(negedge clk); we = 0;
            end else begin
                do_read($urandom % (1<<AW));
            end
        end
        if (errors == 0) $display("TB_BLOCK_TABLE PASS writes=%0d reads=%0d hits=%0d", writes, reads, hits);
        else             $display("TB_BLOCK_TABLE FAIL errors=%0d", errors);
        $finish;
    end
    initial if ($test$plusargs("trace")) begin
        $dumpfile("trace_block_table.vcd");
        $dumpvars(0, tb_block_table);
    end
endmodule
