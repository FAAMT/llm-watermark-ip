// Unit testbench: one vector per cycle into fmix32_pipe, fmix64_pipe, pf_index
// and ctx_fold; every cycle's outputs are printed, and scripts/test_units.py
// lines them up with each block's latency and compares with the golden model.
// Input line:  <u32 hex> <u64 hex> <key hex> <c0> <c1> <c2> <c3> <k>   (c* and k decimal)
`timescale 1ns/1ps
module tb_units;
    reg clk = 0; always #5 clk = ~clk;
    reg [31:0] u32 = 0; reg [63:0] u64 = 0, key = 0; reg [23:0] c0 = 0, c1 = 0, c2 = 0, c3 = 0; reg [2:0] k = 1;
    wire [31:0] f32; wire [63:0] f64, x; wire [14:0] a;
    fmix32_pipe #(1) u_f32 (.clk(clk), .din(u32), .dout(f32));
    fmix64_pipe #(1) u_f64 (.clk(clk), .din(u64), .dout(f64));
    pf_index         u_pf  (.clk(clk), .u(u32), .addr(a));
    ctx_fold         u_cf  (.clk(clk), .key(key), .c0(c0), .c1(c1), .c2(c2), .c3(c3), .k(k), .x(x));
    integer fi, fo, rc, n; integer ic0, ic1, ic2, ic3, ik; reg [8*256-1:0] fin, fout;
    initial begin
        if (!$value$plusargs("in=%s", fin)) fin = "build/units.in.txt";
        if (!$value$plusargs("out=%s", fout)) fout = "build/units.out.txt";
        fi = $fopen(fin, "r"); fo = $fopen(fout, "w"); n = 0;
        while (!$feof(fi)) begin
            rc = $fscanf(fi, "%h %h %h %d %d %d %d %d\n", u32, u64, key, ic0, ic1, ic2, ic3, ik);
            c0 = ic0; c1 = ic1; c2 = ic2; c3 = ic3; k = ik;
            @(posedge clk); #1;
            $fwrite(fo, "%08h %016h %04h %016h\n", f32, f64, a, x);
            n = n + 1;
        end
        repeat (8) begin @(posedge clk); #1; $fwrite(fo, "%08h %016h %04h %016h\n", f32, f64, a, x); end
        $fclose(fo); $finish;
    end
endmodule
