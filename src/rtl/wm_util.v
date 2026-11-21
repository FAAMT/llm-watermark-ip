// Small shared building blocks: delay lines, MurmurHash3 finalisers, context
// fold, pseudo-float ROM address, dual-port ROM. Every block is a free-running
// pipeline (II = 1, no enable): the SAMPLE and DETECT datapaths never stall,
// because their only output is one response written after the last beat.

// W-bit delay line of N registers (N = 0 is a wire). No reset: data only.
module delay #(parameter W = 1, parameter N = 1) (
    input  wire         clk,
    input  wire [W-1:0] d,
    output wire [W-1:0] q
);
    generate
        if (N == 0) begin : g_wire
            assign q = d;
        end else begin : g_regs
            reg [W-1:0] r [0:N-1];
            integer i;
            always @(posedge clk) begin
                r[0] <= d;
                for (i = 1; i < N; i = i + 1) r[i] <= r[i-1];
            end
            assign q = r[N-1];
        end
    endgenerate
endmodule

// Same, with synchronous reset to zero: for valid / last flags.
module delay_rst #(parameter W = 1, parameter N = 1) (
    input  wire         clk,
    input  wire         rst,
    input  wire [W-1:0] d,
    output wire [W-1:0] q
);
    generate
        if (N == 0) begin : g_wire
            assign q = d;
        end else begin : g_regs
            reg [W-1:0] r [0:N-1];
            integer i;
            always @(posedge clk) begin
                if (rst) begin
                    for (i = 0; i < N; i = i + 1) r[i] <= {W{1'b0}};
                end else begin
                    r[0] <= d;
                    for (i = 1; i < N; i = i + 1) r[i] <= r[i-1];
                end
            end
            assign q = r[N-1];
        end
    endgenerate
endmodule

// MurmurHash3 32-bit finaliser. Latency 3 + 2*MUL_REG, II = 1.
// MUL_REG adds a register after each constant multiply so the synthesiser can
// pack it into the DSP48E1 M/P registers.
module fmix32_pipe #(parameter MUL_REG = 1) (
    input  wire        clk,
    input  wire [31:0] din,
    output reg  [31:0] dout
);
    localparam [31:0] M1 = 32'h85EBCA6B;
    localparam [31:0] M2 = 32'hC2B2AE35;
    localparam integer LAT = 3 + 2 * MUL_REG;
    reg  [31:0] p1, p2;
    wire [31:0] p1d, p2d;
    wire [31:0] a = din ^ (din >> 16);
    always @(posedge clk) p1 <= a * M1;
    delay #(32, MUL_REG) d1 (.clk(clk), .d(p1), .q(p1d));
    wire [31:0] b = p1d ^ (p1d >> 13);
    always @(posedge clk) p2 <= b * M2;
    delay #(32, MUL_REG) d2 (.clk(clk), .d(p2), .q(p2d));
    always @(posedge clk) dout <= p2d ^ (p2d >> 16);
endmodule

// MurmurHash3 64-bit finaliser. Latency 3 + 2*MUL_REG, II = 1.
module fmix64_pipe #(parameter MUL_REG = 1) (
    input  wire        clk,
    input  wire [63:0] din,
    output reg  [63:0] dout
);
    localparam [63:0] M1 = 64'hFF51AFD7ED558CCD;
    localparam [63:0] M2 = 64'hC4CEB9FE1A85EC53;
    localparam integer LAT = 3 + 2 * MUL_REG;
    reg  [63:0] p1, p2;
    wire [63:0] p1d, p2d;
    wire [63:0] a = din ^ (din >> 33);
    always @(posedge clk) p1 <= a * M1;
    delay #(64, MUL_REG) d1 (.clk(clk), .d(p1), .q(p1d));
    wire [63:0] b = p1d ^ (p1d >> 33);
    always @(posedge clk) p2 <= b * M2;
    delay #(64, MUL_REG) d2 (.clk(clk), .d(p2), .q(p2d));
    always @(posedge clk) dout <= p2d ^ (p2d >> 33);
endmodule

// Context fold: x = key ^ XOR_{j<k} ((c_j + 1) * C_j mod 2^64), c0 the newest
// token. Latency 2 (multiply, then XOR). Matches step_seed() before fmix64.
module ctx_fold (
    input  wire        clk,
    input  wire [63:0] key,
    input  wire [23:0] c0, c1, c2, c3,
    input  wire [2:0]  k,
    output reg  [63:0] x
);
    localparam [63:0] C0 = 64'h9E3779B97F4A7C15;
    localparam [63:0] C1 = 64'hBF58476D1CE4E5B9;
    localparam [63:0] C2 = 64'h94D049BB133111EB;
    localparam [63:0] C3 = 64'h2545F4914F6CDD1D;
    // (c + 1) * C is computed as c * C + C: the multiplier operand stays 24 bits
    // wide (one DSP48E1 A-port chunk instead of two) and the + C rides the DSP
    // post-adder. Same value mod 2^64.
    reg [63:0] t0, t1, t2, t3, key_r;
    always @(posedge clk) begin
        t0    <= (k >= 3'd1) ? {40'd0, c0} * C0 + C0 : 64'd0;
        t1    <= (k >= 3'd2) ? {40'd0, c1} * C1 + C1 : 64'd0;
        t2    <= (k >= 3'd3) ? {40'd0, c2} * C2 + C2 : 64'd0;
        t3    <= (k >= 3'd4) ? {40'd0, c3} * C3 + C3 : 64'd0;
        key_r <= key;
        x     <= key_r ^ t0 ^ t1 ^ t2 ^ t3;
    end
endmodule

// Pseudo-float ROM address of a 32-bit uniform u. Latency 1.
// w = distance of u to the nearer end of [0, 2^32); e = leading zeros of w;
// m = the 8 bits after w's leading one; addr = ((half ? 33 : 0) + e) * 256 + m.
module pf_index (
    input  wire        clk,
    input  wire [31:0] u,
    output reg  [14:0] addr
);
    wire        half = u[31];
    wire [31:0] w    = half ? ~u : u;          // 2^32 - 1 - u when u >= 2^31
    // leading-zero count as a function in a continuous assignment, so simulation
    // evaluates it at time 0 too (an always @* block waits for an input event)
    function [5:0] clz32(input [31:0] v);
        integer i;
        begin
            clz32 = 6'd32;                     // v == 0
            for (i = 0; i < 32; i = i + 1)
                if (v[i]) clz32 = 6'd31 - i[5:0];   // last hit is the highest set bit
        end
    endfunction
    wire [5:0]  e = clz32(w);
    wire [32:0] sh = {1'b0, w} << (e + 6'd1);  // shifts >= 33 give 0, as in the model
    wire [6:0]  hb = (half ? 7'd33 : 7'd0) + {1'b0, e};
    always @(posedge clk) addr <= {hb, sh[31:24]};
endmodule

// Dual-port synchronous-read ROM, initialised from a hex file. Latency 1.
module rom_dp #(
    parameter W = 16,
    parameter D = 16896,
    parameter AW = 15,
    parameter INIT = "gumbel_rom.hex"
) (
    input  wire          clk,
    input  wire [AW-1:0] a0,
    input  wire [AW-1:0] a1,
    output reg  [W-1:0]  q0,
    output reg  [W-1:0]  q1
);
    (* rom_style = "block" *) reg [W-1:0] mem [0:D-1];
    initial $readmemh(INIT, mem);
    always @(posedge clk) begin
        q0 <= mem[a0];
        q1 <= mem[a1];
    end
endmodule
