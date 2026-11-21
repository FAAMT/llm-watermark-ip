// detect_core: P_d = 2 lanes, one 64-bit beat (two token words) per cycle.
// ctx_shift keeps the last four tokens; each lane folds its own context,
// runs fmix64, hashes its token with fmix32, tests green and reads the score
// ROM (both lanes share one dual-port copy). score_acc sums S, g and N.
// Golden: detect().
module detect_core #(
    parameter MUL_REG = 1,
    parameter SROM = "src/rtl/rom/score_rom.hex"
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [63:0] key,
    input  wire [2:0]  k,
    input  wire [31:0] theta,
    input  wire [24:0] T,
    input  wire        in_valid,
    input  wire [63:0] in_data,
    input  wire        in_last,
    output reg         done,
    output reg  [31:0] S,
    output reg  [31:0] g,
    output reg  [31:0] N
);
    localparam [31:0]  C_IDX = 32'h9E3779B1;
    localparam integer LAT32 = 3 + 2 * MUL_REG;
    localparam integer LAT64 = 3 + 2 * MUL_REG;
    localparam integer L_S   = 2 + LAT64;          // seed ready
    localparam integer L_U   = L_S + LAT32;        // u ready
    localparam integer L_SC  = L_U + 2;            // score ready

    // ---- ctx_shift: position counter and the last four tokens, newest first
    reg [24:0] pos;
    reg [23:0] h0, h1, h2, h3;
    wire [23:0] ta = in_data[23:0];
    wire [23:0] tb = in_data[55:32];
    wire        sa = (pos < T)          && (pos >= k)          && !in_data[31];
    wire        sb = ((pos + 1) < T)    && ((pos + 1) >= k)    && !in_data[63];
    always @(posedge clk) begin
        if (rst || start) begin
            pos <= 25'd0; h0 <= 0; h1 <= 0; h2 <= 0; h3 <= 0;
        end else if (in_valid) begin
            pos <= pos + 25'd2;
            h0 <= tb; h1 <= ta; h2 <= h0; h3 <= h1;
        end
    end

    // ---- two lanes
    wire [63:0] xa, xb, sda, sdb;
    ctx_fold fa (.clk(clk), .key(key), .c0(h0), .c1(h1), .c2(h2), .c3(h3), .k(k), .x(xa));
    ctx_fold fb (.clk(clk), .key(key), .c0(ta), .c1(h0), .c2(h1), .c3(h2), .k(k), .x(xb));
    fmix64_pipe #(MUL_REG) ma (.clk(clk), .din(xa), .dout(sda));
    fmix64_pipe #(MUL_REG) mb (.clk(clk), .din(xb), .dout(sdb));

    reg  [31:0] tma, tmb;                          // token * C_IDX mod 2^32
    always @(posedge clk) begin
        tma <= {8'd0, ta} * C_IDX;
        tmb <= {8'd0, tb} * C_IDX;
    end
    wire [31:0] tma_d, tmb_d;
    delay #(32, L_S - 1) dta (.clk(clk), .d(tma), .q(tma_d));
    delay #(32, L_S - 1) dtb (.clk(clk), .d(tmb), .q(tmb_d));

    wire [31:0] ua, ub;
    fmix32_pipe #(MUL_REG) ha (.clk(clk), .din(tma_d ^ sda[31:0]), .dout(ua));
    fmix32_pipe #(MUL_REG) hb (.clk(clk), .din(tmb_d ^ sdb[31:0]), .dout(ub));

    wire [14:0] aa, ab;
    wire [15:0] sca, scb;
    pf_index ia (.clk(clk), .u(ua), .addr(aa));
    pf_index ib (.clk(clk), .u(ub), .addr(ab));
    rom_dp #(.W(16), .D(16896), .AW(15), .INIT(SROM)) srom (
        .clk(clk), .a0(aa), .a1(ab), .q0(sca), .q1(scb));

    reg ga, gb;
    always @(posedge clk) begin ga <= (ua >= theta); gb <= (ub >= theta); end
    wire ga_d, gb_d;
    delay #(1, 1) dga (.clk(clk), .d(ga), .q(ga_d));
    delay #(1, 1) dgb (.clk(clk), .d(gb), .q(gb_d));

    wire sa_d, sb_d, vld_d, last_d;
    delay_rst #(4, L_SC) dflags (.clk(clk), .rst(rst || start),
        .d({sa && in_valid, sb && in_valid, in_valid, in_last}), .q({sa_d, sb_d, vld_d, last_d}));

    // ---- score_acc
    always @(posedge clk) begin
        done <= 1'b0;
        if (rst || start) begin
            S <= 0; g <= 0; N <= 0;
        end else if (vld_d) begin
            S <= S + (sa_d ? {16'd0, sca} : 32'd0) + (sb_d ? {16'd0, scb} : 32'd0);
            g <= g + (sa_d && ga_d) + (sb_d && gb_d);
            N <= N + sa_d + sb_d;
            if (last_d) done <= 1'b1;
        end
    end
endmodule
