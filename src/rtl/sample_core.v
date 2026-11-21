// sample_core: P = 4 lanes, one 64-bit beat (4 int16 logits) per cycle.
// Per lane: keyed and unkeyed fmix32 (hash_lanes), green test, noise select,
// pseudo-float address (rom_index), Gumbel ROM (two lanes per dual-port copy),
// then add_argmax: v = logit + delta*green + G, a 4-input max tree and a
// running maximum across beats. Ties go to the smallest index.
// Golden: uniforms(), green(), pf_index(), sample().
module sample_core #(
    parameter MUL_REG = 1,
    parameter GROM = "src/rtl/rom/gumbel_rom.hex"
) (
    input  wire               clk,
    input  wire               rst,
    input  wire               start,        // one pulse per SAMPLE, seeds valid
    input  wire [31:0]        s_lo,
    input  wire [31:0]        sf_lo,
    input  wire               noise_free,   // noise from u' (GREEN, or a free step)
    input  wire               green_en,     // scheme == GREEN: add delta to green tokens
    input  wire [31:0]        theta,
    input  wire signed [15:0] delta,
    input  wire [23:0]        V,
    input  wire               in_valid,
    input  wire [63:0]        in_data,
    input  wire               in_last,
    output reg                done,
    output reg  [23:0]        best_idx,
    output reg  signed [17:0] best_val
);
    localparam integer P     = 4;
    localparam [31:0]  C_IDX = 32'h9E3779B1;
    localparam integer LAT32 = 3 + 2 * MUL_REG;
    localparam integer LAT_G = LAT32 + 2;       // to the ROM output

    // ---- stage 0: index accumulators (t * C_IDX mod 2^32 without a multiplier)
    reg [31:0] acc [0:P-1];
    reg [23:0] base;
    reg [31:0] s_lo_r, sf_lo_r;
    integer i;
    always @(posedge clk) begin
        if (start) begin
            for (i = 0; i < P; i = i + 1) acc[i] <= C_IDX * i;
            base    <= 24'd0;
            s_lo_r  <= s_lo;
            sf_lo_r <= sf_lo;
        end else if (in_valid) begin
            for (i = 0; i < P; i = i + 1) acc[i] <= acc[i] + C_IDX * P;
            base <= base + P;
        end
    end

    // carried alongside the hashes: logits, index base, valid, last
    wire [63:0] lg_g;
    wire [23:0] base_g;
    wire        vld_g, last_g;
    delay     #(64, LAT_G) d_lg   (.clk(clk), .d(in_data), .q(lg_g));
    delay     #(24, LAT_G) d_base (.clk(clk), .d(base),    .q(base_g));
    delay_rst #(2,  LAT_G) d_vl   (.clk(clk), .rst(rst || start), .d({in_valid, in_last}), .q({vld_g, last_g}));

    wire [31:0] uk [0:P-1];
    wire [31:0] uf [0:P-1];
    wire [14:0] addr [0:P-1];
    wire [15:0] gval [0:P-1];
    wire [P-1:0] green_g;

    genvar p;
    generate
        for (p = 0; p < P; p = p + 1) begin : lane
            fmix32_pipe #(MUL_REG) hk (.clk(clk), .din(acc[p] ^ s_lo_r),  .dout(uk[p]));
            fmix32_pipe #(MUL_REG) hf (.clk(clk), .din(acc[p] ^ sf_lo_r), .dout(uf[p]));
            wire green = (uk[p] >= theta);
            delay #(1, 2) d_gr (.clk(clk), .d(green), .q(green_g[p]));
            pf_index ix (.clk(clk), .u(noise_free ? uf[p] : uk[p]), .addr(addr[p]));
        end
        for (p = 0; p < P; p = p + 2) begin : rom
            rom_dp #(.W(16), .D(16896), .AW(15), .INIT(GROM)) g (
                .clk(clk), .a0(addr[p]), .a1(addr[p+1]), .q0(gval[p]), .q1(gval[p+1]));
        end
    endgenerate

    // ---- add stage: v = logit + delta*green + G (18-bit), and lane validity
    reg signed [17:0] v  [0:P-1];
    reg [P-1:0]       ok;
    reg [23:0]        base_a;
    reg               vld_a, last_a;
    always @(posedge clk) begin
        for (i = 0; i < P; i = i + 1) begin
            v[i]  <= $signed(lg_g[16*i +: 16]) + $signed(gval[i])
                   + ((green_en && green_g[i]) ? delta : 16'sd0);
            ok[i] <= (base_g + i) < V;
        end
        base_a <= base_g;
        vld_a  <= rst || start ? 1'b0 : vld_g;
        last_a <= last_g;
    end

    // ---- max tree over 4 lanes: the lower index wins ties
    wire        ok01 = ok[0] | ok[1];
    wire        pick1 = !ok[0] || (ok[1] && v[1] > v[0]);
    wire signed [17:0] v01 = pick1 ? v[1] : v[0];
    wire [1:0]  i01 = pick1 ? 2'd1 : 2'd0;
    wire        ok23 = ok[2] | ok[3];
    wire        pick3 = !ok[2] || (ok[3] && v[3] > v[2]);
    wire signed [17:0] v23 = pick3 ? v[3] : v[2];
    wire [1:0]  i23 = pick3 ? 2'd3 : 2'd2;
    wire        pickhi = !ok01 || (ok23 && v23 > v01);
    reg signed [17:0] bv;
    reg [23:0]  bi;
    reg         bok, vld_t, last_t;
    always @(posedge clk) begin
        bv     <= pickhi ? v23 : v01;
        bi     <= base_a + (pickhi ? i23 : i01);
        bok    <= ok01 | ok23;
        vld_t  <= rst || start ? 1'b0 : vld_a;
        last_t <= last_a;
    end

    // ---- running maximum across beats; strictly greater replaces
    reg have;
    always @(posedge clk) begin
        done <= 1'b0;
        if (rst || start) begin
            have <= 1'b0;
        end else if (vld_t) begin
            if (bok && (!have || bv > best_val)) begin
                best_val <= bv;
                best_idx <= bi;
                have     <= 1'b1;
            end
            if (last_t) done <= 1'b1;
        end
    end
endmodule
