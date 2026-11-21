// wm_top: the watermark sampler IP. One AXI4-Stream in (commands and data),
// one AXI4-Stream out (responses), AXI4-Lite for control and status only.
// The controller FSM parses headers, checks errors in the course's order,
// holds CONFIG, and sequences seed_unit -> sample_core, or detect_core, then
// writes the response. Interface details: docs/interface.md.
module wm_top #(
    parameter MUL_REG = 1,
    parameter GROM = "src/rtl/rom/gumbel_rom.hex",
    parameter SROM = "src/rtl/rom/score_rom.hex"
) (
    input  wire        ap_clk,
    input  wire        ap_rst_n,
    // command stream in
    input  wire [63:0] s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire        s_axis_tlast,
    // response stream out
    output wire [63:0] m_axis_tdata,
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,
    output wire        m_axis_tlast,
    // AXI4-Lite
    input  wire [5:0]  s_axil_awaddr,
    input  wire        s_axil_awvalid,
    output wire        s_axil_awready,
    input  wire [31:0] s_axil_wdata,
    input  wire [3:0]  s_axil_wstrb,
    input  wire        s_axil_wvalid,
    output wire        s_axil_wready,
    output wire [1:0]  s_axil_bresp,
    output wire        s_axil_bvalid,
    input  wire        s_axil_bready,
    input  wire [5:0]  s_axil_araddr,
    input  wire        s_axil_arvalid,
    output wire        s_axil_arready,
    output wire [31:0] s_axil_rdata,
    output wire [1:0]  s_axil_rresp,
    output wire        s_axil_rvalid,
    input  wire        s_axil_rready
);
    wire clk = ap_clk;
    wire rst = !ap_rst_n;

    // command codes and error codes (docs/interface.md)
    localparam [7:0] CMD_CONFIG = 8'd0, CMD_SAMPLE = 8'd1, CMD_DETECT = 8'd2, CMD_END = 8'd3;
    localparam [7:0] E_BAD_CMD = 8'd1, E_BAD_PARAM = 8'd2, E_BAD_SIZE = 8'd3,
                     E_NO_CONFIG = 8'd4, E_TLAST_EARLY = 8'd5, E_TLAST_MISSING = 8'd6;

    localparam [3:0] S_IDLE = 4'd0, S_HDR0 = 4'd1, S_HDRN = 4'd2, S_SEED = 4'd3,
                     S_SDATA = 4'd4, S_SWAIT = 4'd5, S_DDATA = 4'd6, S_DWAIT = 4'd7,
                     S_RESP = 4'd8, S_DONE = 4'd9;

    reg [3:0] state;

    // ---- AXI4-Lite
    wire        ap_start;
    reg         done_pulse, halted;
    reg  [7:0]  err;
    reg  [15:0] tx_id;
    reg  [31:0] ncmd;
    axil_ctrl u_axil (
        .clk(clk), .rst(rst),
        .awaddr(s_axil_awaddr), .awvalid(s_axil_awvalid), .awready(s_axil_awready),
        .wdata(s_axil_wdata), .wstrb(s_axil_wstrb), .wvalid(s_axil_wvalid), .wready(s_axil_wready),
        .bresp(s_axil_bresp), .bvalid(s_axil_bvalid), .bready(s_axil_bready),
        .araddr(s_axil_araddr), .arvalid(s_axil_arvalid), .arready(s_axil_arready),
        .rdata(s_axil_rdata), .rresp(s_axil_rresp), .rvalid(s_axil_rvalid), .rready(s_axil_rready),
        .ap_start(ap_start), .busy(state != S_IDLE), .done_pulse(done_pulse),
        .halted(halted), .err(err), .tx_id(tx_id), .ncmd(ncmd));

    // ---- configuration (cleared at every ap_start: nothing carries over)
    reg         cfg_valid;
    reg  [24:0] cfg_V;
    reg  [63:0] cfg_K, cfg_nonce;
    reg  [31:0] cfg_theta;
    reg  signed [15:0] cfg_delta;
    reg  [2:0]  cfg_k;
    reg         cfg_green;

    // ---- header fields
    reg  [7:0]  cmd;
    reg         free_bit;
    reg  [31:0] p32;                 // V, step or T
    reg  [63:0] hw1, hw2;
    reg  [1:0]  hcnt;                // header words still to read

    reg  seed_start, samp_start, det_start;
    wire in_hs  = s_axis_tvalid && s_axis_tready;
    // data states hold tready low for the one cycle their core's start pulse
    // is high, so the first beat never meets a core that is still resetting
    assign s_axis_tready = (state == S_HDR0) || (state == S_HDRN) ||
                           (state == S_SDATA && !samp_start) ||
                           (state == S_DDATA && !det_start);

    // ---- data burst bookkeeping
    reg  [24:0] nbeats, beat;
    wire        last_beat = (beat == nbeats - 25'd1);

    // ---- seed_unit / sample_core / detect_core
    wire seed_ready, samp_done, det_done;
    wire [31:0] s_lo, sf_lo;
    seed_unit #(MUL_REG) u_seed (
        .clk(clk), .rst(rst), .start(seed_start), .key(cfg_K), .nonce(cfg_nonce),
        .step(p32), .c0(hw1[23:0]), .c1(hw1[55:32]), .c2(hw2[23:0]), .c3(hw2[55:32]),
        .k(cfg_k), .ready(seed_ready), .s_lo(s_lo), .sf_lo(sf_lo));

    wire [23:0]        best_idx;
    wire signed [17:0] best_val;
    sample_core #(.MUL_REG(MUL_REG), .GROM(GROM)) u_samp (
        .clk(clk), .rst(rst), .start(samp_start), .s_lo(s_lo), .sf_lo(sf_lo),
        .noise_free(cfg_green || free_bit), .green_en(cfg_green),
        .theta(cfg_theta), .delta(cfg_delta), .V(cfg_V[23:0]),
        .in_valid(in_hs && state == S_SDATA), .in_data(s_axis_tdata), .in_last(last_beat),
        .done(samp_done), .best_idx(best_idx), .best_val(best_val));

    wire [31:0] dS, dg, dN;
    detect_core #(.MUL_REG(MUL_REG), .SROM(SROM)) u_det (
        .clk(clk), .rst(rst), .start(det_start), .key(cfg_K), .k(cfg_k), .theta(cfg_theta),
        .T(p32[24:0]), .in_valid(in_hs && state == S_DDATA), .in_data(s_axis_tdata),
        .in_last(last_beat), .done(det_done), .S(dS), .g(dg), .N(dN));

    // ---- response
    reg  [63:0] resp [0:2];
    reg  [1:0]  rcount, ridx;
    assign m_axis_tvalid = (state == S_RESP);
    assign m_axis_tdata  = resp[ridx];
    assign m_axis_tlast  = (ridx == rcount - 2'd1);

    // header fields of the CONFIG word 3
    wire [31:0]        w3_theta = s_axis_tdata[31:0];
    wire signed [15:0] w3_delta = s_axis_tdata[47:32];
    wire [2:0]         w3_k     = s_axis_tdata[50:48];
    wire [1:0]         w3_sch   = s_axis_tdata[57:56];
    // V < 2^24: token ids, and sample_core's index compare, are 24 bits wide
    wire cfg_bad = (p32 == 32'd0) || (p32 > 32'h00FF_FFFF) || (w3_k == 3'd0) || (w3_k > 3'd4) ||
                   (w3_sch > 2'd1) || (w3_theta == 32'd0) || (w3_delta <= 16'sd0) ||
                   (w3_delta >= 16'sd4096);

    task fail(input [7:0] code);
        begin
            err    <= code;
            halted <= 1'b1;
            state  <= S_DONE;
        end
    endtask

    always @(posedge clk) begin
        seed_start <= 1'b0;
        samp_start <= 1'b0;
        det_start  <= 1'b0;
        done_pulse <= 1'b0;
        if (rst) begin
            state <= S_IDLE; cfg_valid <= 1'b0; halted <= 1'b0; err <= 8'd0;
            tx_id <= 16'd0; ncmd <= 32'd0;
        end else begin
            case (state)
            S_IDLE: if (ap_start) begin
                cfg_valid <= 1'b0; halted <= 1'b0; err <= 8'd0; tx_id <= 16'd0; ncmd <= 32'd0;
                state <= S_HDR0;
            end
            S_HDR0: if (in_hs) begin
                cmd      <= s_axis_tdata[7:0];
                free_bit <= s_axis_tdata[8];
                tx_id    <= s_axis_tdata[31:16];
                p32      <= s_axis_tdata[63:32];
                case (s_axis_tdata[7:0])
                    CMD_CONFIG: begin hcnt <= 2'd3; state <= S_HDRN; end
                    CMD_SAMPLE: begin hcnt <= 2'd2; state <= S_HDRN; end
                    CMD_DETECT: begin
                        if (s_axis_tdata[63:32] == 32'd0 || s_axis_tdata[63:32] > 32'h0100_0000)
                            fail(E_BAD_SIZE);
                        else if (!cfg_valid)
                            fail(E_NO_CONFIG);
                        else begin
                            nbeats    <= (s_axis_tdata[56:32] + 25'd1) >> 1;
                            beat      <= 25'd0;
                            det_start <= 1'b1;
                            state     <= S_DDATA;
                        end
                    end
                    CMD_END: begin ncmd <= ncmd + 1; state <= S_DONE; end
                    default: fail(E_BAD_CMD);
                endcase
            end
            S_HDRN: if (in_hs) begin
                hcnt <= hcnt - 2'd1;
                if (cmd == CMD_CONFIG) begin
                    if (hcnt == 2'd3) cfg_K     <= s_axis_tdata;
                    if (hcnt == 2'd2) cfg_nonce <= s_axis_tdata;
                    if (hcnt == 2'd1) begin
                        if (cfg_bad) fail(E_BAD_PARAM);
                        else begin
                            cfg_V     <= p32[24:0];
                            cfg_theta <= w3_theta;
                            cfg_delta <= w3_delta;
                            cfg_k     <= w3_k;
                            cfg_green <= w3_sch[0];
                            cfg_valid <= 1'b1;
                            ncmd      <= ncmd + 1;
                            state     <= S_HDR0;
                        end
                    end
                end else begin                                  // SAMPLE
                    if (hcnt == 2'd2) hw1 <= s_axis_tdata;
                    if (hcnt == 2'd1) begin
                        hw2 <= s_axis_tdata;
                        if (!cfg_valid) fail(E_NO_CONFIG);
                        else begin
                            seed_start <= 1'b1;
                            state      <= S_SEED;
                        end
                    end
                end
            end
            S_SEED: if (seed_ready && !seed_start) begin
                nbeats     <= (cfg_V + 25'd3) >> 2;
                beat       <= 25'd0;
                samp_start <= 1'b1;
                state      <= S_SDATA;
            end
            S_SDATA: if (in_hs) begin
                beat <= beat + 25'd1;
                if (last_beat) begin
                    if (!s_axis_tlast) fail(E_TLAST_MISSING);
                    else state <= S_SWAIT;
                end else if (s_axis_tlast) begin
                    fail(E_TLAST_EARLY);
                end
            end
            S_SWAIT: if (samp_done) begin
                resp[0] <= {32'd1, tx_id, 8'd0, CMD_SAMPLE};
                resp[1] <= {{14{best_val[17]}}, best_val, 8'd0, best_idx};
                rcount  <= 2'd2;
                ridx    <= 2'd0;
                state   <= S_RESP;
            end
            S_DDATA: if (in_hs) begin
                beat <= beat + 25'd1;
                if (last_beat) begin
                    if (!s_axis_tlast) fail(E_TLAST_MISSING);
                    else state <= S_DWAIT;
                end else if (s_axis_tlast) begin
                    fail(E_TLAST_EARLY);
                end
            end
            S_DWAIT: if (det_done) begin
                resp[0] <= {32'd2, tx_id, 8'd0, CMD_DETECT};
                resp[1] <= {dN, dS};
                resp[2] <= {32'd0, dg};
                rcount  <= 2'd3;
                ridx    <= 2'd0;
                state   <= S_RESP;
            end
            S_RESP: if (m_axis_tready) begin
                if (ridx == rcount - 2'd1) begin
                    ncmd  <= ncmd + 1;
                    state <= S_HDR0;
                end
                ridx <= ridx + 2'd1;
            end
            S_DONE: begin
                done_pulse <= 1'b1;
                state      <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
