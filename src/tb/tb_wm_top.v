// Top-level testbench. Replays a stimulus file through the real ports:
// AXI4-Lite ap_start, the input AXI4-Stream (with random bubbles), the output
// AXI4-Stream (with random backpressure), then polls ap_done and reads the
// status registers. One activation per block of lines; after each activation
// the rest of its lines are dropped, which models the host resetting the
// stream path (course command-response pattern, rule 5).
//
// Stimulus line:  <act> <hdr> <last> <64-bit hex>     (hdr = 1 on a header's first word)
// Output lines:   R <act> <last> <hex>                 response word
//                 H <act> <cycle> <hex>                header word accepted
//                 L <act> <cycle>                      data word with TLAST accepted
//                 Z <act> <cycle>                      response TLAST accepted
//                 S <act> <halted> <err> <tx_id> <ncmd> <cycles>   status after ap_done
// Plusargs: +in= +out= +seed= +bubble=<0-99> +stall=<0-99> +maxcyc=
`timescale 1ns/1ps
module tb_wm_top;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;                       // 100 MHz nominal
    reg [63:0] cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;

    reg  [63:0] s_tdata = 0; reg s_tvalid = 0, s_tlast = 0; wire s_tready;
    wire [63:0] m_tdata; wire m_tvalid, m_tlast; reg m_tready = 0;
    reg  [5:0]  awaddr = 0, araddr = 0; reg awvalid = 0, wvalid = 0, bready = 0, arvalid = 0, rready = 0;
    reg  [31:0] wdata = 0; wire awready, wready, bvalid, arready, rvalid; wire [1:0] bresp, rresp;
    wire [31:0] rdata;

    wm_top dut (
        .ap_clk(clk), .ap_rst_n(rst_n),
        .s_axis_tdata(s_tdata), .s_axis_tvalid(s_tvalid), .s_axis_tready(s_tready), .s_axis_tlast(s_tlast),
        .m_axis_tdata(m_tdata), .m_axis_tvalid(m_tvalid), .m_axis_tready(m_tready), .m_axis_tlast(m_tlast),
        .s_axil_awaddr(awaddr), .s_axil_awvalid(awvalid), .s_axil_awready(awready),
        .s_axil_wdata(wdata), .s_axil_wstrb(4'hF), .s_axil_wvalid(wvalid), .s_axil_wready(wready),
        .s_axil_bresp(bresp), .s_axil_bvalid(bvalid), .s_axil_bready(bready),
        .s_axil_araddr(araddr), .s_axil_arvalid(arvalid), .s_axil_arready(arready),
        .s_axil_rdata(rdata), .s_axil_rresp(rresp), .s_axil_rvalid(rvalid), .s_axil_rready(rready));

    integer fi, fo, seed, bubble, stall, rc;
    reg [8*256-1:0] fin, fout;
    reg [63:0] maxcyc;
    integer cur_act, ln_act, ln_hdr, ln_last; reg [63:0] ln_data; reg have_line;
    reg kernel_done = 0, feed_busy = 0;
    event feed_ev;

    task next_line;
        begin
            rc = $fscanf(fi, "%d %d %d %h\n", ln_act, ln_hdr, ln_last, ln_data);
            have_line = (rc == 4);
        end
    endtask

    function integer pct; input integer dummy; begin pct = {$random(seed)} % 100; end endfunction

    // output backpressure and response capture
    always @(posedge clk) begin
        if (m_tvalid && m_tready) begin
            $fwrite(fo, "R %0d %0d %016h\n", cur_act, m_tlast, m_tdata);
            if (m_tlast) $fwrite(fo, "Z %0d %0d\n", cur_act, cyc);
        end
        #1 m_tready = (pct(0) >= stall);
    end

    // feeder: drives the input stream for the current activation
    always begin
        @(feed_ev);
        feed_busy = 1;
        while (have_line && ln_act == cur_act && !kernel_done) begin
            while (pct(0) < bubble) begin @(posedge clk); #1; end
            s_tdata = ln_data; s_tlast = ln_last[0]; s_tvalid = 1;
            begin : wait_accept
                while (!kernel_done) begin
                    @(posedge clk);
                    if (s_tready) begin
                        if (ln_hdr) $fwrite(fo, "H %0d %0d %016h\n", cur_act, cyc, ln_data);
                        else if (ln_last) $fwrite(fo, "L %0d %0d\n", cur_act, cyc);
                        #1 s_tvalid = 0;
                        disable wait_accept;
                    end
                    #1;
                end
            end
            s_tvalid = 0;
            if (!kernel_done || 1) next_line;
        end
        s_tvalid = 0;
        while (have_line && ln_act == cur_act) next_line;   // host resets the stream path
        feed_busy = 0;
    end

    task axil_write(input [5:0] a, input [31:0] d);
        begin
            @(posedge clk); #1 awaddr = a; wdata = d; awvalid = 1; wvalid = 1; bready = 1;
            @(posedge clk); while (!(awready && wready)) @(posedge clk);
            #1 awvalid = 0; wvalid = 0;
            while (!bvalid) begin @(posedge clk); #1; end
            @(posedge clk); #1 bready = 0;
        end
    endtask

    task axil_read(input [5:0] a, output [31:0] d);
        begin
            @(posedge clk); #1 araddr = a; arvalid = 1;
            @(posedge clk); while (!arready) @(posedge clk);
            #1 arvalid = 0; rready = 1;
            while (!rvalid) begin @(posedge clk); #1; end
            d = rdata;
            @(posedge clk); #1 rready = 0;
        end
    endtask

    reg [31:0] r, st, er, tx, nc; reg [63:0] t0;
    initial begin
        if (!$value$plusargs("in=%s", fin)) fin = "build/in.txt";
        if (!$value$plusargs("out=%s", fout)) fout = "build/out.txt";
        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        if (!$value$plusargs("bubble=%d", bubble)) bubble = 0;
        if (!$value$plusargs("stall=%d", stall)) stall = 0;
        if (!$value$plusargs("maxcyc=%d", maxcyc)) maxcyc = 64'd2000000000;
        fi = $fopen(fin, "r"); fo = $fopen(fout, "w");
        if (fi == 0) begin $display("cannot open %0s", fin); $finish; end
        repeat (4) @(posedge clk); #1 rst_n = 1;
        next_line;
        while (have_line) begin
            cur_act = ln_act; kernel_done = 0; t0 = cyc;
            axil_write(6'h00, 32'd1);
            -> feed_ev;
            r = 0;
            while (!r[1]) begin
                axil_read(6'h00, r);
                if (cyc > maxcyc) begin $display("TIMEOUT act %0d", cur_act); $fclose(fo); $finish; end
            end
            kernel_done = 1;
            wait (!feed_busy);
            axil_read(6'h10, st); axil_read(6'h14, er); axil_read(6'h18, tx); axil_read(6'h1C, nc);
            $fwrite(fo, "S %0d %0d %0d %0d %0d %0d\n", cur_act, st[0], er, tx, nc, cyc - t0);
            repeat (2) @(posedge clk);
        end
        $fclose(fo);
        $display("tb_wm_top: done at cycle %0d", cyc);
        $finish;
    end
endmodule
