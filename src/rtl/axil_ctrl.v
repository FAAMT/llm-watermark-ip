// AXI4-Lite control and status, the course command-response convention:
// control only, no configuration. One outstanding transaction per channel.
//   0x00 CTRL   W: bit0 = ap_start.  R: bit0 ap_start (busy), bit1 ap_done
//               (clear on read), bit2 ap_idle
//   0x10 STATUS R: bit0 halted (the activation ended in an error)
//   0x14 ERROR  R: error code, 0 = none
//   0x18 TX_ID  R: tx_id of the last command read (the failing one on error)
//   0x1C NCMD   R: commands completed in this activation
module axil_ctrl (
    input  wire        clk,
    input  wire        rst,
    input  wire [5:0]  awaddr,
    input  wire        awvalid,
    output wire        awready,
    input  wire [31:0] wdata,
    input  wire [3:0]  wstrb,
    input  wire        wvalid,
    output wire        wready,
    output wire [1:0]  bresp,
    output reg         bvalid,
    input  wire        bready,
    input  wire [5:0]  araddr,
    input  wire        arvalid,
    output wire        arready,
    output reg  [31:0] rdata,
    output wire [1:0]  rresp,
    output reg         rvalid,
    input  wire        rready,
    // kernel side
    output reg         ap_start,     // one-cycle pulse
    input  wire        busy,
    input  wire        done_pulse,
    input  wire        halted,
    input  wire [7:0]  err,
    input  wire [15:0] tx_id,
    input  wire [31:0] ncmd
);
    wire wr = awvalid && wvalid && !bvalid;
    assign awready = wr;
    assign wready  = wr;
    assign bresp   = 2'b00;
    assign arready = !rvalid;
    assign rresp   = 2'b00;

    reg done_flag;
    always @(posedge clk) begin
        ap_start <= 1'b0;
        if (rst) begin
            bvalid <= 1'b0; rvalid <= 1'b0; done_flag <= 1'b0;
        end else begin
            if (wr) begin
                bvalid <= 1'b1;
                if (awaddr == 6'h00 && wstrb[0] && wdata[0] && !busy) begin
                    ap_start  <= 1'b1;
                    done_flag <= 1'b0;
                end
            end else if (bvalid && bready) begin
                bvalid <= 1'b0;
            end
            if (done_pulse) done_flag <= 1'b1;
            if (arvalid && !rvalid) begin
                rvalid <= 1'b1;
                case (araddr)
                    6'h00: begin
                        rdata <= {29'd0, !busy, done_flag, busy};
                        if (!done_pulse) done_flag <= 1'b0;     // clear on read
                    end
                    6'h10: rdata <= {31'd0, halted};
                    6'h14: rdata <= {24'd0, err};
                    6'h18: rdata <= {16'd0, tx_id};
                    6'h1C: rdata <= ncmd;
                    default: rdata <= 32'd0;
                endcase
            end else if (rvalid && rready) begin
                rvalid <= 1'b0;
            end
        end
    end
endmodule
