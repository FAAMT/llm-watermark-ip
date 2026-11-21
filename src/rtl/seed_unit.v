// seed_unit: once per SAMPLE, the keyed seed s = fmix64(fold(K, context)) and
// the unkeyed seed s_f = fmix64(nonce ^ (step + 1) * C_3). One fmix64 pipeline
// is shared: the keyed input enters one cycle before the free one.
// ready rises 4 + LAT64 cycles after start. Golden: step_seed(), free_seed().
module seed_unit #(parameter MUL_REG = 1) (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [63:0] key,
    input  wire [63:0] nonce,
    input  wire [31:0] step,
    input  wire [23:0] c0, c1, c2, c3,
    input  wire [2:0]  k,
    output reg         ready,
    output reg  [31:0] s_lo,
    output reg  [31:0] sf_lo
);
    localparam [63:0] C3 = 64'h2545F4914F6CDD1D;
    localparam integer LAT64 = 3 + 2 * MUL_REG;

    wire [63:0] xk;
    ctx_fold u_fold (.clk(clk), .key(key), .c0(c0), .c1(c1), .c2(c2), .c3(c3), .k(k), .x(xk));

    reg [63:0] tf, nonce_r, xf, xf_d;
    always @(posedge clk) begin
        tf      <= {32'd0, step} * C3 + C3;      // (step + 1) * C3, operand kept 32 bits
        nonce_r <= nonce;
        xf      <= nonce_r ^ tf;               // same 2-cycle latency as ctx_fold
        xf_d    <= xf;
    end

    // st[n] is start delayed by n cycles
    reg [LAT64+4:0] st;
    always @(posedge clk) st <= rst ? 0 : {st[LAT64+3:0], start};

    wire [63:0] h_in = st[2] ? xk : xf_d;      // cycle 2: keyed; cycle 3: free
    wire [63:0] h_out;
    fmix64_pipe #(MUL_REG) u_fmix (.clk(clk), .din(h_in), .dout(h_out));

    always @(posedge clk) begin
        if (rst || start) ready <= 1'b0;
        if (st[2 + LAT64]) s_lo  <= h_out[31:0];
        if (st[3 + LAT64]) begin sf_lo <= h_out[31:0]; ready <= 1'b1; end
    end
endmodule
