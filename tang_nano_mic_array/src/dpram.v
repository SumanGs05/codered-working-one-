// Simple Dual-Port RAM — one write port, one read port
// Coded for clean Gowin BSRAM inference
module dpram #(
    parameter ADDR_W = 13,
    parameter DATA_W = 16
)(
    input  wire                  clk,
    input  wire                  wr_en,
    input  wire [ADDR_W-1:0]    wr_addr,
    input  wire [DATA_W-1:0]    wr_data,
    input  wire [ADDR_W-1:0]    rd_addr,
    output reg  [DATA_W-1:0]    rd_data
);

    localparam DEPTH = 1 << ADDR_W;

    reg [DATA_W-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
        rd_data <= mem[rd_addr];
    end

endmodule
