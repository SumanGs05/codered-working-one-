// UART Transmitter — 8N1
//
// Default: 3 Mbaud from 27 MHz clock (divider = 9, exact).
// Pulse `start` for one cycle with `data` valid.  `busy` is high
// from the cycle after start until the stop bit completes.
module uart_tx #(
    parameter CLK_FREQ = 27_000_000,
    parameter BAUD     = 3_000_000
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] data,
    input  wire       start,
    output reg        tx,
    output reg        busy
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD;  // 9 at default

    reg [3:0]  bit_idx;   // 0-9: start(0) data(1-8) stop(9)
    reg [7:0]  shift;
    reg [$clog2(CLKS_PER_BIT+1)-1:0] tick_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx       <= 1'b1;
            busy     <= 1'b0;
            bit_idx  <= 4'd0;
            tick_cnt <= 0;
            shift    <= 8'd0;
        end else if (!busy) begin
            tx <= 1'b1;
            if (start) begin
                busy     <= 1'b1;
                shift    <= data;
                bit_idx  <= 4'd0;
                tick_cnt <= 0;
                tx       <= 1'b0;       // start bit
            end
        end else begin
            if (tick_cnt < CLKS_PER_BIT - 1) begin
                tick_cnt <= tick_cnt + 1;
            end else begin
                tick_cnt <= 0;
                if (bit_idx == 4'd9) begin
                    busy <= 1'b0;
                    tx   <= 1'b1;
                end else begin
                    bit_idx <= bit_idx + 4'd1;
                    if (bit_idx < 4'd8) begin
                        tx    <= shift[0];
                        shift <= {1'b0, shift[7:1]};
                    end else begin
                        tx <= 1'b1;      // stop bit
                    end
                end
            end
        end
    end

endmodule
