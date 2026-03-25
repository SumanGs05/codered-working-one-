// I2S Master Clock Generator
// Derives BCLK and WS from the 27 MHz system clock.
//
// Default: BCLK = 27 MHz / (2*13) ≈ 1.038 MHz
//          WS   = BCLK / 64       ≈ 16.2 kHz  (sample rate)
//
// MSM261S4030H0 mic accepts SCK 1.0–4.0 MHz, so this is in range.
module i2s_clock_gen #(
    parameter BCLK_HALF = 13   // half-period in sys_clk cycles
)(
    input  wire       clk,     // 27 MHz system clock
    input  wire       rst_n,
    output reg        bclk,    // I2S bit clock (directly to mic array)
    output reg        ws,      // I2S word select / LRCLK
    output reg        bclk_re, // single-cycle strobe on BCLK rising edge
    output reg  [5:0] bit_cnt  // position within 64-bit I2S frame (0-63)
);

    reg [7:0] div_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            div_cnt  <= 8'd0;
            bclk     <= 1'b0;
            ws       <= 1'b0;
            bit_cnt  <= 6'd0;
            bclk_re  <= 1'b0;
        end else begin
            bclk_re <= 1'b0;

            if (div_cnt == BCLK_HALF - 1) begin
                div_cnt <= 8'd0;
                bclk    <= ~bclk;

                if (!bclk) begin
                    // BCLK about to go HIGH → rising-edge strobe
                    bclk_re <= 1'b1;
                end else begin
                    // BCLK about to go LOW → falling edge
                    // I2S spec: WS and bit_cnt advance on falling edge
                    if (bit_cnt == 6'd63) begin
                        bit_cnt <= 6'd0;
                        ws      <= 1'b0;   // left channel
                    end else begin
                        bit_cnt <= bit_cnt + 6'd1;
                        if (bit_cnt == 6'd31)
                            ws <= 1'b1;     // right channel
                    end
                end
            end else begin
                div_cnt <= div_cnt + 8'd1;
            end
        end
    end

endmodule
