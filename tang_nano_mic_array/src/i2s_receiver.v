// 4-Line I2S Receiver for Sipeed 6+1 Microphone Array
//
// Captures data from all four data lines simultaneously:
//   D0 → mic 0 (left)  + mic 1 (right)
//   D1 → mic 2 (left)  + mic 3 (right)
//   D2 → mic 4 (left)  + mic 5 (right)
//   D3 → mic 6 center  (left slot)
//
// Outputs 7 × 16-bit samples per frame with a frame_valid strobe.
// Accounts for the standard I2S 1-bit delay: MSB appears on the
// second rising BCLK edge after a WS transition.
module i2s_receiver (
    input  wire        clk,       // 27 MHz system clock
    input  wire        rst_n,
    input  wire        bclk_re,   // strobe on BCLK rising edge
    input  wire [5:0]  bit_cnt,   // 0-63 frame position
    input  wire        sd0,       // data line 0
    input  wire        sd1,       // data line 1
    input  wire        sd2,       // data line 2
    input  wire        sd3,       // data line 3
    output reg  [15:0] ch0,       // mic 0 – D0 left
    output reg  [15:0] ch1,       // mic 1 – D0 right
    output reg  [15:0] ch2,       // mic 2 – D1 left
    output reg  [15:0] ch3,       // mic 3 – D1 right
    output reg  [15:0] ch4,       // mic 4 – D2 left
    output reg  [15:0] ch5,       // mic 5 – D2 right
    output reg  [15:0] ch6,       // mic 6 – D3 left (center)
    output reg         frame_valid
);

    // 2-stage synchronizer for the four asynchronous data lines
    reg [3:0] sd_meta, sd_sync;
    always @(posedge clk) begin
        sd_meta <= {sd3, sd2, sd1, sd0};
        sd_sync <= sd_meta;
    end

    // 32-bit shift registers, one per data line
    reg [31:0] sr0, sr1, sr2, sr3;

    // Latched left-channel results (held until right-channel completes)
    reg [15:0] lat_l0, lat_l1, lat_l2, lat_l3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sr0 <= 32'd0;  sr1 <= 32'd0;
            sr2 <= 32'd0;  sr3 <= 32'd0;
            lat_l0 <= 16'd0; lat_l1 <= 16'd0;
            lat_l2 <= 16'd0; lat_l3 <= 16'd0;
            ch0 <= 16'd0; ch1 <= 16'd0; ch2 <= 16'd0; ch3 <= 16'd0;
            ch4 <= 16'd0; ch5 <= 16'd0; ch6 <= 16'd0;
            frame_valid <= 1'b0;
        end else begin
            frame_valid <= 1'b0;

            if (bclk_re) begin
                // Shift in one bit per data line (MSB first)
                sr0 <= {sr0[30:0], sd_sync[0]};
                sr1 <= {sr1[30:0], sd_sync[1]};
                sr2 <= {sr2[30:0], sd_sync[2]};
                sr3 <= {sr3[30:0], sd_sync[3]};

                // --- end of LEFT phase (32 bits shifted at bit_cnt 0-31) ---
                // At bit_cnt 32 the shift regs hold all left-channel bits.
                // I2S 1-bit delay: MSB is at sr[30], top-16 audio = sr[30:15].
                // Non-blocking reads see the value BEFORE this cycle's shift,
                // which already contains bits 0-31 (the complete left word).
                if (bit_cnt == 6'd32) begin
                    lat_l0 <= sr0[30:15];
                    lat_l1 <= sr1[30:15];
                    lat_l2 <= sr2[30:15];
                    lat_l3 <= sr3[30:15];
                end

                // --- end of RIGHT phase (32 bits shifted at bit_cnt 32-63) ---
                // At bit_cnt 0 (start of next frame) the shift regs hold
                // all right-channel bits from the previous frame.
                if (bit_cnt == 6'd0) begin
                    ch0 <= lat_l0;             // D0 left
                    ch1 <= sr0[30:15];         // D0 right
                    ch2 <= lat_l1;             // D1 left
                    ch3 <= sr1[30:15];         // D1 right
                    ch4 <= lat_l2;             // D2 left
                    ch5 <= sr2[30:15];         // D2 right
                    ch6 <= lat_l3;             // D3 left (center mic)
                    frame_valid <= 1'b1;
                end
            end
        end
    end

endmodule
