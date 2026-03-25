// Top-level module for Tang Nano 9K ↔ Sipeed 6+1 Mic Array
//
// Captures all 7 microphone channels via I2S, buffers them in
// ping-pong BSRAM, and streams MIC-header packets over the
// onboard USB-UART (BL702) at 3 Mbaud.
//
// Data flow:
//   27 MHz ──► i2s_clock_gen ──► BCLK / WS  ──► mic array
//                                    │
//              mic D0-D3 ──► i2s_receiver ──► 7 × 16-bit samples
//                                                   │
//                              audio_packetizer ◄───┘
//                                    │
//                                uart_tx ──► USB-UART ──► laptop / Pi
module top (
    input  wire       sys_clk,     // 27 MHz crystal oscillator
    input  wire       btn_rst_n,   // Button S1 — active-low reset

    // Sipeed mic array
    output wire       mic_bclk,    // I2S bit clock
    output wire       mic_ws,      // I2S word select
    input  wire       mic_d0,      // data line 0 (mics 0,1)
    input  wire       mic_d1,      // data line 1 (mics 2,3)
    input  wire       mic_d2,      // data line 2 (mics 4,5)
    input  wire       mic_d3,      // data line 3 (mic 6, center)

    // USB-UART via BL702
    output wire       uart_txd,

    // Onboard LEDs (active-low)
    output wire [5:0] led
);

    // ----------------------------------------------------------------
    //  Power-on reset  (~1.2 ms at 27 MHz)
    // ----------------------------------------------------------------
    reg [15:0] por_cnt = 16'd0;
    always @(posedge sys_clk)
        if (!por_cnt[15])
            por_cnt <= por_cnt + 16'd1;

    reg [1:0] btn_sync;
    always @(posedge sys_clk)
        btn_sync <= {btn_sync[0], btn_rst_n};

    wire rst_n = por_cnt[15] & btn_sync[1];

    // ----------------------------------------------------------------
    //  I2S clock generator  (BCLK ≈ 1.038 MHz, Fs ≈ 16.2 kHz)
    // ----------------------------------------------------------------
    wire       bclk_w, ws_w, bclk_re;
    wire [5:0] bit_cnt;

    i2s_clock_gen u_clkgen (
        .clk     (sys_clk),
        .rst_n   (rst_n),
        .bclk    (bclk_w),
        .ws      (ws_w),
        .bclk_re (bclk_re),
        .bit_cnt (bit_cnt)
    );

    assign mic_bclk = bclk_w;
    assign mic_ws   = ws_w;

    // ----------------------------------------------------------------
    //  I2S receiver — 4 data lines → 7 channels
    // ----------------------------------------------------------------
    wire [15:0] ch0, ch1, ch2, ch3, ch4, ch5, ch6;
    wire        frame_valid;

    i2s_receiver u_rx (
        .clk         (sys_clk),
        .rst_n       (rst_n),
        .bclk_re     (bclk_re),
        .bit_cnt     (bit_cnt),
        .sd0         (mic_d0),
        .sd1         (mic_d1),
        .sd2         (mic_d2),
        .sd3         (mic_d3),
        .ch0 (ch0), .ch1 (ch1), .ch2 (ch2), .ch3 (ch3),
        .ch4 (ch4), .ch5 (ch5), .ch6 (ch6),
        .frame_valid (frame_valid)
    );

    // ----------------------------------------------------------------
    //  UART transmitter  (3 Mbaud, exact divider from 27 MHz)
    // ----------------------------------------------------------------
    wire [7:0] pkt_tx_byte;
    wire       pkt_tx_start;
    wire       tx_busy;

    uart_tx #(
        .CLK_FREQ (27_000_000),
        .BAUD     (3_000_000)
    ) u_uart (
        .clk   (sys_clk),
        .rst_n (rst_n),
        .data  (pkt_tx_byte),
        .start (pkt_tx_start),
        .tx    (uart_txd),
        .busy  (tx_busy)
    );

    // ----------------------------------------------------------------
    //  Audio packetizer  (ping-pong BSRAM + MIC-header framing)
    // ----------------------------------------------------------------
    wire led_sending;

    audio_packetizer #(
        .CHUNK (512),
        .N_CH  (7)
    ) u_pkt (
        .clk         (sys_clk),
        .rst_n       (rst_n),
        .frame_valid (frame_valid),
        .ch0 (ch0), .ch1 (ch1), .ch2 (ch2), .ch3 (ch3),
        .ch4 (ch4), .ch5 (ch5), .ch6 (ch6),
        .tx_byte     (pkt_tx_byte),
        .tx_start    (pkt_tx_start),
        .tx_busy     (tx_busy),
        .led_sending (led_sending)
    );

    // ----------------------------------------------------------------
    //  LED indicators  (active-low: 0 = ON)
    // ----------------------------------------------------------------
    reg [23:0] heartbeat;
    always @(posedge sys_clk)
        heartbeat <= heartbeat + 24'd1;

    assign led[0] = ~(~rst_n);          // ON during reset
    assign led[1] = ~led_sending;       // ON while packet is being sent
    assign led[2] = ~heartbeat[23];     // ~3 Hz heartbeat
    assign led[3] = ~heartbeat[22];     // ~6 Hz
    assign led[4] = 1'b1;              // OFF (spare)
    assign led[5] = 1'b1;              // OFF (spare)

endmodule
