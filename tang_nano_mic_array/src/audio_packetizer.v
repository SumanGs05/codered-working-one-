// Audio Packetizer — ping-pong buffer + MIC-header serial framing
//
// Collects CHUNK frames of N_CH channels into one memory bank while
// simultaneously transmitting the other bank over UART.  Uses the
// same packet format as the ESP32-S3 stream_audio.ino:
//
//   Header (6 bytes):  'M' 'I' 'C'  chunk_lo  chunk_hi  n_channels
//   Payload:           CHUNK * N_CH * 2 bytes  (int16 LE per sample)
//
// Memory is a dual-port BSRAM, split into two banks (ping/pong)
// via the MSB of the address.  Write port fills one bank, read
// port drains the other — no contention.
module audio_packetizer #(
    parameter CHUNK = 512,
    parameter N_CH  = 7
)(
    input  wire        clk,
    input  wire        rst_n,

    // from I2S receiver
    input  wire        frame_valid,
    input  wire [15:0] ch0, ch1, ch2, ch3, ch4, ch5, ch6,

    // to UART TX
    output reg  [7:0]  tx_byte,
    output reg         tx_start,
    input  wire        tx_busy,

    // status
    output wire        led_sending
);

    // Words per bank = 512 × 7 = 3584.
    // Each bank occupies addresses 0..4095 (12 bits), 1 bank-select bit → 13 bits.
    localparam WPB  = CHUNK * N_CH;       // 3584
    localparam AW   = 13;                 // address width (8192 words total)

    // ----------------------------------------------------------------
    //  Dual-port RAM  (write port ← I2S, read port → UART sender)
    // ----------------------------------------------------------------
    reg              mem_wr_en;
    reg  [AW-1:0]   mem_wr_addr;
    reg  [15:0]     mem_wr_data;
    reg  [AW-1:0]   mem_rd_addr;
    wire [15:0]     mem_rd_data;

    dpram #(.ADDR_W(AW), .DATA_W(16)) u_ram (
        .clk     (clk),
        .wr_en   (mem_wr_en),
        .wr_addr (mem_wr_addr),
        .wr_data (mem_wr_data),
        .rd_addr (mem_rd_addr),
        .rd_data (mem_rd_data)
    );

    // ----------------------------------------------------------------
    //  Write-side registers
    // ----------------------------------------------------------------
    reg        wr_bank;            // which bank we are filling (0 or 1)
    reg [11:0] wr_ptr;             // word offset inside current bank
    reg [9:0]  wr_frame;           // frames written to current bank
    reg        wr_active;          // 7-cycle write burst in progress
    reg [2:0]  wr_ch;              // channel index during burst

    reg [15:0] ws0, ws1, ws2, ws3, ws4, ws5, ws6;  // latched samples

    // ----------------------------------------------------------------
    //  Send-side registers
    // ----------------------------------------------------------------
    reg        send_trigger;       // set by write-side, consumed by send FSM
    reg        send_active;        // send FSM is running
    reg        send_bank;          // bank being transmitted

    localparam SS_IDLE      = 4'd0,
               SS_HDR       = 4'd1,
               SS_WAIT_ACK  = 4'd2,
               SS_WAIT_DONE = 4'd3,
               SS_RD_SETUP  = 4'd4,
               SS_RD_LATCH  = 4'd5,
               SS_SEND_LO   = 4'd6,
               SS_SEND_HI   = 4'd7;

    reg [3:0]  ss;                 // send state
    reg [3:0]  ss_ret;             // return state after WAIT_DONE
    reg [2:0]  hdr_idx;
    reg [11:0] rd_ptr;             // word offset inside send bank
    reg [15:0] rd_word;            // latched read data

    assign led_sending = send_active;

    // ----------------------------------------------------------------
    //  Unified always block
    // ----------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // write side
            wr_bank      <= 1'b0;
            wr_ptr       <= 12'd0;
            wr_frame     <= 10'd0;
            wr_active    <= 1'b0;
            wr_ch        <= 3'd0;
            ws0 <= 0; ws1 <= 0; ws2 <= 0; ws3 <= 0;
            ws4 <= 0; ws5 <= 0; ws6 <= 0;
            mem_wr_en    <= 1'b0;
            mem_wr_addr  <= 0;
            mem_wr_data  <= 0;
            // send side
            send_trigger <= 1'b0;
            send_active  <= 1'b0;
            send_bank    <= 1'b0;
            ss           <= SS_IDLE;
            ss_ret       <= SS_IDLE;
            hdr_idx      <= 3'd0;
            rd_ptr       <= 12'd0;
            rd_word      <= 16'd0;
            mem_rd_addr  <= 0;
            tx_byte      <= 8'd0;
            tx_start     <= 1'b0;
        end else begin

            // defaults each cycle
            tx_start  <= 1'b0;
            mem_wr_en <= 1'b0;

            // ========================================================
            //  WRITE LOGIC  (runs every cycle, independent of send)
            // ========================================================

            // Latch new frame when not already writing
            if (frame_valid && !wr_active) begin
                ws0 <= ch0;  ws1 <= ch1;  ws2 <= ch2;  ws3 <= ch3;
                ws4 <= ch4;  ws5 <= ch5;  ws6 <= ch6;
                wr_active <= 1'b1;
                wr_ch     <= 3'd0;
            end

            // 7-cycle burst: one channel per clock
            if (wr_active) begin
                mem_wr_en   <= 1'b1;
                mem_wr_addr <= {wr_bank, wr_ptr};

                case (wr_ch)
                    3'd0: mem_wr_data <= ws0;
                    3'd1: mem_wr_data <= ws1;
                    3'd2: mem_wr_data <= ws2;
                    3'd3: mem_wr_data <= ws3;
                    3'd4: mem_wr_data <= ws4;
                    3'd5: mem_wr_data <= ws5;
                    3'd6: mem_wr_data <= ws6;
                    default: mem_wr_data <= 16'd0;
                endcase

                wr_ptr <= wr_ptr + 12'd1;

                if (wr_ch == 3'd6) begin
                    wr_active <= 1'b0;
                    if (wr_frame == CHUNK - 1) begin
                        // bank full
                        wr_frame <= 10'd0;
                        wr_ptr   <= 12'd0;
                        if (!send_active) begin
                            send_trigger <= 1'b1;
                            send_bank    <= wr_bank;
                            wr_bank      <= ~wr_bank;
                        end else begin
                            // send still running — overwrite same bank (overflow)
                            // should not happen at 3 Mbaud / 16 kHz
                        end
                    end else begin
                        wr_frame <= wr_frame + 10'd1;
                    end
                end else begin
                    wr_ch <= wr_ch + 3'd1;
                end
            end

            // ========================================================
            //  SEND FSM  (drains one bank over UART)
            // ========================================================

            case (ss)

            SS_IDLE: begin
                if (send_trigger) begin
                    send_trigger <= 1'b0;
                    send_active  <= 1'b1;
                    hdr_idx      <= 3'd0;
                    ss           <= SS_HDR;
                end
            end

            // --- header bytes ---
            SS_HDR: begin
                if (!tx_busy) begin
                    case (hdr_idx)
                        3'd0: tx_byte <= 8'h4D;            // 'M'
                        3'd1: tx_byte <= 8'h49;            // 'I'
                        3'd2: tx_byte <= 8'h43;            // 'C'
                        3'd3: tx_byte <= CHUNK[7:0];       // samples low
                        3'd4: tx_byte <= CHUNK[15:8];      // samples high
                        3'd5: tx_byte <= N_CH[7:0];        // channels
                        default: tx_byte <= 8'd0;
                    endcase
                    tx_start <= 1'b1;
                    ss       <= SS_WAIT_ACK;

                    if (hdr_idx == 3'd5) begin
                        ss_ret <= SS_RD_SETUP;
                        rd_ptr <= 12'd0;
                    end else begin
                        ss_ret  <= SS_HDR;
                        hdr_idx <= hdr_idx + 3'd1;
                    end
                end
            end

            // --- handshake: wait for UART busy to rise then fall ---
            SS_WAIT_ACK: begin
                if (tx_busy) ss <= SS_WAIT_DONE;
            end

            SS_WAIT_DONE: begin
                if (!tx_busy) ss <= ss_ret;
            end

            // --- data bytes (read word, send low then high) ---
            SS_RD_SETUP: begin
                mem_rd_addr <= {send_bank, rd_ptr};
                ss          <= SS_RD_LATCH;
            end

            SS_RD_LATCH: begin
                // 1-cycle read latency for dpram
                ss <= SS_SEND_LO;
            end

            SS_SEND_LO: begin
                rd_word <= mem_rd_data;   // capture stable read
                if (!tx_busy) begin
                    tx_byte  <= mem_rd_data[7:0];
                    tx_start <= 1'b1;
                    ss       <= SS_WAIT_ACK;
                    ss_ret   <= SS_SEND_HI;
                end
            end

            SS_SEND_HI: begin
                if (!tx_busy) begin
                    tx_byte  <= rd_word[15:8];
                    tx_start <= 1'b1;
                    ss       <= SS_WAIT_ACK;

                    if (rd_ptr == WPB - 1) begin
                        ss_ret      <= SS_IDLE;
                        send_active <= 1'b0;
                    end else begin
                        rd_ptr <= rd_ptr + 12'd1;
                        ss_ret <= SS_RD_SETUP;
                    end
                end
            end

            default: ss <= SS_IDLE;

            endcase
        end
    end

endmodule
