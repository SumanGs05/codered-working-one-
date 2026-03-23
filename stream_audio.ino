// this is the code for esp 32 s3
#include <driver/i2s.h>

#define SAMPLE_RATE     16000
#define CHANNELS        7
#define CHUNK_SAMPLES   512
#define DMA_BUF_LEN     256
#define DMA_BUF_COUNT   8 


#define MIC_WS_PIN      1
#define MIC_CK_PIN      2
#define MIC_D0_PIN      3


int32_t i2s_buffer[DMA_BUF_LEN * 2];
int16_t output_buffer[CHUNK_SAMPLES * CHANNELS];
size_t output_idx = 0;


unsigned long packets_sent = 0;
bool led_state = false;

void setup() {
    Serial.begin(2000000);
    Serial.setTimeout(10);
    
    // Small delay for serial to stabilize
    delay(500);

    Serial.println("ZEUS IS STARTING");
    Serial.flush();
    delay(100);
    
    // Configure I2S
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = DMA_BUF_COUNT,
        .dma_buf_len = DMA_BUF_LEN,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    
    esp_err_t err = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.println("ERR");
        while(1) delay(1000);
    }
    
    i2s_pin_config_t pin_config = {
        .bck_io_num = MIC_CK_PIN,
        .ws_io_num = MIC_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = MIC_D0_PIN
    };
    
    err = i2s_set_pin(I2S_NUM_0, &pin_config);
    if (err != ESP_OK) {
        Serial.println("ERR");
        while(1) delay(1000);
    }
    
    i2s_zero_dma_buffer(I2S_NUM_0);
    
    Serial.println("START");
    Serial.flush();
    
    delay(1000);
    
    
    pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
    size_t bytes_read;
    
    // Read I2S data
    esp_err_t result = i2s_read(I2S_NUM_0, i2s_buffer, 
                                sizeof(i2s_buffer), &bytes_read, portMAX_DELAY);
    
    // to handle fail
    if (result != ESP_OK || bytes_read == 0) {
        return;
    }
    
    size_t samples = bytes_read / 8;
    
    // to convert and accumulate
    for (size_t i = 0; i < samples && output_idx < CHUNK_SAMPLES; i++) {
        int32_t left = i2s_buffer[i * 2];
        int32_t right = i2s_buffer[i * 2 + 1];
        int16_t left16 = (int16_t)(left >> 16);
        int16_t right16 = (int16_t)(right >> 16);
        
        size_t base = output_idx * CHANNELS;
        output_buffer[base + 0] = left16;
        output_buffer[base + 1] = right16;
        output_buffer[base + 2] = left16;
        output_buffer[base + 3] = right16;
        output_buffer[base + 4] = left16;
        output_buffer[base + 5] = right16;
        output_buffer[base + 6] = (left16 + right16) / 2;
        
        output_idx++;
    }
    
    // Send when chunk is full
    if (output_idx >= CHUNK_SAMPLES) {

        uint8_t header[6];
        header[0] = 'M';
        header[1] = 'I';
        header[2] = 'C';
        header[3] = (uint8_t)(CHUNK_SAMPLES & 0xFF);
        header[4] = (uint8_t)((CHUNK_SAMPLES >> 8) & 0xFF);
        header[5] = (uint8_t)CHANNELS;
        
        Serial.write(header, 6);
        Serial.write((uint8_t*)output_buffer, CHUNK_SAMPLES * CHANNELS * 2);
    
        
        output_idx = 0;
        packets_sent++;
        if (packets_sent % 30 == 0) {
            led_state = !led_state;
            digitalWrite(LED_BUILTIN, led_state);
        }
    }
}

