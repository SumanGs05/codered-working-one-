#!/usr/bin/env python3
"""
2-Mic MVDR Beamforming for D0 Only (ESP32 sends 7ch but only 0-1 real)
Frequency-domain MVDR with noise reduction
Optimized for Raspberry Pi 4B
"""

import serial
import struct
import numpy as np
import pyaudio
from scipy.signal import stft, istft, butter, sosfilt
from collections import deque
import threading
import time
import sys

# Configuration
SERIAL_PORT = '/dev/ttyACM0'
SERIAL_BAUD = 2000000

SAMPLE_RATE = 16000
CHANNELS = 7  # ESP32 sends 7
REAL_MICS = 2  # Only Ch0-1 are real!
CHUNK_SIZE = 512

# STFT parameters
NFFT = 256  # Smaller FFT for lower latency
HOP_LENGTH = 128

# 2-Mic linear geometry
MIC_SPACING = 0.045  # 45mm
MIC_POSITIONS = np.array([
    [-MIC_SPACING/2, 0, 0],  # Ch0: Left
    [MIC_SPACING/2, 0, 0]    # Ch1: Right
])

SPEED_OF_SOUND = 343.0
FREQ_MIN = 300   # Hz - speech band start
FREQ_MAX = 4000  # Hz - speech band end
REGULARIZATION = 1e-3
HIGH_PASS_CUTOFF = 100  # Remove low-frequency electronic noise

# Global state
current_azimuth = 0.0
audio_output = None
running = True
stats = {'packets': 0, 'errors': 0}


class TwoMicMVDR:
    """2-Mic MVDR with noise reduction"""
    
    def __init__(self, mic_positions, sample_rate):
        self.mic_positions = mic_positions
        self.n_mics = 2
        self.sample_rate = sample_rate
        self.mic_spacing = np.linalg.norm(mic_positions[1] - mic_positions[0])
        
        self.nfft = NFFT
        self.hop_length = HOP_LENGTH
        self.window = np.hanning(self.nfft)
        
        # Frequency bins
        self.freqs = np.fft.rfftfreq(self.nfft, 1/sample_rate)
        self.n_freqs = len(self.freqs)
        
        # Frequency indices for beamforming
        self.freq_min_idx = np.argmin(np.abs(self.freqs - FREQ_MIN))
        self.freq_max_idx = np.argmin(np.abs(self.freqs - FREQ_MAX))
        
        # Noise covariance (2x2 per frequency)
        self.noise_cov = np.zeros((self.n_freqs, 2, 2), dtype=complex)
        for f in range(self.n_freqs):
            self.noise_cov[f] = np.eye(2) * REGULARIZATION
        
        self.noise_buffer = deque(maxlen=30)
        self.noise_update_counter = 0
        
        # High-pass filter for noise reduction
        self.sos_hp = butter(4, HIGH_PASS_CUTOFF, 'highpass', 
                            fs=sample_rate, output='sos')
        self.filter_state_0 = None
        self.filter_state_1 = None
        
        print(f"  2-Mic MVDR Configuration:")
        print(f"    Spacing: {self.mic_spacing*1000:.1f} mm")
        print(f"    FFT size: {self.nfft}")
        print(f"    Beamforming: {FREQ_MIN}-{FREQ_MAX} Hz")
        print(f"    High-pass filter: {HIGH_PASS_CUTOFF} Hz")
    
    def steering_vector(self, azimuth_deg, freq_hz):
        """Compute 2-mic steering vector"""
        az_rad = np.radians(azimuth_deg)
        direction = np.array([np.cos(az_rad), np.sin(az_rad), 0])
        
        # Time delays
        delays = np.dot(self.mic_positions, direction) / SPEED_OF_SOUND
        
        # Phase shifts
        omega = 2 * np.pi * freq_hz
        steering = np.exp(-1j * omega * delays)
        
        return steering / np.linalg.norm(steering)
    
    def update_noise_covariance(self, stft_data):
        """Update noise estimate from quiet sections"""
        self.noise_buffer.append(stft_data)
        
        if len(self.noise_buffer) >= 10:
            all_frames = np.concatenate(list(self.noise_buffer), axis=2)
            
            for f in range(self.n_freqs):
                X = all_frames[f, :, :]
                self.noise_cov[f] = (X @ X.conj().T) / X.shape[1]
                self.noise_cov[f] += np.eye(2) * REGULARIZATION
    
    def process_chunk(self, audio_chunk, azimuth_deg):
        """Process with MVDR beamforming
        
        Args:
            audio_chunk: (n_samples, 2) - only Ch0 and Ch1
            azimuth_deg: Target direction
        """
        n_samples = audio_chunk.shape[0]
        
        # High-pass filter to remove electronic noise
        filtered = np.zeros_like(audio_chunk)
        
        if self.filter_state_0 is None:
            self.filter_state_0 = np.zeros((self.sos_hp.shape[0], 2))
        if self.filter_state_1 is None:
            self.filter_state_1 = np.zeros((self.sos_hp.shape[0], 2))
        
        filtered[:, 0], self.filter_state_0 = sosfilt(
            self.sos_hp, audio_chunk[:, 0], zi=self.filter_state_0
        )
        filtered[:, 1], self.filter_state_1 = sosfilt(
            self.sos_hp, audio_chunk[:, 1], zi=self.filter_state_1
        )
        
        # STFT for both mics
        freqs, times, Zxx0 = stft(filtered[:, 0], fs=self.sample_rate,
                                  window=self.window, nperseg=self.nfft,
                                  noverlap=self.hop_length)
        _, _, Zxx1 = stft(filtered[:, 1], fs=self.sample_rate,
                         window=self.window, nperseg=self.nfft,
                         noverlap=self.hop_length)
        
        # Stack: (2, n_freqs, n_frames)
        stft_data = np.array([Zxx0, Zxx1])
        # Transpose: (n_freqs, 2, n_frames)
        stft_data = stft_data.transpose(1, 0, 2)
        
        n_frames = stft_data.shape[2]
        
        # Update noise periodically
        self.noise_update_counter += 1
        if self.noise_update_counter % 20 == 0:
            self.update_noise_covariance(stft_data)
        
        # Process each frequency
        output_stft = np.zeros((self.n_freqs, n_frames), dtype=complex)
        
        for f in range(self.n_freqs):
            freq_hz = self.freqs[f]
            
            # Only MVDR in speech band
            if freq_hz < FREQ_MIN or freq_hz > FREQ_MAX:
                # Delay-sum outside speech band
                a = self.steering_vector(azimuth_deg, freq_hz)
                w = a / 2
            else:
                # MVDR beamforming
                X = stft_data[f, :, :]  # (2, n_frames)
                
                # Spatial covariance
                R = (X @ X.conj().T) / n_frames
                R += np.eye(2) * REGULARIZATION
                
                # Steering vector
                a = self.steering_vector(azimuth_deg, freq_hz)
                
                # MVDR: w = (R^-1 @ a) / (a^H @ R^-1 @ a)
                try:
                    R_inv = np.linalg.inv(R)
                    w_num = R_inv @ a
                    w_den = np.conj(a) @ R_inv @ a
                    w = w_num / (w_den + 1e-10)
                except np.linalg.LinAlgError:
                    w = a / 2
            
            # Apply: y = w^H @ X
            output_stft[f, :] = np.conj(w) @ stft_data[f, :, :]
        
        # Inverse STFT
        _, output_time = istft(output_stft, fs=self.sample_rate,
                              window=self.window, nperseg=self.nfft,
                              noverlap=self.hop_length)
        
        # Match length
        if len(output_time) < n_samples:
            output_time = np.pad(output_time, (0, n_samples - len(output_time)))
        else:
            output_time = output_time[:n_samples]
        
        # Normalize
        max_val = np.max(np.abs(output_time))
        if max_val > 0.7:
            output_time = output_time * 0.7 / max_val
        
        return output_time


def read_audio_stream(ser, beamformer):
    """Read from ESP32, extract Ch0-1, apply MVDR"""
    global running, stats, audio_output, current_azimuth
    
    print("Starting audio receiver...")
    
    sync_buffer = bytearray()
    
    def find_header():
        while running:
            byte = ser.read(1)
            if not byte:
                return False
            sync_buffer.append(byte[0])
            if len(sync_buffer) > 3:
                sync_buffer.pop(0)
            if len(sync_buffer) == 3 and bytes(sync_buffer) == b'MIC':
                sync_buffer.clear()
                return True
        return False
    
    while running:
        try:
            if not find_header():
                continue
            
            header = ser.read(3)
            if len(header) != 3:
                stats['errors'] += 1
                continue
            
            n_samples = struct.unpack('<H', header[:2])[0]
            n_channels = header[2]
            
            if n_channels != CHANNELS or n_samples > 2048:
                stats['errors'] += 1
                continue
            
            # Read all 7 channels
            data_size = n_samples * n_channels * 2
            audio_bytes = ser.read(data_size)
            
            if len(audio_bytes) != data_size:
                stats['errors'] += 1
                continue
            
            # Parse
            audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_all = audio_int.reshape(n_samples, n_channels).astype(np.float32) / 32768.0
            
            # EXTRACT ONLY Ch0 and Ch1!
            audio_2mic = audio_all[:, :2]
            
            # MVDR beamform
            output = beamformer.process_chunk(audio_2mic, current_azimuth)
            
            # Output
            output = np.clip(output, -1.0, 1.0)
            output_int16 = (output * 32767).astype(np.int16)
            
            if audio_output:
                audio_output.write(output_int16.tobytes())
            
            stats['packets'] += 1
            
        except Exception as e:
            stats['errors'] += 1
            if stats['errors'] % 100 == 1:
                print(f"\nError: {e}")
            time.sleep(0.01)


def setup_audio():
    global audio_output
    p = pyaudio.PyAudio()
    
    print("\nAudio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
    
    try:
        audio_output = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )
        print(f"✓ Audio output ready")
    except Exception as e:
        print(f"✗ Audio failed: {e}")
    
    return p


def stats_thread():
    global running, stats
    last_packets = 0
    start_time = time.time()
    
    while running:
        time.sleep(2.0)
        packets = stats['packets']
        errors = stats['errors']
        elapsed = time.time() - start_time
        pps = (packets - last_packets) / 2.0
        
        print(f"\r[{elapsed:>6.1f}s] Packets: {packets:>6} | "
              f"Rate: {pps:>5.1f}/s | Errors: {errors:>4} | "
              f"Direction: {current_azimuth:>+5.0f}° | "
              f"Latency: ~{NFFT/SAMPLE_RATE*1000:.0f}ms", 
              end='', flush=True)
        
        last_packets = packets


def control_thread():
    global running, current_azimuth
    
    print("\nControls: -90 (left) | 0 (center) | 90 (right) | 'q' (quit)\n")
    
    while running:
        try:
            cmd = input("Direction: ").strip().lower()
            if cmd == 'q':
                running = False
                break
            try:
                angle = float(cmd)
                if -90 <= angle <= 90:
                    current_azimuth = angle
                    print(f"✓ Beam at {angle}°")
                else:
                    print("⚠ -90 to 90 only")
            except ValueError:
                print("⚠ Invalid")
        except (KeyboardInterrupt, EOFError):
            running = False
            break


def main():
    global running, audio_output
    
    print("="*70)
    print("2-Mic MVDR Beamforming (D0 Only) - Raspberry Pi")
    print("="*70)
    print(f"Configuration:")
    print(f"  Serial: {SERIAL_PORT} @ {SERIAL_BAUD}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  ESP32 sends: {CHANNELS} channels")
    print(f"  Using: Ch0-1 only (2 real mics)")
    print(f"  Algorithm: MVDR + highpass filter")
    
    beamformer = TwoMicMVDR(MIC_POSITIONS, SAMPLE_RATE)
    print("✓ MVDR beamformer ready")
    
    p_audio = setup_audio()
    
    print(f"\nConnecting to {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        print("✓ Serial connected")
    except Exception as e:
        print(f"✗ Serial error: {e}")
        sys.exit(1)
    
    audio_thread = threading.Thread(target=read_audio_stream, 
                                    args=(ser, beamformer), daemon=True)
    stats_thread_obj = threading.Thread(target=stats_thread, daemon=True)
    control_thread_obj = threading.Thread(target=control_thread, daemon=True)
    
    audio_thread.start()
    stats_thread_obj.start()
    control_thread_obj.start()
    
    print("\n" + "="*70)
    print("RUNNING - Press Ctrl+C to stop")
    print("="*70)
    
    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        running = False
    
    time.sleep(1)
    
    if audio_output:
        audio_output.stop_stream()
        audio_output.close()
    p_audio.terminate()
    ser.close()
    
    print("✓ Done")


if __name__ == '__main__':
    main()

import serial
import struct
import numpy as np
import pyaudio
from scipy.signal import butter, sosfilt
import threading
import time
import sys

# Configuration
SERIAL_PORT = '/dev/ttyACM0'
SERIAL_BAUD = 2000000

SAMPLE_RATE = 16000
CHANNELS = 7  # ESP32 sends 7, but only 0-1 are real!
REAL_MICS = 2  # Only using Ch0 and Ch1
CHUNK_SIZE = 512

# 2-Mic linear geometry (D0 stereo)
MIC_SPACING = 0.045  # 45mm between left and right
MIC_POSITIONS = np.array([
    [-MIC_SPACING/2, 0, 0],  # Ch0: Left mic
    [MIC_SPACING/2, 0, 0]    # Ch1: Right mic
])

SPEED_OF_SOUND = 343.0
HIGH_PASS_CUTOFF = 80  # Remove low-frequency noise

# Global state
current_azimuth = 0.0  # -90=left, 0=center, 90=right
audio_output = None
running = True
stats = {'packets': 0, 'errors': 0}


class TwoMicBeamformer:
    def __init__(self, mic_positions, sample_rate):
        self.mic_positions = mic_positions
        self.n_mics = 2
        self.sample_rate = sample_rate
        self.mic_spacing = np.linalg.norm(mic_positions[1] - mic_positions[0])
        
        # High-pass filter
        self.sos_hp = butter(2, HIGH_PASS_CUTOFF, 'highpass', 
                            fs=sample_rate, output='sos')
        self.filter_state_0 = None
        self.filter_state_1 = None
        
        print(f"  2-Mic Configuration:")
        print(f"    Spacing: {self.mic_spacing*1000:.1f} mm")
        print(f"    Using channels: 0 (left) and 1 (right)")
        print(f"    Max delay: {self.mic_spacing/SPEED_OF_SOUND*1000:.2f} ms")
    
    def calculate_delay(self, azimuth_deg):
        """Calculate time delay between mics for direction"""
        az_rad = np.radians(azimuth_deg)
        direction = np.array([np.cos(az_rad), np.sin(az_rad), 0])
        
        delays = np.dot(self.mic_positions, direction) / SPEED_OF_SOUND
        delay_samples = int((delays[1] - delays[0]) * self.sample_rate)
        
        return delay_samples
    
    def process_chunk(self, audio_chunk, azimuth_deg):
        """Process with delay-and-sum
        
        Args:
            audio_chunk: (n_samples, 2) - only left and right mics
            azimuth_deg: Target direction
        """
        n_samples = audio_chunk.shape[0]
        
        # High-pass filter both channels
        filtered = np.zeros_like(audio_chunk)
        filtered[:, 0], self.filter_state_0 = sosfilt(
            self.sos_hp, audio_chunk[:, 0], zi=self.filter_state_0
        )
        filtered[:, 1], self.filter_state_1 = sosfilt(
            self.sos_hp, audio_chunk[:, 1], zi=self.filter_state_1
        )
        
        # Calculate delay
        delay_samples = self.calculate_delay(azimuth_deg)
        
        # Delay-and-sum beamforming
        if delay_samples > 0:
            # Sound from right: delay left channel
            if delay_samples < n_samples:
                beamformed = np.zeros(n_samples)
                beamformed[delay_samples:] = (filtered[:-delay_samples, 0] + 
                                             filtered[delay_samples:, 1]) / 2
                beamformed[:delay_samples] = filtered[:delay_samples, 1]
            else:
                beamformed = filtered[:, 1]
        
        elif delay_samples < 0:
            # Sound from left: delay right channel
            delay_samples = abs(delay_samples)
            if delay_samples < n_samples:
                beamformed = np.zeros(n_samples)
                beamformed[delay_samples:] = (filtered[delay_samples:, 0] + 
                                             filtered[:-delay_samples, 1]) / 2
                beamformed[:delay_samples] = filtered[:delay_samples, 0]
            else:
                beamformed = filtered[:, 0]
        
        else:
            # Center: no delay
            beamformed = np.mean(filtered, axis=1)
        
        # Normalize
        max_val = np.max(np.abs(beamformed))
        if max_val > 0.8:
            beamformed = beamformed * 0.8 / max_val
        
        return beamformed


def read_audio_stream(ser, beamformer):
    """Read from ESP32, extract Ch0-1, beamform"""
    global running, stats, audio_output, current_azimuth
    
    print("Starting audio receiver...")
    
    sync_buffer = bytearray()
    
    def find_header():
        while running:
            byte = ser.read(1)
            if not byte:
                return False
            sync_buffer.append(byte[0])
            if len(sync_buffer) > 3:
                sync_buffer.pop(0)
            if len(sync_buffer) == 3 and bytes(sync_buffer) == b'MIC':
                sync_buffer.clear()
                return True
        return False
    
    while running:
        try:
            if not find_header():
                continue
            
            # Read header
            header = ser.read(3)
            if len(header) != 3:
                stats['errors'] += 1
                continue
            
            n_samples = struct.unpack('<H', header[:2])[0]
            n_channels = header[2]
            
            if n_channels != CHANNELS or n_samples > 2048:
                stats['errors'] += 1
                continue
            
            # Read all 7 channels
            data_size = n_samples * n_channels * 2
            audio_bytes = ser.read(data_size)
            
            if len(audio_bytes) != data_size:
                stats['errors'] += 1
                continue
            
            # Parse all channels
            audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_all = audio_int.reshape(n_samples, n_channels).astype(np.float32) / 32768.0
            
            # EXTRACT ONLY Ch0 and Ch1 (the real mics!)
            audio_2mic = audio_all[:, :2]  # Take only first 2 columns
            
            # Beamform
            output = beamformer.process_chunk(audio_2mic, current_azimuth)
            
            # Output
            output = np.clip(output, -1.0, 1.0)
            output_int16 = (output * 32767).astype(np.int16)
            
            if audio_output:
                audio_output.write(output_int16.tobytes())
            
            stats['packets'] += 1
            
        except Exception as e:
            stats['errors'] += 1
            if stats['errors'] % 100 == 1:
                print(f"\nError: {e}")
            time.sleep(0.01)


def setup_audio():
    """Setup audio output"""
    global audio_output
    
    p = pyaudio.PyAudio()
    
    print("\nAudio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
    
    try:
        audio_output = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )
        print(f"✓ Audio output ready")
    except Exception as e:
        print(f"✗ Audio failed: {e}")
    
    return p


def stats_thread():
    """Stats display"""
    global running, stats
    
    last_packets = 0
    start_time = time.time()
    
    while running:
        time.sleep(2.0)
        
        packets = stats['packets']
        errors = stats['errors']
        elapsed = time.time() - start_time
        pps = (packets - last_packets) / 2.0
        
        print(f"\r[{elapsed:>6.1f}s] Packets: {packets:>6} | "
              f"Rate: {pps:>5.1f}/s | Errors: {errors:>4} | "
              f"Direction: {current_azimuth:>+5.0f}°", 
              end='', flush=True)
        
        last_packets = packets


def control_thread():
    """Control beam direction"""
    global running, current_azimuth
    
    print("\nControls:")
    print("  -90 = Left | 0 = Center | 90 = Right")
    print("  'q' to quit\n")
    
    while running:
        try:
            cmd = input("Direction (-90 to 90): ").strip().lower()
            
            if cmd == 'q':
                running = False
                break
            
            try:
                angle = float(cmd)
                if -90 <= angle <= 90:
                    current_azimuth = angle
                    print(f"✓ Pointing {angle}°")
                else:
                    print("⚠ Must be -90 to 90")
            except ValueError:
                print("⚠ Invalid")
        
        except (KeyboardInterrupt, EOFError):
            running = False
            break


def main():
    global running, audio_output
    
    print("="*70)
    print("2-Microphone Beamforming (D0 Only) - Raspberry Pi")
    print("="*70)
    print(f"Configuration:")
    print(f"  Serial: {SERIAL_PORT} @ {SERIAL_BAUD}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  ESP32 sends: {CHANNELS} channels")
    print(f"  Using: Ch0 and Ch1 only (2 real mics)")
    print(f"  Chunk size: {CHUNK_SIZE}")
    
    # Init beamformer
    beamformer = TwoMicBeamformer(MIC_POSITIONS, SAMPLE_RATE)
    print("✓ Beamformer ready")
    
    # Audio
    p_audio = setup_audio()
    
    # Serial
    print(f"\nConnecting to {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        print("✓ Serial connected")
    except Exception as e:
        print(f"✗ Serial error: {e}")
        sys.exit(1)
    
    # Start threads
    audio_thread = threading.Thread(target=read_audio_stream, 
                                    args=(ser, beamformer), daemon=True)
    stats_thread_obj = threading.Thread(target=stats_thread, daemon=True)
    control_thread_obj = threading.Thread(target=control_thread, daemon=True)
    
    audio_thread.start()
    stats_thread_obj.start()
    control_thread_obj.start()
    
    print("\n" + "="*70)
    print("RUNNING - Press Ctrl+C to stop")
    print("="*70)
    
    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        running = False
    
    time.sleep(1)
    
    if audio_output:
        audio_output.stop_stream()
        audio_output.close()
    p_audio.terminate()
    ser.close()
    
    print("✓ Done")


if __name__ == '__main__':
    main()
    def __init__(self, mic_positions, sample_rate):
        self.mic_positions = mic_positions
        self.n_mics = 4
        self.sample_rate = sample_rate
        
        # High-pass filter to remove DC and noise
        self.sos_hp = butter(2, HIGH_PASS_CUTOFF, 'highpass', 
                            fs=sample_rate, output='sos')
        self.filter_states = [None] * 4
        
        print(f"  4-Mic Configuration:")
        print(f"    Array radius: {MIC_RADIUS*1000:.1f} mm")
        for i in range(4):
            pos = mic_positions[i]
            print(f"    Mic {i}: [{pos[0]*1000:>6.1f}, {pos[1]*1000:>6.1f}] mm")
    
    def calculate_delays(self, azimuth_deg):
        """Calculate delays for each mic relative to first mic"""
        az_rad = np.radians(azimuth_deg)
        
        # Direction of arrival
        direction = np.array([np.cos(az_rad), np.sin(az_rad), 0])
        
        # Time delays for each mic
        delays = np.dot(self.mic_positions, direction) / SPEED_OF_SOUND
        
        # Convert to samples, relative to first mic
        delay_samples = ((delays - delays[0]) * self.sample_rate).astype(int)
        
        return delay_samples
    
    def process_chunk(self, audio_chunk, azimuth_deg):
        """Process with delay-and-sum beamforming"""
        n_samples = audio_chunk.shape[0]
        
        # Apply high-pass filter to each channel
        filtered = np.zeros_like(audio_chunk)
        for ch in range(4):
            filtered[:, ch], self.filter_states[ch] = sosfilt(
                self.sos_hp, audio_chunk[:, ch], 
                zi=self.filter_states[ch]
            )
        
        # Calculate delays
        delays = self.calculate_delays(azimuth_deg)
        
        # Delay-and-sum
        output = np.zeros(n_samples)
        count = np.zeros(n_samples)
        
        for mic in range(4):
            delay = delays[mic]
            
            if delay >= 0:
                # Positive delay: shift forward
                if delay < n_samples:
                    output[delay:] += filtered[:n_samples-delay, mic]
                    count[delay:] += 1
            else:
                # Negative delay: shift backward
                delay = abs(delay)
                if delay < n_samples:
                    output[:n_samples-delay] += filtered[delay:, mic]
                    count[:n_samples-delay] += 1
        
        # Average where we have samples
        mask = count > 0
        output[mask] /= count[mask]
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0.8:
            output = output * 0.8 / max_val
        
        return output


def read_audio_stream(ser, beamformer):
    """Read from ESP32 and beamform"""
    global running, stats, audio_output, current_azimuth
    
    print("Starting audio receiver...")
    
    sync_buffer = bytearray()
    
    def find_header():
        while running:
            byte = ser.read(1)
            if not byte:
                return False
            sync_buffer.append(byte[0])
            if len(sync_buffer) > 3:
                sync_buffer.pop(0)
            if len(sync_buffer) == 3 and bytes(sync_buffer) == b'MIC':
                sync_buffer.clear()
                return True
        return False
    
    while running:
        try:
            if not find_header():
                continue
            
            # Read header: samples(2) + channels(1)
            header = ser.read(3)
            if len(header) != 3:
                stats['errors'] += 1
                continue
            
            n_samples = struct.unpack('<H', header[:2])[0]
            n_channels = header[2]
            
            if n_channels != CHANNELS or n_samples > 2048:
                stats['errors'] += 1
                continue
            
            # Read audio
            data_size = n_samples * n_channels * 2
            audio_bytes = ser.read(data_size)
            
            if len(audio_bytes) != data_size:
                stats['errors'] += 1
                continue
            
            # Parse to numpy
            audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_4ch = audio_int.reshape(n_samples, n_channels).astype(np.float32) / 32768.0
            
            # Beamform
            output = beamformer.process_chunk(audio_4ch, current_azimuth)
            
            # Output
            output = np.clip(output, -1.0, 1.0)
            output_int16 = (output * 32767).astype(np.int16)
            
            if audio_output:
                audio_output.write(output_int16.tobytes())
            
            stats['packets'] += 1
            
        except Exception as e:
            stats['errors'] += 1
            if stats['errors'] % 100 == 1:
                print(f"\nError: {e}")
            time.sleep(0.01)


def setup_audio():
    """Setup audio output"""
    global audio_output
    
    p = pyaudio.PyAudio()
    
    print("\nAudio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
    
    try:
        audio_output = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )
        print(f"✓ Audio output ready")
    except Exception as e:
        print(f"✗ Audio failed: {e}")
    
    return p


def stats_thread():
    """Stats display"""
    global running, stats
    
    last_packets = 0
    start_time = time.time()
    
    while running:
        time.sleep(2.0)
        
        packets = stats['packets']
        errors = stats['errors']
        elapsed = time.time() - start_time
        pps = (packets - last_packets) / 2.0
        
        print(f"\r[{elapsed:>6.1f}s] Packets: {packets:>6} | "
              f"Rate: {pps:>5.1f}/s | Errors: {errors:>4} | "
              f"Direction: {current_azimuth:>+5.0f}°", 
              end='', flush=True)
        
        last_packets = packets


def control_thread():
    """Control beam direction"""
    global running, current_azimuth
    
    print("\nControls:")
    print("  Enter angle: -180 to 180 (0=front, 90=right, 180=back)")
    print("  'q' to quit\n")
    
    while running:
        try:
            cmd = input("Direction: ").strip().lower()
            
            if cmd == 'q':
                running = False
                break
            
            try:
                angle = float(cmd)
                if -180 <= angle <= 180:
                    current_azimuth = angle
                    print(f"✓ Pointing {angle}°")
                else:
                    print("⚠ Must be -180 to 180")
            except ValueError:
                print("⚠ Invalid")
        
        except (KeyboardInterrupt, EOFError):
            running = False
            break


def main():
    global running, audio_output
    
    print("="*70)
    print("4-Microphone Beamforming (D0+D2) - Raspberry Pi")
    print("="*70)
    print(f"Configuration:")
    print(f"  Serial: {SERIAL_PORT} @ {SERIAL_BAUD}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Channels: {CHANNELS}")
    print(f"  Chunk size: {CHUNK_SIZE}")
    
    # Init beamformer
    beamformer = FourMicBeamformer(MIC_POSITIONS, SAMPLE_RATE)
    print("✓ Beamformer ready")
    
    # Audio
    p_audio = setup_audio()
    
    # Serial
    print(f"\nConnecting to {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        print("✓ Serial connected")
    except Exception as e:
        print(f"✗ Serial error: {e}")
        sys.exit(1)
    
    # Start threads
    audio_thread = threading.Thread(target=read_audio_stream, 
                                    args=(ser, beamformer), daemon=True)
    stats_thread_obj = threading.Thread(target=stats_thread, daemon=True)
    control_thread_obj = threading.Thread(target=control_thread, daemon=True)
    
    audio_thread.start()
    stats_thread_obj.start()
    control_thread_obj.start()
    
    print("\n" + "="*70)
    print("RUNNING - Press Ctrl+C to stop")
    print("="*70)
    
    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        running = False
    
    time.sleep(1)
    
    if audio_output:
        audio_output.stop_stream()
        audio_output.close()
    p_audio.terminate()
    ser.close()
    
    print("✓ Done")


if __name__ == '__main__':
    main()
