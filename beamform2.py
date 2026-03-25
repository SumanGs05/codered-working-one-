#!/usr/bin/env python3
"""
Real-time Frequency-Domain Audio Beamforming for Sipeed 6+1 Mic Array
STFT-based MVDR/GEV beamforming with 7 real microphone channels
via Tang Nano 9K FPGA — full circular array configuration

Channel mapping (from FPGA i2s_receiver):
  ch0 = D0 left  → mic 0 (outer, 0°)
  ch1 = D0 right → mic 1 (outer, 60°)
  ch2 = D1 left  → mic 2 (outer, 120°)
  ch3 = D1 right → mic 3 (outer, 180°)
  ch4 = D2 left  → mic 4 (outer, 240°)
  ch5 = D2 right → mic 5 (outer, 300°)
  ch6 = D3 left  → mic 6 (center)
"""

import struct
import numpy as np
import pyaudio
from scipy.linalg import eigh
from scipy.signal import stft, istft
from collections import deque
import threading
import time
import sys
import platform
import os
import subprocess

# ============================================================================
# SERIAL CONFIGURATION  (Tang Nano 9K via BL702 USB-UART)
# ============================================================================

SERIAL_BAUD = 3_000_000

def find_serial_port():
    """Auto-detect Tang Nano 9K COM port"""
    if platform.system() == 'Windows':
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        for p in ports:
            desc = (p.description or '').lower()
            if any(k in desc for k in ['bl702', 'tang', 'gowin', 'usb serial', 'usb-serial']):
                return p.device
        for p in ports:
            if 'COM' in p.device:
                return p.device
        return 'COM3'
    else:
        for dev in ['/dev/ttyUSB1', '/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyAMA0']:
            if os.path.exists(dev):
                return dev
        return '/dev/ttyUSB1'


class RawSerialPort:
    """Raw serial reader that bypasses pyserial on Linux.

    The BL702 chip on the Tang Nano 9K crashes when pyserial sends
    DTR/RTS modem control signals during open().  This class uses
    stty + os.open(O_NOCTTY) to avoid that entirely.
    On Windows, falls back to pyserial which works fine.
    """

    def __init__(self, port, baudrate, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._fd = None
        self._ser = None
        self._buf = b''

        if platform.system() == 'Windows':
            import serial
            self._ser = serial.Serial(port, baudrate, timeout=timeout)
        else:
            self._init_linux()

    def _init_linux(self):
        latency_path = None
        dev_name = os.path.basename(self.port)
        lt = f'/sys/bus/usb-serial/devices/{dev_name}/latency_timer'
        if os.path.exists(lt):
            try:
                subprocess.run(['sudo', 'tee', lt],
                               input=b'1', capture_output=True, timeout=5)
                latency_path = lt
            except Exception:
                pass

        subprocess.run([
            'stty', '-F', self.port,
            str(self.baudrate), 'raw', '-echo', '-crtscts', '-clocal'
        ], check=True, timeout=5)

        self._fd = os.open(self.port, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)

        import select
        self._poll = select.poll()
        self._poll.register(self._fd, select.POLLIN)

    def read(self, size):
        if self._ser is not None:
            return self._ser.read(size)

        data = bytearray()
        deadline = time.monotonic() + self.timeout
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait_ms = max(1, int(remaining * 1000))
            events = self._poll.poll(wait_ms)
            if events:
                try:
                    chunk = os.read(self._fd, size - len(data))
                    if chunk:
                        data.extend(chunk)
                except BlockingIOError:
                    pass
        return bytes(data)

    def reset_input_buffer(self):
        if self._ser is not None:
            self._ser.reset_input_buffer()
            return
        try:
            import termios
            termios.tcflush(self._fd, termios.TCIFLUSH)
        except Exception:
            while True:
                events = self._poll.poll(0)
                if not events:
                    break
                try:
                    os.read(self._fd, 65536)
                except BlockingIOError:
                    break

    def close(self):
        if self._ser is not None:
            self._ser.close()
        elif self._fd is not None:
            os.close(self._fd)
            self._fd = None


SERIAL_PORT = find_serial_port()

# ============================================================================
# AUDIO / FPGA PARAMETERS
# ============================================================================

# Tang Nano 9K: BCLK = 27 MHz / (2*13) = 1038461 Hz, WS = BCLK / 64
SAMPLE_RATE = 16226       # actual Fs from FPGA (27e6 / 26 / 64)
CHANNELS = 7              # all 7 real mics from FPGA
CHUNK_SIZE = 512           # frames per packet (matches FPGA packetizer)
BITS_PER_SAMPLE = 16

# STFT parameters — larger FFT for better frequency resolution with 7 mics
NFFT = 1024
OVERLAP = NFFT * 3 // 4   # 75% overlap for better time resolution
HOP_LENGTH = NFFT - OVERLAP

# ============================================================================
# SIPEED 6+1 CIRCULAR MICROPHONE ARRAY GEOMETRY
# ============================================================================

# The Sipeed array has 6 outer mics in a circle + 1 center mic.
# Outer mic radius ≈ 35 mm from center — MEASURE YOUR ARRAY and adjust.
ARRAY_RADIUS = 0.035  # meters

# 6 outer mics at 60° intervals (counter-clockwise from +X axis)
# Center mic at origin
MIC_POSITIONS = np.array([
    [ARRAY_RADIUS * np.cos(np.radians(  0)), ARRAY_RADIUS * np.sin(np.radians(  0)), 0.0],  # ch0: mic 0 @ 0°
    [ARRAY_RADIUS * np.cos(np.radians( 60)), ARRAY_RADIUS * np.sin(np.radians( 60)), 0.0],  # ch1: mic 1 @ 60°
    [ARRAY_RADIUS * np.cos(np.radians(120)), ARRAY_RADIUS * np.sin(np.radians(120)), 0.0],  # ch2: mic 2 @ 120°
    [ARRAY_RADIUS * np.cos(np.radians(180)), ARRAY_RADIUS * np.sin(np.radians(180)), 0.0],  # ch3: mic 3 @ 180°
    [ARRAY_RADIUS * np.cos(np.radians(240)), ARRAY_RADIUS * np.sin(np.radians(240)), 0.0],  # ch4: mic 4 @ 240°
    [ARRAY_RADIUS * np.cos(np.radians(300)), ARRAY_RADIUS * np.sin(np.radians(300)), 0.0],  # ch5: mic 5 @ 300°
    [0.0, 0.0, 0.0],                                                                          # ch6: center mic
])

# ============================================================================
# BEAMFORMING PARAMETERS
# ============================================================================

BEAMFORMER_TYPE = 'MVDR'   # 'MVDR' or 'GEV' or 'DelaySum'
SPEED_OF_SOUND = 343.0     # m/s
NOISE_UPDATE_RATE = 0.15   # seconds between noise covariance updates
REGULARIZATION = 1e-4      # diagonal loading (lower = more aggressive nulling)

# Frequency range for adaptive beamforming (speech band)
FREQ_MIN = 200    # Hz — circular array resolves lower freqs than 2-mic
FREQ_MAX = 5000   # Hz — extend upper range for better clarity

# Target direction
AZIMUTH_FRONT = 0.0    # degrees (0° = +X axis)
ELEVATION_FRONT = 0.0  # degrees (0° = horizontal)

# Zoom angle range for future audio zoom control
ZOOM_MIN_ANGLE = 8
ZOOM_MAX_ANGLE = 60

# ============================================================================
# GLOBAL STATE
# ============================================================================

current_zoom_angle = 60.0
current_azimuth = AZIMUTH_FRONT
current_elevation = ELEVATION_FRONT
audio_output = None
running = True
stats = {'packets': 0, 'errors': 0, 'drops': 0, 'last_time': time.time()}

# ============================================================================
# FREQUENCY-DOMAIN BEAMFORMING (optimized for 7-channel circular array)
# ============================================================================

class FrequencyDomainBeamformer:
    def __init__(self, mic_positions, sample_rate):
        self.mic_positions = mic_positions
        self.n_mics = len(mic_positions)
        self.sample_rate = sample_rate
        self.nfft = NFFT
        self.hop_length = HOP_LENGTH

        self.freqs = np.fft.rfftfreq(self.nfft, 1.0 / sample_rate)
        self.n_freqs = len(self.freqs)

        self.freq_min_idx = np.argmin(np.abs(self.freqs - FREQ_MIN))
        self.freq_max_idx = np.argmin(np.abs(self.freqs - FREQ_MAX))

        self.noise_cov = np.zeros((self.n_freqs, self.n_mics, self.n_mics), dtype=complex)
        for f in range(self.n_freqs):
            self.noise_cov[f] = np.eye(self.n_mics) * REGULARIZATION

        self.noise_buffer = deque(maxlen=80)
        self.noise_update_counter = 0

        self.window = np.hanning(self.nfft)

        # Pre-compute steering vectors for the target direction (updated when direction changes)
        self._cached_az = None
        self._cached_el = None
        self._cached_steering = None

        print(f"  Mics: {self.n_mics} (6 circular + 1 center)")
        print(f"  Frequency bins: {self.n_freqs}")
        print(f"  Beamforming range: {FREQ_MIN}-{FREQ_MAX} Hz "
              f"(bins {self.freq_min_idx}-{self.freq_max_idx})")

    def steering_vector(self, azimuth_deg, freq_hz, elevation_deg=0):
        az_rad = np.radians(azimuth_deg)
        el_rad = np.radians(elevation_deg)

        doa = np.array([
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad)
        ])

        delays = self.mic_positions @ doa / SPEED_OF_SOUND
        omega = 2.0 * np.pi * freq_hz
        steering = np.exp(-1j * omega * delays)
        return steering / np.linalg.norm(steering)

    def get_all_steering(self, azimuth_deg, elevation_deg=0):
        """Pre-compute steering vectors for all frequency bins (cached)"""
        if self._cached_az == azimuth_deg and self._cached_el == elevation_deg and self._cached_steering is not None:
            return self._cached_steering

        sv = np.zeros((self.n_freqs, self.n_mics), dtype=complex)
        for f in range(self.n_freqs):
            sv[f] = self.steering_vector(azimuth_deg, self.freqs[f], elevation_deg)

        self._cached_az = azimuth_deg
        self._cached_el = elevation_deg
        self._cached_steering = sv
        return sv

    def update_noise_covariance(self, stft_data):
        self.noise_buffer.append(stft_data)

        if len(self.noise_buffer) >= 15:
            all_frames = np.concatenate(list(self.noise_buffer), axis=2)
            for f in range(self.n_freqs):
                X = all_frames[f, :, :]
                cov = (X @ X.conj().T) / X.shape[1]
                cov += np.eye(self.n_mics) * REGULARIZATION
                self.noise_cov[f] = cov


class MVDRFrequencyBeamformer(FrequencyDomainBeamformer):
    """MVDR Beamformer — 7-mic circular array

    With 7 mics the MVDR can place up to 6 spatial nulls on interferers,
    giving far superior noise rejection compared to the old 2-mic setup.
    """

    def process_chunk(self, audio_chunk, azimuth_deg, elevation_deg=0):
        n_samples = audio_chunk.shape[0]

        stft_data = []
        for mic in range(self.n_mics):
            _, _, Zxx = stft(audio_chunk[:, mic],
                             fs=self.sample_rate,
                             window=self.window,
                             nperseg=self.nfft,
                             noverlap=OVERLAP)
            stft_data.append(Zxx)

        stft_data = np.array(stft_data).transpose(1, 0, 2)
        n_frames = stft_data.shape[2]

        self.noise_update_counter += 1
        noise_interval = max(1, int(NOISE_UPDATE_RATE * self.sample_rate / CHUNK_SIZE))
        if self.noise_update_counter % noise_interval == 0:
            self.update_noise_covariance(stft_data)

        all_steering = self.get_all_steering(azimuth_deg, elevation_deg)
        output_stft = np.zeros((self.n_freqs, n_frames), dtype=complex)

        eye_reg = np.eye(self.n_mics) * REGULARIZATION

        for f in range(self.n_freqs):
            freq_hz = self.freqs[f]
            a = all_steering[f]

            if freq_hz < FREQ_MIN or freq_hz > FREQ_MAX or freq_hz < 1.0:
                w = a / self.n_mics
            else:
                X = stft_data[f, :, :]
                R = (X @ X.conj().T) / n_frames + eye_reg
                try:
                    R_inv = np.linalg.inv(R)
                    num = R_inv @ a
                    den = a.conj() @ num
                    w = num / (den + 1e-12)
                except np.linalg.LinAlgError:
                    w = a / self.n_mics

            output_stft[f, :] = w.conj() @ stft_data[f, :, :]

        _, output_time = istft(output_stft,
                               fs=self.sample_rate,
                               window=self.window,
                               nperseg=self.nfft,
                               noverlap=OVERLAP)

        output_time = _match_length(output_time, n_samples)
        return _normalize(output_time)


class GEVFrequencyBeamformer(FrequencyDomainBeamformer):
    """GEV Beamformer — maximizes SNR via generalized eigenvalue decomposition.

    Especially effective with 7 mics: the eigenvector corresponding to the
    largest eigenvalue captures the dominant source direction with high fidelity.
    """

    def process_chunk(self, audio_chunk, azimuth_deg, elevation_deg=0):
        n_samples = audio_chunk.shape[0]

        stft_data = []
        for mic in range(self.n_mics):
            _, _, Zxx = stft(audio_chunk[:, mic],
                             fs=self.sample_rate,
                             window=self.window,
                             nperseg=self.nfft,
                             noverlap=OVERLAP)
            stft_data.append(Zxx)

        stft_data = np.array(stft_data).transpose(1, 0, 2)
        n_frames = stft_data.shape[2]

        self.noise_update_counter += 1
        noise_interval = max(1, int(NOISE_UPDATE_RATE * self.sample_rate / CHUNK_SIZE))
        if self.noise_update_counter % noise_interval == 0:
            self.update_noise_covariance(stft_data)

        all_steering = self.get_all_steering(azimuth_deg, elevation_deg)
        output_stft = np.zeros((self.n_freqs, n_frames), dtype=complex)
        eye_reg = np.eye(self.n_mics) * REGULARIZATION

        for f in range(self.n_freqs):
            freq_hz = self.freqs[f]
            a = all_steering[f]

            if freq_hz < FREQ_MIN or freq_hz > FREQ_MAX or freq_hz < 1.0:
                w = a / self.n_mics
            else:
                X = stft_data[f, :, :]
                R_signal = (X @ X.conj().T) / n_frames + eye_reg
                R_noise = self.noise_cov[f]

                try:
                    eigenvalues, eigenvectors = eigh(R_signal, R_noise)
                    w = eigenvectors[:, -1]
                    phase_align = a.conj() @ w
                    if abs(phase_align) > 1e-10:
                        w = w * (phase_align / abs(phase_align))
                except (np.linalg.LinAlgError, ValueError):
                    w = a / self.n_mics

            output_stft[f, :] = w.conj() @ stft_data[f, :, :]

        _, output_time = istft(output_stft,
                               fs=self.sample_rate,
                               window=self.window,
                               nperseg=self.nfft,
                               noverlap=OVERLAP)

        output_time = _match_length(output_time, n_samples)
        return _normalize(output_time)


class DelayAndSumFrequencyBeamformer(FrequencyDomainBeamformer):
    """Delay-and-Sum — simple but robust baseline with 7 mics.

    With the circular array this already gives ~8.5 dB array gain
    (10*log10(7) = 8.45 dB) compared to ~3 dB from 2 mics.
    """

    def process_chunk(self, audio_chunk, azimuth_deg, elevation_deg=0):
        n_samples = audio_chunk.shape[0]

        stft_data = []
        for mic in range(self.n_mics):
            _, _, Zxx = stft(audio_chunk[:, mic],
                             fs=self.sample_rate,
                             window=self.window,
                             nperseg=self.nfft,
                             noverlap=OVERLAP)
            stft_data.append(Zxx)

        stft_data = np.array(stft_data).transpose(1, 0, 2)
        n_frames = stft_data.shape[2]

        all_steering = self.get_all_steering(azimuth_deg, elevation_deg)
        output_stft = np.zeros((self.n_freqs, n_frames), dtype=complex)

        for f in range(self.n_freqs):
            w = all_steering[f] / self.n_mics
            output_stft[f, :] = w.conj() @ stft_data[f, :, :]

        _, output_time = istft(output_stft,
                               fs=self.sample_rate,
                               window=self.window,
                               nperseg=self.nfft,
                               noverlap=OVERLAP)

        output_time = _match_length(output_time, n_samples)
        return _normalize(output_time)


# ============================================================================
# HELPERS
# ============================================================================

def _match_length(signal, target_len):
    if len(signal) < target_len:
        return np.pad(signal, (0, target_len - len(signal)))
    return signal[:target_len]


def _normalize(signal, ceiling=0.85):
    peak = np.max(np.abs(signal))
    if peak > ceiling:
        signal = signal * ceiling / peak
    return signal


# ============================================================================
# SERIAL RECEIVER
# ============================================================================

def read_audio_stream(ser, beamformer):
    """Read MIC-header packets from Tang Nano 9K and apply beamforming"""
    global running, stats, audio_output, current_azimuth, current_elevation

    print("Starting audio receiver...")

    sync_buf = bytearray()

    def find_mic_header():
        while running:
            byte = ser.read(1)
            if not byte:
                return False
            sync_buf.append(byte[0])
            if len(sync_buf) > 3:
                sync_buf.pop(0)
            if len(sync_buf) == 3 and bytes(sync_buf) == b'MIC':
                sync_buf.clear()
                return True
        return False

    while running:
        try:
            if not find_mic_header():
                continue

            header_rest = ser.read(3)
            if len(header_rest) != 3:
                stats['errors'] += 1
                continue

            n_samples = struct.unpack('<H', header_rest[:2])[0]
            n_channels = header_rest[2]

            if n_channels != CHANNELS or n_samples != CHUNK_SIZE or n_samples > 2048:
                stats['errors'] += 1
                continue

            data_size = n_samples * n_channels * 2
            audio_bytes = ser.read(data_size)

            if len(audio_bytes) != data_size:
                stats['drops'] += 1
                continue

            audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_data = audio_int.reshape(n_samples, n_channels).astype(np.float32) / 32768.0

            output_audio = beamformer.process_chunk(
                audio_data, current_azimuth, current_elevation)

            output_audio = np.clip(output_audio, -1.0, 1.0)
            output_int16 = (output_audio * 32767).astype(np.int16)

            if audio_output:
                audio_output.write(output_int16.tobytes())

            stats['packets'] += 1

        except Exception as e:
            print(f"\nError in audio stream: {e}")
            import traceback
            traceback.print_exc()
            stats['errors'] += 1
            time.sleep(0.01)


# ============================================================================
# DIRECTION CONTROL
# ============================================================================

def direction_control_thread():
    """Update beamforming direction.

    Currently fixed forward.  With the circular array, you can steer
    to any azimuth 0-360° by changing current_azimuth, e.g. from a
    DOA estimator or keyboard input.
    """
    global current_azimuth, current_elevation, running

    print("Direction control: fixed front-facing (modify for dynamic steering)")

    while running:
        time.sleep(0.1)


# ============================================================================
# AUDIO OUTPUT
# ============================================================================

def setup_audio_output():
    global audio_output

    p = pyaudio.PyAudio()

    print("\nAvailable audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            print(f"  [{i}] {info['name']} (out: {info['maxOutputChannels']})")

    audio_output = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=CHUNK_SIZE
    )

    print(f"\nAudio output: {SAMPLE_RATE} Hz, mono")
    return p


# ============================================================================
# STATISTICS
# ============================================================================

def stats_thread_fn():
    global running, stats, current_azimuth, current_elevation

    last_packets = 0

    while running:
        time.sleep(2.0)
        packets = stats['packets']
        errors = stats['errors']
        drops = stats['drops']
        pps = (packets - last_packets) / 2.0

        print(f"\r[STATS] pkts={packets}  rate={pps:.1f}/s  "
              f"err={errors}  drop={drops}  "
              f"az={current_azimuth:.0f}  "
              f"lat~{NFFT/SAMPLE_RATE*1000:.0f}ms",
              end='', flush=True)

        last_packets = packets


# ============================================================================
# MAIN
# ============================================================================

def main():
    global running, audio_output

    print("=" * 70)
    print(" Audio Beamforming — Sipeed 6+1 Circular Array via Tang Nano 9K")
    print("=" * 70)
    print(f"\n  Serial:      {SERIAL_PORT} @ {SERIAL_BAUD/1e6:.0f} Mbaud")
    print(f"  Sample rate: {SAMPLE_RATE} Hz  (from FPGA clock divider)")
    print(f"  Channels:    {CHANNELS} (6 circular + 1 center)")
    print(f"  Array radius:{ARRAY_RADIUS*1000:.1f} mm")
    print(f"  FFT:         {NFFT} pts, hop {HOP_LENGTH}, overlap {OVERLAP}")
    print(f"  Beamformer:  {BEAMFORMER_TYPE}")
    print(f"  Speech band: {FREQ_MIN}-{FREQ_MAX} Hz")
    print(f"  Direction:   az={AZIMUTH_FRONT}° el={ELEVATION_FRONT}°")
    print(f"\n  Array gain:  ~{10*np.log10(CHANNELS):.1f} dB (delay-sum)")
    print(f"  Null budget: {CHANNELS - 1} spatial nulls (MVDR)")

    if BEAMFORMER_TYPE == 'MVDR':
        beamformer = MVDRFrequencyBeamformer(MIC_POSITIONS, SAMPLE_RATE)
    elif BEAMFORMER_TYPE == 'GEV':
        beamformer = GEVFrequencyBeamformer(MIC_POSITIONS, SAMPLE_RATE)
    else:
        beamformer = DelayAndSumFrequencyBeamformer(MIC_POSITIONS, SAMPLE_RATE)

    print(f"\nBeamformer initialized")

    p_audio = setup_audio_output()

    print(f"\nConnecting to Tang Nano 9K on {SERIAL_PORT}...")
    try:
        ser = RawSerialPort(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(0.5)
        ser.reset_input_buffer()
        print(f"Serial connected ({'pyserial' if ser._ser else 'raw I/O'})")
    except Exception as e:
        print(f"ERROR: Could not open serial port: {e}")
        print("Check that the Tang Nano 9K is plugged in and the port is correct.")
        sys.exit(1)

    audio_thread = threading.Thread(
        target=read_audio_stream, args=(ser, beamformer), daemon=True)
    direction_thread = threading.Thread(
        target=direction_control_thread, daemon=True)
    stats_t = threading.Thread(
        target=stats_thread_fn, daemon=True)

    audio_thread.start()
    direction_thread.start()
    stats_t.start()

    print("\n" + "=" * 70)
    print(" RUNNING — 7-channel beamforming active")
    print(" Press Ctrl+C to stop")
    print("=" * 70 + "\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        running = False
        time.sleep(1)

        if audio_output:
            audio_output.stop_stream()
            audio_output.close()
        p_audio.terminate()
        ser.close()

        print("Done.")

if __name__ == '__main__':
    main()
