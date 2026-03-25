#!/usr/bin/env python3
"""
Real-time Audio Beamforming for Sipeed 6+1 Mic Array
via Tang Nano 9K FPGA — optimized for Raspberry Pi 4B

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
from collections import deque
import threading
import queue
import time
import sys
import platform
import os
import subprocess

# ============================================================================
# SERIAL CONFIGURATION
# ============================================================================

SERIAL_BAUD = 3_000_000

def find_serial_port():
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
    """Raw serial reader bypassing pyserial on Linux (BL702 workaround)."""

    def __init__(self, port, baudrate, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._fd = None
        self._ser = None

        if platform.system() == 'Windows':
            import serial
            self._ser = serial.Serial(port, baudrate, timeout=timeout)
        else:
            self._init_linux()

    def _init_linux(self):
        dev_name = os.path.basename(self.port)
        lt = f'/sys/bus/usb-serial/devices/{dev_name}/latency_timer'
        if os.path.exists(lt):
            try:
                subprocess.run(['sudo', 'tee', lt],
                               input=b'1', capture_output=True, timeout=5)
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

    def read_available(self, max_size=65536):
        """Non-blocking read of whatever is available."""
        if self._ser is not None:
            waiting = self._ser.in_waiting
            if waiting > 0:
                return self._ser.read(min(waiting, max_size))
            return b''

        try:
            return os.read(self._fd, max_size)
        except BlockingIOError:
            return b''

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

SAMPLE_RATE = 16226
CHANNELS = 7
CHUNK_SIZE = 512
BITS_PER_SAMPLE = 16
SPEED_OF_SOUND = 343.0

# ============================================================================
# MICROPHONE ARRAY GEOMETRY
# ============================================================================

ARRAY_RADIUS = 0.035

MIC_POSITIONS = np.array([
    [ARRAY_RADIUS * np.cos(np.radians(  0)), ARRAY_RADIUS * np.sin(np.radians(  0)), 0.0],
    [ARRAY_RADIUS * np.cos(np.radians( 60)), ARRAY_RADIUS * np.sin(np.radians( 60)), 0.0],
    [ARRAY_RADIUS * np.cos(np.radians(120)), ARRAY_RADIUS * np.sin(np.radians(120)), 0.0],
    [ARRAY_RADIUS * np.cos(np.radians(180)), ARRAY_RADIUS * np.sin(np.radians(180)), 0.0],
    [ARRAY_RADIUS * np.cos(np.radians(240)), ARRAY_RADIUS * np.sin(np.radians(240)), 0.0],
    [ARRAY_RADIUS * np.cos(np.radians(300)), ARRAY_RADIUS * np.sin(np.radians(300)), 0.0],
    [0.0, 0.0, 0.0],
])

# ============================================================================
# BEAMFORMING PARAMETERS
# ============================================================================

BEAMFORMER_TYPE = 'MVDR'
AZIMUTH_FRONT = 0.0
ELEVATION_FRONT = 0.0

FREQ_MIN = 200
FREQ_MAX = 5000
REGULARIZATION = 1e-3

AGC_TARGET = 0.75
AGC_ATTACK = 0.02
AGC_RELEASE = 0.0005

# Output buffer: holds chunks for smooth playback (absorbs processing jitter)
OUTPUT_QUEUE_SIZE = 8

# ============================================================================
# GLOBAL STATE
# ============================================================================

current_azimuth = AZIMUTH_FRONT
current_elevation = ELEVATION_FRONT
running = True
stats = {'packets': 0, 'errors': 0, 'drops': 0}

# ============================================================================
# VECTORIZED MVDR BEAMFORMER (batch numpy, no Python loops over freq bins)
# ============================================================================

class FastMVDRBeamformer:
    """MVDR via single FFT + batch matrix ops.  No scipy STFT overhead.

    All 257 frequency bins are processed with vectorized numpy (einsum),
    making it ~50x faster than a Python for-loop over bins.
    """

    def __init__(self, mic_positions, sample_rate, n_samples):
        self.mic_positions = mic_positions
        self.n_mics = len(mic_positions)
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.freqs = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
        self.n_freqs = len(self.freqs)

        self.window = np.hanning(n_samples).astype(np.float32)

        self.fmin_idx = max(1, np.argmin(np.abs(self.freqs - FREQ_MIN)))
        self.fmax_idx = np.argmin(np.abs(self.freqs - FREQ_MAX))
        self.n_band = self.fmax_idx - self.fmin_idx

        self.eye_reg = np.stack([np.eye(self.n_mics, dtype=complex) * REGULARIZATION]
                                * self.n_band)

        self._cached_az = None
        self._cached_el = None
        self._steering_full = None  # (n_freqs, n_mics)
        self._steering_band = None  # (n_band, n_mics)
        self._ds_weights = None     # delay-sum fallback for out-of-band

    def _update_steering(self, azimuth_deg, elevation_deg):
        if self._cached_az == azimuth_deg and self._cached_el == elevation_deg:
            return

        az = np.radians(azimuth_deg)
        el = np.radians(elevation_deg)
        doa = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])

        delays = self.mic_positions @ doa / SPEED_OF_SOUND  # (n_mics,)
        omega = 2.0 * np.pi * self.freqs[:, None]           # (n_freqs, 1)
        sv = np.exp(-1j * omega * delays[None, :])           # (n_freqs, n_mics)
        norms = np.linalg.norm(sv, axis=1, keepdims=True)
        sv /= (norms + 1e-12)

        self._steering_full = sv
        self._steering_band = sv[self.fmin_idx:self.fmax_idx]
        self._ds_weights = sv / self.n_mics

        self._cached_az = azimuth_deg
        self._cached_el = elevation_deg

    def process(self, audio_chunk, azimuth_deg, elevation_deg=0):
        self._update_steering(azimuth_deg, elevation_deg)

        windowed = audio_chunk * self.window[:, None]
        X = np.fft.rfft(windowed, axis=0)  # (n_freqs, n_mics)

        output = np.zeros(self.n_freqs, dtype=complex)

        # Out-of-band: simple delay-and-sum (cheap)
        ds = self._ds_weights
        output[:self.fmin_idx] = np.einsum('fi,fi->f',
                                           ds[:self.fmin_idx].conj(),
                                           X[:self.fmin_idx])
        output[self.fmax_idx:] = np.einsum('fi,fi->f',
                                           ds[self.fmax_idx:].conj(),
                                           X[self.fmax_idx:])

        # In-band MVDR: batch covariance + batch solve
        Xb = X[self.fmin_idx:self.fmax_idx]            # (n_band, n_mics)
        ab = self._steering_band                        # (n_band, n_mics)

        # Rank-1 covariance from single snapshot + regularization
        R = np.einsum('fi,fj->fij', Xb, Xb.conj())    # (n_band, M, M)
        R += self.eye_reg

        try:
            R_inv_a = np.linalg.solve(R, ab[:, :, None]).squeeze(-1)
            den = np.einsum('fi,fi->f', ab.conj(), R_inv_a)
            w = R_inv_a / (den[:, None] + 1e-12)
        except np.linalg.LinAlgError:
            w = ab / self.n_mics

        output[self.fmin_idx:self.fmax_idx] = np.einsum('fi,fi->f', w.conj(), Xb)

        result = np.fft.irfft(output, n=self.n_samples)
        return result


# ============================================================================
# SIMPLE AGC (smooth, no pumping)
# ============================================================================

class SmoothAGC:
    def __init__(self, target=AGC_TARGET, attack=AGC_ATTACK, release=AGC_RELEASE):
        self.target = target
        self.attack = attack
        self.release = release
        self.gain = 1.0

    def apply(self, signal):
        peak = np.max(np.abs(signal))
        if peak < 1e-8:
            return signal

        desired_gain = self.target / peak

        if desired_gain < self.gain:
            self.gain += self.attack * (desired_gain - self.gain)
        else:
            self.gain += self.release * (desired_gain - self.gain)

        self.gain = np.clip(self.gain, 0.01, 50.0)
        return signal * self.gain


# ============================================================================
# SERIAL PACKET READER (buffered, efficient)
# ============================================================================

HEADER_MARKER = b'MIC'
PACKET_DATA_SIZE = CHUNK_SIZE * CHANNELS * 2
FULL_PACKET_SIZE = 6 + PACKET_DATA_SIZE   # 'MIC' + 2-byte len + 1-byte nch + data

def serial_reader_thread(ser, audio_queue):
    """Reads serial data in large chunks, extracts MIC packets, pushes audio."""
    global running, stats

    buf = bytearray()
    CHUNK_READ = 8192

    while running:
        try:
            new_data = ser.read(CHUNK_READ)
            if not new_data:
                time.sleep(0.001)
                continue

            buf.extend(new_data)

            while len(buf) >= FULL_PACKET_SIZE:
                idx = buf.find(HEADER_MARKER)
                if idx < 0:
                    buf = buf[-2:]
                    break

                if idx > 0:
                    buf = buf[idx:]

                if len(buf) < 6:
                    break

                n_samples = struct.unpack('<H', buf[3:5])[0]
                n_channels = buf[5]

                if n_channels != CHANNELS or n_samples != CHUNK_SIZE:
                    stats['errors'] += 1
                    buf = buf[3:]
                    continue

                needed = 6 + n_samples * n_channels * 2
                if len(buf) < needed:
                    break

                audio_bytes = bytes(buf[6:needed])
                buf = buf[needed:]

                audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_data = audio_int.reshape(n_samples, n_channels).astype(np.float32) / 32768.0

                if audio_queue.full():
                    try:
                        audio_queue.get_nowait()
                        stats['drops'] += 1
                    except queue.Empty:
                        pass

                audio_queue.put(audio_data)
                stats['packets'] += 1

        except Exception as e:
            print(f"\nSerial read error: {e}")
            stats['errors'] += 1
            time.sleep(0.01)


# ============================================================================
# AUDIO PROCESSING + PLAYBACK THREAD
# ============================================================================

def audio_playback_thread(audio_queue, beamformer, agc, p_audio_stream):
    """Takes chunks from queue, beamforms, applies AGC, writes to speaker."""
    global running, current_azimuth, current_elevation

    silence = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()

    while running:
        try:
            audio_data = audio_queue.get(timeout=0.05)
        except queue.Empty:
            p_audio_stream.write(silence)
            continue

        try:
            output = beamformer.process(audio_data, current_azimuth, current_elevation)
            output = agc.apply(output)
            output = np.clip(output, -1.0, 1.0)
            output_int16 = (output * 32767).astype(np.int16)
            p_audio_stream.write(output_int16.tobytes())
        except Exception as e:
            print(f"\nProcessing error: {e}")
            p_audio_stream.write(silence)


# ============================================================================
# STATS
# ============================================================================

def stats_thread_fn():
    global running, stats
    last_packets = 0

    while running:
        time.sleep(2.0)
        pkts = stats['packets']
        pps = (pkts - last_packets) / 2.0
        print(f"\r[STATS] pkts={pkts}  rate={pps:.1f}/s  "
              f"err={stats['errors']}  drop={stats['drops']}  "
              f"az={current_azimuth:.0f}°",
              end='', flush=True)
        last_packets = pkts


# ============================================================================
# MAIN
# ============================================================================

def main():
    global running

    print("=" * 60)
    print(" Beamforming — Sipeed 6+1 Array + Tang Nano 9K")
    print("=" * 60)
    print(f"  Port:       {SERIAL_PORT} @ {SERIAL_BAUD/1e6:.0f} Mbaud")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Channels:    {CHANNELS}")
    print(f"  Beamformer:  {BEAMFORMER_TYPE}")
    print(f"  Direction:   az={AZIMUTH_FRONT}° el={ELEVATION_FRONT}°")
    print(f"  Array gain:  ~{10*np.log10(CHANNELS):.1f} dB")

    beamformer = FastMVDRBeamformer(MIC_POSITIONS, SAMPLE_RATE, CHUNK_SIZE)
    agc = SmoothAGC()
    print(f"MVDR beamformer ready ({beamformer.n_band} active bins, "
          f"{FREQ_MIN}-{FREQ_MAX} Hz)")

    p = pyaudio.PyAudio()
    print("\nAudio output devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            print(f"  [{i}] {info['name']}")

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=CHUNK_SIZE * 2
    )
    print(f"Audio output: {SAMPLE_RATE} Hz mono")

    print(f"\nConnecting to {SERIAL_PORT}...")
    try:
        ser = RawSerialPort(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(0.5)
        ser.reset_input_buffer()
        print(f"Connected ({'pyserial' if ser._ser else 'raw I/O'})")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    audio_q = queue.Queue(maxsize=OUTPUT_QUEUE_SIZE)

    t_serial = threading.Thread(target=serial_reader_thread,
                                args=(ser, audio_q), daemon=True)
    t_playback = threading.Thread(target=audio_playback_thread,
                                  args=(audio_q, beamformer, agc, stream), daemon=True)
    t_stats = threading.Thread(target=stats_thread_fn, daemon=True)

    t_serial.start()
    t_playback.start()
    t_stats.start()

    print("\n" + "=" * 60)
    print(" RUNNING — press Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        running = False
        time.sleep(0.5)
        stream.stop_stream()
        stream.close()
        p.terminate()
        ser.close()
        print("Done.")


if __name__ == '__main__':
    main()
