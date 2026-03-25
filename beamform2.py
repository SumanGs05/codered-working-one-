#!/usr/bin/env python3
"""
Real-time MVDR Beamforming — Sipeed 6+1 Mic Array + Tang Nano 9K
Designed for Raspberry Pi 4B real-time operation.

Architecture:
  [serial_reader] --Queue--> [processor] --Queue--> [audio_writer]

Key design choices:
  - Noise calibration phase (first few seconds) builds a clean noise
    covariance matrix.  MVDR then uses this FIXED noise estimate so it
    never self-cancels the target signal.
  - Overlap-add synthesis with Hanning window eliminates chunk-boundary
    clicks.
  - Vectorized batch numpy (einsum + batched solve) — zero Python loops
    over frequency bins.
  - Short serial timeout (100ms) so USB hiccups don't cause 1-second gaps.

Channel mapping (from FPGA i2s_receiver):
  ch0 = D0 left  -> mic 0 (outer, 0 deg)
  ch1 = D0 right -> mic 1 (outer, 60 deg)
  ch2 = D1 left  -> mic 2 (outer, 120 deg)
  ch3 = D1 right -> mic 3 (outer, 180 deg)
  ch4 = D2 left  -> mic 4 (outer, 240 deg)
  ch5 = D2 right -> mic 5 (outer, 300 deg)
  ch6 = D3 left  -> mic 6 (center)
"""

import struct
import numpy as np
import pyaudio
import threading
import queue
import time
import sys
import platform
import os
import subprocess

# ============================================================================
# SERIAL
# ============================================================================

SERIAL_BAUD = 3_000_000

def find_serial_port():
    if platform.system() == 'Windows':
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            desc = (p.description or '').lower()
            if any(k in desc for k in ['bl702', 'tang', 'gowin', 'usb serial']):
                return p.device
        for p in serial.tools.list_ports.comports():
            if 'COM' in p.device:
                return p.device
        return 'COM3'
    for dev in ['/dev/ttyUSB1', '/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyAMA0']:
        if os.path.exists(dev):
            return dev
    return '/dev/ttyUSB1'


class RawSerialPort:
    """Bypasses pyserial on Linux (BL702 crashes on DTR/RTS toggle)."""

    def __init__(self, port, baudrate, timeout=0.1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._fd = None
        self._ser = None
        self._poll = None

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
            events = self._poll.poll(max(1, int(remaining * 1000)))
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
# PARAMETERS
# ============================================================================

SAMPLE_RATE = 16226
CHANNELS = 7
CHUNK_SIZE = 512
SPEED_OF_SOUND = 343.0
ARRAY_RADIUS = 0.035

MIC_POSITIONS = np.array([
    [ARRAY_RADIUS * np.cos(np.radians(a)), ARRAY_RADIUS * np.sin(np.radians(a)), 0.0]
    for a in [0, 60, 120, 180, 240, 300]
] + [[0.0, 0.0, 0.0]])

# MVDR
FREQ_MIN = 200
FREQ_MAX = 5000
REGULARIZATION = 1e-3
NOISE_CAL_SECONDS = 3       # seconds of silence to calibrate noise

# Overlap-add: process 2*CHUNK_SIZE with 50% overlap for click-free output
OLA_NFFT = CHUNK_SIZE * 2   # 1024 samples
OLA_HOP = CHUNK_SIZE         # 512 samples (50% overlap)

# AGC
AGC_TARGET = 0.80
AGC_ATTACK = 0.15
AGC_RELEASE = 0.008

# Pipeline
INPUT_QUEUE_SIZE = 16
OUTPUT_QUEUE_SIZE = 16

# Direction
AZIMUTH = 0.0
ELEVATION = 0.0

# ============================================================================
# GLOBAL STATE
# ============================================================================

running = True
stats = {'packets': 0, 'errors': 0, 'drops': 0, 'phase': 'init'}

# ============================================================================
# MVDR BEAMFORMER WITH NOISE CALIBRATION + OVERLAP-ADD
# ============================================================================

class MVDRBeamformer:
    def __init__(self, mic_positions, sample_rate):
        self.n_mics = len(mic_positions)
        self.mic_positions = mic_positions
        self.sample_rate = sample_rate

        self.nfft = OLA_NFFT
        self.hop = OLA_HOP
        self.freqs = np.fft.rfftfreq(self.nfft, 1.0 / sample_rate)
        self.n_freqs = len(self.freqs)

        self.fmin = max(1, int(np.argmin(np.abs(self.freqs - FREQ_MIN))))
        self.fmax = int(np.argmin(np.abs(self.freqs - FREQ_MAX)))
        self.n_band = self.fmax - self.fmin

        # Synthesis window (sqrt-Hann for perfect reconstruction with 50% overlap)
        self.win_analysis = np.sqrt(np.hanning(self.nfft)).astype(np.float32)
        self.win_synthesis = self.win_analysis.copy()

        # Noise covariance (calibrated, then frozen)
        self.R_noise = None
        self.noise_frames = []
        self.calibrated = False

        # Previous chunk for overlap-add
        self.prev_chunk = np.zeros(self.hop, dtype=np.float32)

        # Steering vectors (cached)
        self._cached_az = None
        self._cached_el = None
        self._sv_band = None
        self._ds_weights = None

        # Pre-allocate regularization
        self.eye_reg = np.stack(
            [np.eye(self.n_mics, dtype=complex) * REGULARIZATION] * self.n_band
        )

    def _update_steering(self, az_deg, el_deg):
        if self._cached_az == az_deg and self._cached_el == el_deg:
            return
        az = np.radians(az_deg)
        el = np.radians(el_deg)
        doa = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])

        delays = self.mic_positions @ doa / SPEED_OF_SOUND
        omega = 2.0 * np.pi * self.freqs[:, None]
        sv = np.exp(-1j * omega * delays[None, :])
        norms = np.linalg.norm(sv, axis=1, keepdims=True)
        sv /= (norms + 1e-12)

        self._sv_band = sv[self.fmin:self.fmax]
        self._ds_weights = sv / self.n_mics
        self._cached_az = az_deg
        self._cached_el = el_deg

    def feed_noise(self, chunk_7ch):
        """Accumulate noise frames during calibration phase."""
        self.noise_frames.append(chunk_7ch.copy())

    def finish_calibration(self):
        """Build noise covariance from accumulated frames."""
        if not self.noise_frames:
            self.R_noise = self.eye_reg.copy()
            self.calibrated = True
            return

        all_noise = np.concatenate(self.noise_frames, axis=0)
        n_full = (len(all_noise) // self.nfft) * self.nfft
        if n_full < self.nfft:
            self.R_noise = self.eye_reg.copy()
            self.calibrated = True
            return

        all_noise = all_noise[:n_full]
        n_segments = n_full // self.hop - 1

        R_acc = np.zeros((self.n_band, self.n_mics, self.n_mics), dtype=complex)
        count = 0

        for i in range(n_segments):
            seg = all_noise[i * self.hop: i * self.hop + self.nfft]
            windowed = seg * self.win_analysis[:, None]
            X = np.fft.rfft(windowed, axis=0)
            Xb = X[self.fmin:self.fmax]
            R_acc += np.einsum('fi,fj->fij', Xb, Xb.conj())
            count += 1

        if count > 0:
            R_acc /= count

        self.R_noise = R_acc + self.eye_reg
        self.noise_frames = []
        self.calibrated = True
        print(f"\n  Noise calibration done ({count} segments, "
              f"{len(all_noise)/self.sample_rate:.1f}s)")

    def process(self, prev_chunk, curr_chunk, az_deg, el_deg=0):
        """Process two consecutive chunks with overlap-add.

        prev_chunk: (CHUNK_SIZE, 7) — previous 512 samples
        curr_chunk: (CHUNK_SIZE, 7) — current 512 samples
        Returns: (CHUNK_SIZE,) — output audio for current hop
        """
        self._update_steering(az_deg, el_deg)

        # Concatenate for OLA_NFFT = 1024 samples
        frame = np.concatenate([prev_chunk, curr_chunk], axis=0)  # (1024, 7)

        # Windowed FFT
        windowed = frame * self.win_analysis[:, None]
        X = np.fft.rfft(windowed, axis=0)  # (n_freqs, 7)

        output_spectrum = np.zeros(self.n_freqs, dtype=complex)

        # Out-of-band: delay-and-sum
        ds = self._ds_weights
        output_spectrum[:self.fmin] = np.einsum(
            'fi,fi->f', ds[:self.fmin].conj(), X[:self.fmin])
        output_spectrum[self.fmax:] = np.einsum(
            'fi,fi->f', ds[self.fmax:].conj(), X[self.fmax:])

        # In-band: MVDR with calibrated noise covariance
        Xb = X[self.fmin:self.fmax]
        ab = self._sv_band

        if self.calibrated and self.R_noise is not None:
            R = self.R_noise
        else:
            R = np.einsum('fi,fj->fij', Xb, Xb.conj()) + self.eye_reg

        try:
            R_inv_a = np.linalg.solve(R, ab[:, :, None]).squeeze(-1)
            den = np.einsum('fi,fi->f', ab.conj(), R_inv_a)
            w = R_inv_a / (den[:, None] + 1e-12)
        except np.linalg.LinAlgError:
            w = ab / self.n_mics

        output_spectrum[self.fmin:self.fmax] = np.einsum('fi,fi->f', w.conj(), Xb)

        # IFFT + synthesis window
        time_out = np.fft.irfft(output_spectrum, n=self.nfft)
        time_out *= self.win_synthesis

        # Overlap-add: second half of previous + first half of current
        result = self.prev_chunk + time_out[:self.hop]
        self.prev_chunk = time_out[self.hop:].astype(np.float32)

        return result.astype(np.float32)


# ============================================================================
# SMOOTH AGC
# ============================================================================

class SmoothAGC:
    def __init__(self):
        self.gain = 1.0

    def apply(self, signal):
        peak = np.max(np.abs(signal))
        if peak < 1e-8:
            return signal
        desired = AGC_TARGET / peak
        rate = AGC_ATTACK if desired < self.gain else AGC_RELEASE
        self.gain += rate * (desired - self.gain)
        self.gain = np.clip(self.gain, 0.1, 100.0)
        return signal * self.gain


# ============================================================================
# SERIAL READER THREAD
# ============================================================================

HEADER = b'MIC'
PKT_PAYLOAD = CHUNK_SIZE * CHANNELS * 2
PKT_TOTAL = 6 + PKT_PAYLOAD


def serial_reader_thread(ser, input_q):
    global running, stats
    buf = bytearray()
    empty_streak = 0

    while running:
        try:
            data = ser.read(8192)
            if not data:
                empty_streak += 1
                if empty_streak > 20:
                    buf.clear()
                    try:
                        ser.reset_input_buffer()
                    except Exception:
                        pass
                    empty_streak = 0
                    time.sleep(0.2)
                else:
                    time.sleep(0.005)
                continue

            empty_streak = 0
            buf.extend(data)

            if len(buf) > 150000:
                buf = buf[-PKT_TOTAL * 3:]
                stats['drops'] += 1

            while len(buf) >= PKT_TOTAL:
                idx = buf.find(HEADER)
                if idx < 0:
                    buf = buf[-2:]
                    break
                if idx > 0:
                    buf = buf[idx:]
                if len(buf) < 6:
                    break

                ns = struct.unpack('<H', buf[3:5])[0]
                nc = buf[5]

                if nc != CHANNELS or ns != CHUNK_SIZE:
                    stats['errors'] += 1
                    buf = buf[3:]
                    continue

                needed = 6 + ns * nc * 2
                if len(buf) < needed:
                    break

                raw = bytes(buf[6:needed])
                buf = buf[needed:]

                samples = np.frombuffer(raw, dtype=np.int16).reshape(ns, nc)
                audio = samples.astype(np.float32) / 32768.0

                if input_q.full():
                    try:
                        input_q.get_nowait()
                        stats['drops'] += 1
                    except queue.Empty:
                        pass

                input_q.put(audio)
                stats['packets'] += 1

        except Exception as e:
            print(f"\n[SER] {e}")
            stats['errors'] += 1
            buf.clear()
            time.sleep(0.3)


# ============================================================================
# PROCESSOR THREAD (beamforming)
# ============================================================================

def processor_thread(input_q, output_q, beamformer, agc):
    global running, stats

    prev_chunk = np.zeros((CHUNK_SIZE, CHANNELS), dtype=np.float32)
    cal_chunks = 0
    cal_target = int(NOISE_CAL_SECONDS * SAMPLE_RATE / CHUNK_SIZE)

    stats['phase'] = 'calibrating'
    print(f"\n  Noise calibration: stay QUIET for {NOISE_CAL_SECONDS}s...")

    while running:
        try:
            chunk = input_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if not beamformer.calibrated:
            beamformer.feed_noise(chunk)
            cal_chunks += 1
            if cal_chunks % 10 == 0:
                pct = min(100, int(100 * cal_chunks / cal_target))
                print(f"\r  Calibrating... {pct}%", end='', flush=True)
            if cal_chunks >= cal_target:
                beamformer.finish_calibration()
                stats['phase'] = 'beamforming'
                print("  Beamforming active!\n")
            prev_chunk = chunk.copy()
            continue

        try:
            output = beamformer.process(prev_chunk, chunk, AZIMUTH, ELEVATION)
            output = agc.apply(output)
            output = np.clip(output, -1.0, 1.0)
            out_int16 = (output * 32767).astype(np.int16)
        except Exception as e:
            print(f"\n[PROC] Error: {e}")
            stats['errors'] += 1
            # Fallback: average all channels
            out_int16 = (np.mean(chunk, axis=1) * 32767).astype(np.int16)

        if output_q.full():
            try:
                output_q.get_nowait()
                stats['drops'] += 1
            except queue.Empty:
                pass

        output_q.put(out_int16.tobytes())
        prev_chunk = chunk.copy()


# ============================================================================
# AUDIO WRITER THREAD
# ============================================================================

def audio_writer_thread(output_q, stream):
    global running
    silence = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()

    while running:
        try:
            data = output_q.get(timeout=0.05)
            stream.write(data)
        except queue.Empty:
            stream.write(silence)


# ============================================================================
# STATS THREAD
# ============================================================================

def stats_thread():
    global running, stats
    last = 0
    while running:
        time.sleep(2.0)
        p = stats['packets']
        pps = (p - last) / 2.0
        print(f"\r[{stats['phase']}] pkts={p}  {pps:.1f}/s  "
              f"err={stats['errors']}  drop={stats['drops']}",
              end='', flush=True)
        last = p


# ============================================================================
# MAIN
# ============================================================================

def main():
    global running

    print("=" * 60)
    print(" MVDR Beamforming — Sipeed 6+1 Array + Tang Nano 9K")
    print("=" * 60)
    print(f"  Port:        {SERIAL_PORT} @ {SERIAL_BAUD/1e6:.0f} Mbaud")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Channels:    {CHANNELS}")
    print(f"  MVDR band:   {FREQ_MIN}-{FREQ_MAX} Hz")
    print(f"  OLA FFT:     {OLA_NFFT} pts, hop {OLA_HOP} (50% overlap)")
    print(f"  Noise cal:   {NOISE_CAL_SECONDS}s")
    print(f"  Array gain:  ~{10*np.log10(CHANNELS):.1f} dB")

    beamformer = MVDRBeamformer(MIC_POSITIONS, SAMPLE_RATE)
    agc = SmoothAGC()
    print(f"  Active bins: {beamformer.n_band} ({FREQ_MIN}-{FREQ_MAX} Hz)")

    p = pyaudio.PyAudio()
    print("\nAudio outputs:")
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

    print(f"\nOpening {SERIAL_PORT}...")
    try:
        ser = RawSerialPort(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
        time.sleep(0.3)
        ser.reset_input_buffer()
        print(f"Connected ({'pyserial' if ser._ser else 'raw I/O'})")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    in_q = queue.Queue(maxsize=INPUT_QUEUE_SIZE)
    out_q = queue.Queue(maxsize=OUTPUT_QUEUE_SIZE)

    threads = [
        threading.Thread(target=serial_reader_thread, args=(ser, in_q), daemon=True),
        threading.Thread(target=processor_thread, args=(in_q, out_q, beamformer, agc), daemon=True),
        threading.Thread(target=audio_writer_thread, args=(out_q, stream), daemon=True),
        threading.Thread(target=stats_thread, daemon=True),
    ]
    for t in threads:
        t.start()

    print("\n" + "=" * 60)
    print(" RUNNING — press Ctrl+C to stop")
    print("=" * 60)

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
