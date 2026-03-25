#!/usr/bin/env python3
"""Diagnostic: record raw 7-channel audio from Tang Nano 9K and analyze it.

Run this BEFORE beamform2.py to verify all mics are alive and data is correct.
Saves a 7-channel WAV file and prints per-channel statistics.

Usage on Pi:
    sudo python3 diag_audio.py
"""

import struct
import numpy as np
import wave
import time
import sys
import os
import platform
import subprocess

SERIAL_BAUD = 3_000_000
SAMPLE_RATE = 16226
CHANNELS = 7
CHUNK_SIZE = 512
RECORD_SECONDS = 5
HEADER_MARKER = b'MIC'
PACKET_DATA_SIZE = CHUNK_SIZE * CHANNELS * 2
FULL_PACKET_SIZE = 6 + PACKET_DATA_SIZE

OUTPUT_WAV = 'diag_raw_7ch.wav'


def open_serial(port):
    if platform.system() == 'Windows':
        import serial
        return serial.Serial(port, SERIAL_BAUD, timeout=0.1)

    dev_name = os.path.basename(port)
    lt = f'/sys/bus/usb-serial/devices/{dev_name}/latency_timer'
    if os.path.exists(lt):
        try:
            subprocess.run(['sudo', 'tee', lt],
                           input=b'1', capture_output=True, timeout=5)
        except Exception:
            pass

    subprocess.run([
        'stty', '-F', port,
        str(SERIAL_BAUD), 'raw', '-echo', '-crtscts', '-clocal'
    ], check=True, timeout=5)

    import select
    fd = os.open(port, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    return fd, poll


def read_bytes(ser, size, timeout=0.2):
    if hasattr(ser, 'read'):
        return ser.read(size)

    fd, poll = ser
    data = bytearray()
    deadline = time.monotonic() + timeout
    while len(data) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        events = poll.poll(max(1, int(remaining * 1000)))
        if events:
            try:
                chunk = os.read(fd, size - len(data))
                if chunk:
                    data.extend(chunk)
            except BlockingIOError:
                pass
    return bytes(data)


def find_port():
    if platform.system() == 'Windows':
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            if 'COM' in p.device:
                return p.device
        return 'COM3'
    for dev in ['/dev/ttyUSB1', '/dev/ttyUSB0', '/dev/ttyACM0']:
        if os.path.exists(dev):
            return dev
    return '/dev/ttyUSB1'


def main():
    port = find_port()
    print(f"Opening {port} @ {SERIAL_BAUD/1e6:.0f} Mbaud...")
    ser = open_serial(port)

    needed_packets = int(RECORD_SECONDS * SAMPLE_RATE / CHUNK_SIZE) + 5
    all_audio = []
    packets = 0
    errors = 0

    print(f"Recording {RECORD_SECONDS}s of audio ({needed_packets} packets)...")
    print("Speak, clap, or play music near the mic array.\n")

    buf = bytearray()
    t_start = time.time()

    while packets < needed_packets and (time.time() - t_start) < RECORD_SECONDS + 10:
        new_data = read_bytes(ser, 8192, timeout=0.2)
        if not new_data:
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
                errors += 1
                buf = buf[3:]
                continue

            needed = 6 + n_samples * n_channels * 2
            if len(buf) < needed:
                break

            audio_bytes = bytes(buf[6:needed])
            buf = buf[needed:]

            audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_data = audio_int.reshape(n_samples, n_channels)
            all_audio.append(audio_data)
            packets += 1

            if packets % 10 == 0:
                print(f"\r  Captured {packets}/{needed_packets} packets...", end='', flush=True)

    elapsed = time.time() - t_start
    print(f"\n\nDone: {packets} packets in {elapsed:.1f}s ({errors} errors)")

    if packets == 0:
        print("ERROR: No packets captured! Check FPGA connection.")
        sys.exit(1)

    audio = np.concatenate(all_audio, axis=0)
    total_samples = audio.shape[0]
    duration = total_samples / SAMPLE_RATE

    print(f"\nTotal: {total_samples} samples ({duration:.2f}s) × {CHANNELS} channels")
    print(f"{'':->60}")
    print(f"{'Channel':<10} {'RMS':>8} {'Peak':>8} {'dBFS':>8} {'Status':>10}")
    print(f"{'':->60}")

    ch_names = ['D0-L(0°)', 'D0-R(60°)', 'D1-L(120°)', 'D1-R(180°)',
                'D2-L(240°)', 'D2-R(300°)', 'D3-L(ctr)']

    alive_count = 0
    for ch in range(CHANNELS):
        samples = audio[:, ch].astype(np.float64)
        rms = np.sqrt(np.mean(samples ** 2))
        peak = np.max(np.abs(samples))
        dbfs = 20 * np.log10(rms / 32768.0 + 1e-10)
        status = "OK" if rms > 50 else ("WEAK" if rms > 5 else "DEAD")
        if rms > 50:
            alive_count += 1
        print(f"  ch{ch} {ch_names[ch]:<12} {rms:8.1f} {peak:8.0f} {dbfs:8.1f} {status:>10}")

    print(f"{'':->60}")
    print(f"\nAlive channels: {alive_count}/{CHANNELS}")

    if alive_count < 3:
        print("\nWARNING: Too few live channels — check I2S wiring and pin mapping!")

    # Save WAV
    with wave.open(OUTPUT_WAV, 'w') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    print(f"\nSaved: {OUTPUT_WAV} ({os.path.getsize(OUTPUT_WAV) / 1024:.0f} KB)")
    print("Play individual channels with: "
          "sox diag_raw_7ch.wav ch0.wav remix 1")

    if hasattr(ser, 'close'):
        ser.close()
    else:
        os.close(ser[0])


if __name__ == '__main__':
    main()
