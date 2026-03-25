#!/usr/bin/env python3
"""
Real-time Frequency-Domain Audio Beamforming for Sipeed 6+1 Mic Array
STFT-based MVDR/GEV beamforming with proper complex steering vectors
Optimized for Raspberry Pi 4B - FRONT-FACING CONFIGURATION
"""

import serial
import struct
import numpy as np
import pyaudio
from scipy.linalg import eigh
from scipy.signal import stft, istft
from collections import deque
import threading
import time
import sys

SERIAL_PORT = '/dev/ttyACM0'
SERIAL_BAUD = 2000000

# Audio parameters
SAMPLE_RATE = 16000
CHANNELS = 2  # Only using 2 mics from D0 line (left + right)
CHUNK_SIZE = 512  # Time-domain chunk from ESP32
BITS_PER_SAMPLE = 16

# STFT parameters for frequency-domain processing
NFFT = 512  # FFT size
OVERLAP = NFFT // 2  # 50% overlap
HOP_LENGTH = NFFT - OVERLAP

# Microphone array geometry (in meters) - ENDFIRE ARRAY (both facing front)
# Both mics face forward (+X direction), arranged along X-axis
# MEASURE THE DISTANCE BETWEEN YOUR TWO MICS!
MIC_SPACING = 0.065  # Distance between mics in meters - MEASURE THIS!
MIC_POSITIONS = np.array([
    [0.0, 0.0, 0.0],              # Ch0: Rear mic (reference)
    [MIC_SPACING, 0.0, 0.0],      # Ch1: Front mic (toward target)
])
# This creates maximum sensitivity along +X axis (front direction)

# Beamforming parameters
BEAMFORMER_TYPE = 'MVDR'  # 'MVDR' or 'GEV' or 'DelaySum'
SPEED_OF_SOUND = 343.0  # m/s
NOISE_UPDATE_RATE = 0.2  # Update noise covariance every 0.2s
REGULARIZATION = 1e-3  # Diagonal loading

# Frequency range for beamforming (speech band)
FREQ_MIN = 300   # Hz - below this, use delay-sum
FREQ_MAX = 4000  # Hz - above this, use delay-sum

# TARGET DIRECTION (FRONT-FACING)
# Azimuth angle for "front" - adjust based on your physical setup
# If your device points along +X axis, use AZIMUTH_FRONT = 0
# If your device points along +Y axis, use AZIMUTH_FRONT = 90
AZIMUTH_FRONT = 0.0  # degrees (0° = +X axis, 90° = +Y axis)
ELEVATION_FRONT = 0.0  # degrees (0° = horizontal plane)

# Zoom mapping (if needed later)
ZOOM_MIN_ANGLE = 12
ZOOM_MAX_ANGLE = 60

# ============================================================================
# GLOBAL STATE
# ============================================================================

current_zoom_angle = 60.0
current_azimuth = AZIMUTH_FRONT  # Target direction
current_elevation = ELEVATION_FRONT
audio_output = None
running = True
stats = {'packets': 0, 'errors': 0, 'last_time': time.time()}

# ============================================================================
# FREQUENCY-DOMAIN BEAMFORMING
# ============================================================================

class FrequencyDomainBeamformer:
    def __init__(self, mic_positions, sample_rate):
        self.mic_positions = mic_positions
        self.n_mics = len(mic_positions)
        self.sample_rate = sample_rate
        self.nfft = NFFT
        self.hop_length = HOP_LENGTH

        # Frequency bins
        self.freqs = np.fft.rfftfreq(self.nfft, 1/sample_rate)
        self.n_freqs = len(self.freqs)

        # Frequency range for beamforming
        self.freq_min_idx = np.argmin(np.abs(self.freqs - FREQ_MIN))
        self.freq_max_idx = np.argmin(np.abs(self.freqs - FREQ_MAX))

        # Noise covariance matrices (one per frequency bin)
        self.noise_cov = np.zeros((self.n_freqs, self.n_mics, self.n_mics), dtype=complex)
        for f in range(self.n_freqs):
            self.noise_cov[f] = np.eye(self.n_mics) * REGULARIZATION

        # Buffer for noise estimation
        self.noise_buffer = deque(maxlen=50)
        self.noise_update_counter = 0

        # Overlap-add buffer for synthesis
        self.synthesis_buffer = np.zeros(self.nfft)
        self.window = np.hanning(self.nfft)

        print(f"  Frequency bins: {self.n_freqs}")
        print(f"  Beamforming range: {FREQ_MIN}-{FREQ_MAX} Hz (bins {self.freq_min_idx}-{self.freq_max_idx})")
        print(f"  Target direction: Azimuth={AZIMUTH_FRONT}°, Elevation={ELEVATION_FRONT}°")

    def steering_vector(self, azimuth_deg, freq_hz, elevation_deg=0):
        """Compute frequency-domain steering vector

        This is the CORRECT formula for narrowband frequency-domain beamforming

        Args:
            azimuth_deg: Target azimuth angle (0° = front/+X, 90° = right/+Y)
            freq_hz: Frequency in Hz
            elevation_deg: Elevation angle (0° = horizontal)

        Returns:
            Complex steering vector of shape (n_mics,)
        """
        az_rad = np.radians(azimuth_deg)
        el_rad = np.radians(elevation_deg)

        # Direction of arrival (DOA) unit vector
        # Standard spherical coordinates:
        # x = cos(el) * cos(az)
        # y = cos(el) * sin(az)
        # z = sin(el)
        doa = np.array([
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad)
        ])

        # Time delays for each microphone (in seconds)
        # Positive delay = signal arrives later at this mic
        delays = np.dot(self.mic_positions, doa) / SPEED_OF_SOUND

        # Convert to phase shifts at this frequency
        # CRITICAL: omega = 2*pi*f, steering_vector = exp(-j*omega*tau)
        # The negative sign is for signal arriving FROM direction doa
        omega = 2 * np.pi * freq_hz
        steering = np.exp(-1j * omega * delays)

        # Normalize to unit magnitude
        return steering / np.linalg.norm(steering)

    def update_noise_covariance(self, stft_data):
        """Update noise covariance from STFT data

        Args:
            stft_data: Complex array of shape (n_freqs, n_mics, n_frames)
        """
        self.noise_buffer.append(stft_data)

        if len(self.noise_buffer) >= 10:
            # Stack all frames
            all_frames = np.concatenate(list(self.noise_buffer), axis=2)  # (n_freqs, n_mics, many_frames)

            # Compute covariance for each frequency
            for f in range(self.n_freqs):
                X = all_frames[f, :, :]  # (n_mics, n_frames)
                # Covariance: R = E[X * X^H] = (X @ X^H) / n_frames
                self.noise_cov[f] = (X @ X.conj().T) / X.shape[1]
                # Add regularization
                self.noise_cov[f] += np.eye(self.n_mics) * REGULARIZATION


class MVDRFrequencyBeamformer(FrequencyDomainBeamformer):
    """MVDR Beamformer in Frequency Domain - CORRECT FORMULA

    MVDR (Minimum Variance Distortionless Response) beamformer:
    - Maintains unit gain in the look direction
    - Minimizes output power (suppresses noise/interference)
    - Optimal for known target direction
    """

    def process_chunk(self, audio_chunk, azimuth_deg, elevation_deg=0):
        """Process one chunk of multi-channel audio

        Args:
            audio_chunk: (n_samples, n_mics) time-domain audio
            azimuth_deg: Target azimuth angle
            elevation_deg: Target elevation angle

        Returns:
            (n_samples,) beamformed mono audio
        """
        n_samples = audio_chunk.shape[0]

        # Apply STFT to each channel
        stft_data = []
        for mic in range(self.n_mics):
            f, t, Zxx = stft(audio_chunk[:, mic],
                            fs=self.sample_rate,
                            window=self.window,
                            nperseg=self.nfft,
                            noverlap=self.hop_length)
            stft_data.append(Zxx)

        # Shape: (n_mics, n_freqs, n_frames)
        stft_data = np.array(stft_data)
        # Transpose to: (n_freqs, n_mics, n_frames)
        stft_data = stft_data.transpose(1, 0, 2)

        n_frames = stft_data.shape[2]

        # Update noise covariance periodically
        self.noise_update_counter += 1
        if self.noise_update_counter % int(NOISE_UPDATE_RATE * self.sample_rate / CHUNK_SIZE) == 0:
            self.update_noise_covariance(stft_data)

        # Process each frequency bin
        output_stft = np.zeros((self.n_freqs, n_frames), dtype=complex)

        for f in range(self.n_freqs):
            freq_hz = self.freqs[f]

            # Only beamform in speech band, use delay-sum elsewhere
            if freq_hz < FREQ_MIN or freq_hz > FREQ_MAX:
                # Simple delay-and-sum for low/high frequencies
                a = self.steering_vector(azimuth_deg, freq_hz, elevation_deg)
                w = a / self.n_mics
            else:
                # MVDR beamforming
                # Get data for this frequency: (n_mics, n_frames)
                X = stft_data[f, :, :]

                # Compute spatial covariance matrix R = E[X * X^H]
                R = (X @ X.conj().T) / n_frames
                R += np.eye(self.n_mics) * REGULARIZATION

                # Steering vector for target direction
                a = self.steering_vector(azimuth_deg, freq_hz, elevation_deg)

                # MVDR weights formula:
                # w = (R^-1 @ a) / (a^H @ R^-1 @ a)
                try:
                    R_inv = np.linalg.inv(R)
                    w_numerator = R_inv @ a
                    w_denominator = np.conj(a) @ R_inv @ a
                    w = w_numerator / (w_denominator + 1e-10)
                except np.linalg.LinAlgError:
                    # Fallback to delay-and-sum
                    w = a / self.n_mics

            # Apply weights: y = w^H @ X
            output_stft[f, :] = np.conj(w) @ stft_data[f, :, :]

        # Inverse STFT to get time-domain signal
        _, output_time = istft(output_stft,
                               fs=self.sample_rate,
                               window=self.window,
                               nperseg=self.nfft,
                               noverlap=self.hop_length)

        # Match output length to input
        if len(output_time) < n_samples:
            output_time = np.pad(output_time, (0, n_samples - len(output_time)))
        else:
            output_time = output_time[:n_samples]

        # Normalize to prevent clipping
        max_val = np.max(np.abs(output_time))
        if max_val > 0.8:
            output_time = output_time * 0.8 / max_val

        return output_time


class GEVFrequencyBeamformer(FrequencyDomainBeamformer):
    """GEV Beamformer in Frequency Domain - CORRECT FORMULA

    GEV (Generalized Eigenvalue) beamformer:
    - Maximizes signal-to-noise ratio
    - Adapts to noise characteristics
    - Better for non-stationary noise
    """

    def process_chunk(self, audio_chunk, azimuth_deg, elevation_deg=0):
        """Process one chunk with GEV beamforming"""
        n_samples = audio_chunk.shape[0]

        # Apply STFT
        stft_data = []
        for mic in range(self.n_mics):
            f, t, Zxx = stft(audio_chunk[:, mic],
                            fs=self.sample_rate,
                            window=self.window,
                            nperseg=self.nfft,
                            noverlap=self.hop_length)
            stft_data.append(Zxx)

        stft_data = np.array(stft_data).transpose(1, 0, 2)
        n_frames = stft_data.shape[2]

        # Update noise covariance
        self.noise_update_counter += 1
        if self.noise_update_counter % int(NOISE_UPDATE_RATE * self.sample_rate / CHUNK_SIZE) == 0:
            self.update_noise_covariance(stft_data)

        # Process each frequency
        output_stft = np.zeros((self.n_freqs, n_frames), dtype=complex)

        for f in range(self.n_freqs):
            freq_hz = self.freqs[f]

            if freq_hz < FREQ_MIN or freq_hz > FREQ_MAX:
                a = self.steering_vector(azimuth_deg, freq_hz, elevation_deg)
                w = a / self.n_mics
            else:
                X = stft_data[f, :, :]

                # Signal covariance
                R_signal = (X @ X.conj().T) / n_frames
                R_signal += np.eye(self.n_mics) * REGULARIZATION

                # Noise covariance
                R_noise = self.noise_cov[f]

                # GEV: solve R_signal @ w = lambda * R_noise @ w
                try:
                    eigenvalues, eigenvectors = eigh(R_signal, R_noise)
                    # Largest eigenvalue eigenvector
                    w = eigenvectors[:, -1]

                    # Apply distortionless constraint toward target
                    a = self.steering_vector(azimuth_deg, freq_hz, elevation_deg)
                    w = w * (np.conj(a) @ w)

                except np.linalg.LinAlgError:
                    a = self.steering_vector(azimuth_deg, freq_hz, elevation_deg)
                    w = a / self.n_mics

            output_stft[f, :] = np.conj(w) @ stft_data[f, :, :]

        # Inverse STFT
        _, output_time = istft(output_stft,
                               fs=self.sample_rate,
                               window=self.window,
                               nperseg=self.nfft,
                               noverlap=self.hop_length)

        if len(output_time) < n_samples:
            output_time = np.pad(output_time, (0, n_samples - len(output_time)))
        else:
            output_time = output_time[:n_samples]

        # Normalize
        max_val = np.max(np.abs(output_time))
        if max_val > 0.8:
            output_time = output_time * 0.8 / max_val

        return output_time


class DelayAndSumFrequencyBeamformer(FrequencyDomainBeamformer):
    """Simple Delay-and-Sum in Frequency Domain

    Classic beamforming approach:
    - Phase-aligns signals from target direction
    - Simple and robust
    - Good starting point for testing
    """

    def process_chunk(self, audio_chunk, azimuth_deg, elevation_deg=0):
        """Process with frequency-domain delay-and-sum"""
        n_samples = audio_chunk.shape[0]

        # Apply STFT
        stft_data = []
        for mic in range(self.n_mics):
            f, t, Zxx = stft(audio_chunk[:, mic],
                            fs=self.sample_rate,
                            window=self.window,
                            nperseg=self.nfft,
                            noverlap=self.hop_length)
            stft_data.append(Zxx)

        stft_data = np.array(stft_data).transpose(1, 0, 2)
        n_frames = stft_data.shape[2]

        # Process each frequency
        output_stft = np.zeros((self.n_freqs, n_frames), dtype=complex)

        for f in range(self.n_freqs):
            freq_hz = self.freqs[f]

            # Steering vector (phase alignment)
            a = self.steering_vector(azimuth_deg, freq_hz, elevation_deg)

            # Delay-and-sum weights
            w = a / self.n_mics

            # Apply: y = w^H @ X
            output_stft[f, :] = np.conj(w) @ stft_data[f, :, :]

        # Inverse STFT
        _, output_time = istft(output_stft,
                               fs=self.sample_rate,
                               window=self.window,
                               nperseg=self.nfft,
                               noverlap=self.hop_length)

        if len(output_time) < n_samples:
            output_time = np.pad(output_time, (0, n_samples - len(output_time)))
        else:
            output_time = output_time[:n_samples]

        # Normalize
        max_val = np.max(np.abs(output_time))
        if max_val > 0.8:
            output_time = output_time * 0.8 / max_val

        return output_time

# ============================================================================
# SERIAL RECEIVER
# ============================================================================

def read_audio_stream(ser, beamformer):
    """Read audio packets from ESP32 and apply beamforming"""
    global running, stats, audio_output, current_azimuth, current_elevation

    print("Starting audio receiver...")

    sync_buffer = bytearray()

    def find_mic_header():
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
            if not find_mic_header():
                continue

            header_rest = ser.read(3)
            if len(header_rest) != 3:
                stats['errors'] += 1
                continue

            n_samples = struct.unpack('<H', header_rest[:2])[0]
            n_channels = header_rest[2]

            if n_channels != CHANNELS or n_samples > 2048:
                stats['errors'] += 1
                continue

            data_size = n_samples * n_channels * 2
            audio_bytes = ser.read(data_size)

            if len(audio_bytes) != data_size:
                stats['errors'] += 1
                continue

            # Convert to numpy array
            audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_data = audio_int.reshape(n_samples, n_channels).astype(np.float32) / 32768.0

            # Apply beamforming - POINTING TO FRONT
            output_audio = beamformer.process_chunk(audio_data, current_azimuth, current_elevation)

            # Final clipping
            output_audio = np.clip(output_audio, -1.0, 1.0)
            output_int16 = (output_audio * 32767).astype(np.int16)

            # Send to audio output
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
    """Thread to update beamforming direction

    Currently fixed to front-facing, but can be modified for dynamic steering
    """
    global current_azimuth, current_elevation, running

    print("Direction control thread started (fixed front-facing)")

    while running:
        # Keep pointing forward
        # Modify this if you want dynamic steering based on DOA estimation
        time.sleep(0.1)

# ============================================================================
# AUDIO OUTPUT
# ============================================================================

def setup_audio_output():
    """Initialize PyAudio for playback"""
    global audio_output

    p = pyaudio.PyAudio()

    print("\nAvailable audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"  [{i}] {info['name']} (out: {info['maxOutputChannels']})")

    audio_output = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=CHUNK_SIZE
    )

    print(f"\n✓ Audio output initialized ({SAMPLE_RATE} Hz, mono)")
    return p

# ============================================================================
# STATISTICS
# ============================================================================

def stats_thread():
    """Print statistics"""
    global running, stats, current_azimuth, current_elevation

    last_packets = 0

    while running:
        time.sleep(2.0)

        packets = stats['packets']
        errors = stats['errors']
        pps = (packets - last_packets) / 2.0

        print(f"\r[STATS] Packets: {packets} | Rate: {pps:.1f} pkt/s | "
              f"Errors: {errors} | Direction: Az={current_azimuth:.1f}° El={current_elevation:.1f}° | "
              f"Latency: ~{NFFT/SAMPLE_RATE*1000:.1f}ms", end='', flush=True)

        last_packets = packets

# ============================================================================
# MAIN
# ============================================================================

def main():
    global running, audio_output

    print("=" * 70)
    print("FREQUENCY-DOMAIN Audio Beamforming - FRONT-FACING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Sample Rate: {SAMPLE_RATE} Hz")
    print(f"  Channels: {CHANNELS} (ENDFIRE - 2 mics both facing FRONT)")
    print(f"  FFT Size: {NFFT}, Hop: {HOP_LENGTH}")
    print(f"  Beamformer: {BEAMFORMER_TYPE}")
    print(f"  Mic Spacing: {MIC_SPACING*1000:.1f} mm front-to-back")
    print(f"  Target Direction: FORWARD (+X axis)")
    print(f"  Serial Port: {SERIAL_PORT} @ {SERIAL_BAUD} baud")
    print(f"\n  *** ENDFIRE CONFIGURATION ***")
    print(f"  Maximum gain: FRONT (0°)")
    print(f"  Maximum rejection: REAR (180°)")

    # Initialize beamformer
    if BEAMFORMER_TYPE == 'MVDR':
        beamformer = MVDRFrequencyBeamformer(MIC_POSITIONS, SAMPLE_RATE)
    elif BEAMFORMER_TYPE == 'GEV':
        beamformer = GEVFrequencyBeamformer(MIC_POSITIONS, SAMPLE_RATE)
    else:
        beamformer = DelayAndSumFrequencyBeamformer(MIC_POSITIONS, SAMPLE_RATE)

    print(f"\n✓ Frequency-domain beamformer initialized")

    # Setup audio output
    p_audio = setup_audio_output()

    # Open serial
    print(f"\nConnecting to ESP32 on {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        print("✓ Serial connection established")
    except Exception as e:
        print(f"ERROR: Could not open serial port: {e}")
        sys.exit(1)

    # Start threads
    audio_thread = threading.Thread(target=read_audio_stream, args=(ser, beamformer), daemon=True)
    direction_thread = threading.Thread(target=direction_control_thread, daemon=True)
    stats_thread_obj = threading.Thread(target=stats_thread, daemon=True)

    audio_thread.start()
    direction_thread.start()
    stats_thread_obj.start()

    print("\n" + "=" * 70)
    print("SYSTEM RUNNING - Beamforming to FRONT")
    print("Press Ctrl+C to stop")
    print("=" * 70 + "\n")

    try:
        while True:
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

        print("✓ Shutdown complete")

if __name__ == '__main__':
    main()
