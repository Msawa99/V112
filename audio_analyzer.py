import librosa
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
from pydub import AudioSegment
import soundfile as sf
import python_speech_features
import noisereduce as nr
import webrtcvad
import warnings
warnings.filterwarnings("ignore")

class AudioAnalyzer:
    """Comprehensive audio analysis class using multiple libraries"""
    
    def __init__(self):
        self.supported_formats = ['wav', 'mp3', 'flac', 'm4a', 'aac', 'ogg']
    
    def analyze_file(self, file_path, sample_rate=22050, hop_length=1024, n_mfcc=13, extract_prosody=True):
        """Analyze a single audio file and extract comprehensive features"""
        
        features = {}
        
        try:
            # Load audio file with librosa
            y, sr = librosa.load(file_path, sr=sample_rate)
            
            # Basic file information
            features.update(self._extract_basic_info(file_path, y, sr))
            
            # Spectral features using librosa
            features.update(self._extract_librosa_features(y, sr, hop_length, n_mfcc))
            
            # PyDub features
            features.update(self._extract_pydub_features(file_path))
            
            # SciPy signal processing features
            features.update(self._extract_scipy_features(y, sr))
            
            # Enhanced spectral features using python_speech_features
            features.update(self._extract_speech_features(y, sr))
            
            # Noise reduction and voice activity detection
            features.update(self._extract_voice_activity_features(y, sr))
            
            # Prosodic features (simplified without Parselmouth)
            if extract_prosody:
                features.update(self._extract_basic_prosodic_features(y, sr))
            
            # Additional spectral and rhythmic features
            features.update(self._extract_advanced_features(y, sr, hop_length))
            
            # Enhanced features using FeatureExtractor
            from feature_extractor import FeatureExtractor
            enhanced_features = FeatureExtractor.extract_all_enhanced_features(y, sr, hop_length)
            features.update(enhanced_features)
            
        except Exception as e:
            raise Exception(f"Error analyzing file: {str(e)}")
        
        return features
    
    def _extract_basic_info(self, file_path, y, sr):
        """Extract basic audio file information"""
        
        features = {}
        features['duration'] = len(y) / sr
        features['sample_rate'] = sr
        features['samples'] = len(y)
        features['channels'] = 1  # librosa loads as mono by default
        
        return features
    
    def _extract_librosa_features(self, y, sr, hop_length, n_mfcc):
        """Extract features using librosa library"""
        
        features = {}
        
        try:
            # MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
            for i in range(n_mfcc):
                features[f'mfcc_mean_{i+1}'] = np.mean(mfccs[i])
                features[f'mfcc_std_{i+1}'] = np.std(mfccs[i])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
            for i in range(12):
                features[f'chroma_{i+1}_mean'] = np.mean(chroma[i])
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroid)
            features['spectral_centroid_std'] = np.std(spectral_centroid)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            # Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            features['tempo'] = tempo
            features['beats_count'] = len(beats)
            
            # RMS Energy
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
            features['spectral_contrast_mean'] = np.mean(contrast)
            features['spectral_contrast_std'] = np.std(contrast)
            
            # Tonnetz (harmonic network)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features['tonnetz_mean'] = np.mean(tonnetz)
            features['tonnetz_std'] = np.std(tonnetz)
            
        except Exception as e:
            print(f"Warning: Error extracting librosa features: {str(e)}")
        
        return features
    
    def _extract_pydub_features(self, file_path):
        """Extract features using PyDub library"""
        
        features = {}
        
        try:
            # Load with PyDub
            audio = AudioSegment.from_file(file_path)
            
            # Basic properties
            features['pydub_duration'] = len(audio) / 1000.0  # Convert to seconds
            features['pydub_frame_rate'] = audio.frame_rate
            features['pydub_channels'] = audio.channels
            features['pydub_sample_width'] = audio.sample_width
            
            # Convert to mono for analysis
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Signal strength (dBFS)
            features['dbfs'] = audio.dBFS
            features['max_dbfs'] = audio.max_dBFS
            
            # RMS and peak analysis
            raw_audio = audio.raw_data
            samples = np.frombuffer(raw_audio, dtype=np.int16)
            features['pydub_rms'] = np.sqrt(np.mean(samples**2))
            features['pydub_peak'] = np.max(np.abs(samples))
            
        except Exception as e:
            print(f"Warning: Error extracting PyDub features: {str(e)}")
        
        return features
    
    def _extract_scipy_features(self, y, sr):
        """Extract features using SciPy signal processing"""
        
        features = {}
        
        try:
            # FFT analysis
            fft = np.fft.fft(y)
            magnitude = np.abs(fft)
            
            # Power spectral density
            freqs, psd = signal.periodogram(y, sr)
            
            # Spectral statistics
            features['fft_mean'] = np.mean(magnitude)
            features['fft_std'] = np.std(magnitude)
            features['fft_skew'] = skew(magnitude)
            features['fft_kurtosis'] = kurtosis(magnitude)
            
            # Power spectrum statistics
            features['psd_mean'] = np.mean(psd)
            features['psd_std'] = np.std(psd)
            features['psd_max'] = np.max(psd)
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(psd)
            features['dominant_frequency'] = freqs[dominant_freq_idx]
            
            # Spectral entropy
            normalized_psd = psd / np.sum(psd)
            spectral_entropy = -np.sum(normalized_psd * np.log2(normalized_psd + 1e-12))
            features['spectral_entropy'] = spectral_entropy
            
            # Signal statistics
            features['signal_mean'] = np.mean(y)
            features['signal_std'] = np.std(y)
            features['signal_skew'] = skew(y)
            features['signal_kurtosis'] = kurtosis(y)
            features['signal_min'] = np.min(y)
            features['signal_max'] = np.max(y)
            
        except Exception as e:
            print(f"Warning: Error extracting SciPy features: {str(e)}")
        
        return features
    
    def _extract_speech_features(self, y, sr):
        """Extract features using python_speech_features library"""
        
        features = {}
        
        try:
            # Convert to 16-bit integer format for speech features
            y_int = (y * 32767).astype(np.int16)
            
            # MFCC using python_speech_features
            mfcc_features = python_speech_features.mfcc(
                y_int, sr, numcep=13, nfilt=26, nfft=1024
            )
            
            # Statistics for each MFCC coefficient
            for i in range(mfcc_features.shape[1]):
                features[f'psf_mfcc_mean_{i+1}'] = np.mean(mfcc_features[:, i])
                features[f'psf_mfcc_std_{i+1}'] = np.std(mfcc_features[:, i])
            
            # Log filterbank energies
            fbank_features = python_speech_features.logfbank(
                y_int, sr, nfilt=26, nfft=1024
            )
            features['fbank_mean'] = np.mean(fbank_features)
            features['fbank_std'] = np.std(fbank_features)
            
            # Delta features (first derivatives)
            delta_mfcc = python_speech_features.delta(mfcc_features, 2)
            features['delta_mfcc_mean'] = np.mean(delta_mfcc)
            features['delta_mfcc_std'] = np.std(delta_mfcc)
            
            # Spectral subband centroids (SSC)
            ssc_features = python_speech_features.ssc(
                y_int, sr, nfilt=26, nfft=1024
            )
            features['ssc_mean'] = np.mean(ssc_features)
            features['ssc_std'] = np.std(ssc_features)
            
        except Exception as e:
            print(f"Warning: Error extracting speech features: {str(e)}")
        
        return features
    
    def _extract_voice_activity_features(self, y, sr):
        """Extract voice activity and noise reduction features"""
        
        features = {}
        
        try:
            # Noise reduction
            y_reduced = nr.reduce_noise(y=y, sr=sr)
            
            # Noise level estimation
            noise_level = np.mean(np.abs(y - y_reduced))
            features['noise_level'] = noise_level
            features['signal_to_noise_ratio'] = np.mean(np.abs(y_reduced)) / (noise_level + 1e-8)
            
            # Voice Activity Detection using WebRTC VAD
            if sr == 16000:  # WebRTC VAD works best at 16kHz
                vad = webrtcvad.Vad(2)  # Aggressiveness level 2
                
                # Convert to 16-bit PCM
                y_vad = (y * 32767).astype(np.int16)
                
                # Process in 30ms frames
                frame_duration = 30  # milliseconds
                frame_size = int(sr * frame_duration / 1000)
                
                voice_frames = 0
                total_frames = 0
                
                for i in range(0, len(y_vad) - frame_size, frame_size):
                    frame = y_vad[i:i + frame_size]
                    if len(frame) == frame_size:
                        try:
                            is_speech = vad.is_speech(frame.tobytes(), sr)
                            if is_speech:
                                voice_frames += 1
                            total_frames += 1
                        except:
                            pass
                
                if total_frames > 0:
                    features['voice_activity_ratio'] = voice_frames / total_frames
                else:
                    features['voice_activity_ratio'] = 0
            else:
                # Fallback energy-based VAD for other sample rates
                frame_length = 1024
                hop_length = 512
                
                # Frame-based energy
                frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
                energy = np.sum(frames ** 2, axis=0)
                
                # Threshold-based voice activity
                energy_threshold = np.percentile(energy, 30)
                voice_frames = np.sum(energy > energy_threshold)
                features['voice_activity_ratio'] = voice_frames / len(energy)
            
            # Spectral flux (measure of spectral change)
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            spectral_flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
            features['spectral_flux_mean'] = np.mean(spectral_flux)
            features['spectral_flux_std'] = np.std(spectral_flux)
            
        except Exception as e:
            print(f"Warning: Error extracting voice activity features: {str(e)}")
            # Set default values
            features['noise_level'] = 0
            features['signal_to_noise_ratio'] = 1
            features['voice_activity_ratio'] = 0.5
            features['spectral_flux_mean'] = 0
            features['spectral_flux_std'] = 0
        
        return features
    
    def _extract_basic_prosodic_features(self, y, sr):
        """Extract basic prosodic features using librosa and scipy"""
        
        features = {}
        
        try:
            # Pitch estimation using librosa piptrack
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1, fmin=50, fmax=400)
            
            # Extract pitch values
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_min'] = np.min(pitch_values)
                features['pitch_max'] = np.max(pitch_values)
                features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_min'] = 0
                features['pitch_max'] = 0
                features['pitch_range'] = 0
            
            # Estimate formants using spectral peaks
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            freqs = librosa.fft_frequencies(sr=sr)
            
            # Find spectral peaks as formant estimates
            formant_estimates = []
            for frame in magnitude.T:
                # Smooth the spectrum
                smoothed = signal.savgol_filter(frame, window_length=5, polyorder=2)
                peaks, _ = signal.find_peaks(smoothed, height=np.max(smoothed) * 0.1)
                frame_formants = freqs[peaks][:3]  # Take first 3 peaks as formants
                formant_estimates.append(frame_formants)
            
            # Calculate formant statistics
            for i in range(3):
                formant_values = []
                for frame_formants in formant_estimates:
                    if len(frame_formants) > i and frame_formants[i] > 0:
                        formant_values.append(frame_formants[i])
                
                if formant_values:
                    features[f'formant_{i+1}_mean'] = np.mean(formant_values)
                    features[f'formant_{i+1}_std'] = np.std(formant_values)
                else:
                    features[f'formant_{i+1}_mean'] = 0
                    features[f'formant_{i+1}_std'] = 0
            
            # Simplified jitter calculation
            if len(pitch_values) > 1:
                pitch_periods = 1.0 / np.array(pitch_values)
                period_diffs = np.diff(pitch_periods)
                if len(period_diffs) > 0:
                    features['jitter'] = np.std(period_diffs) / np.mean(pitch_periods)
                else:
                    features['jitter'] = 0
            else:
                features['jitter'] = 0
            
            # Simplified shimmer calculation using RMS energy
            frame_length = 1024
            hop_length = 512
            frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
            rms_energy = np.sqrt(np.mean(frames ** 2, axis=0))
            
            if len(rms_energy) > 1:
                energy_diffs = np.diff(rms_energy)
                features['shimmer'] = np.std(energy_diffs) / np.mean(rms_energy)
            else:
                features['shimmer'] = 0
            
            # Harmonics-to-noise ratio estimation
            # Use spectral harmonicity as a proxy
            harmonic_energy = np.sum(magnitude[freqs <= sr//4]**2, axis=0)
            total_energy = np.sum(magnitude**2, axis=0)
            harmonicity_ratio = harmonic_energy / (total_energy + 1e-8)
            
            features['hnr_mean'] = np.mean(harmonicity_ratio)
            features['hnr_std'] = np.std(harmonicity_ratio)
            
        except Exception as e:
            print(f"Warning: Error extracting basic prosodic features: {str(e)}")
            # Set default values if extraction fails
            default_features = [
                'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_range',
                'hnr_mean', 'hnr_std', 'jitter', 'shimmer',
                'formant_1_mean', 'formant_1_std',
                'formant_2_mean', 'formant_2_std',
                'formant_3_mean', 'formant_3_std'
            ]
            for feat in default_features:
                if feat not in features:
                    features[feat] = 0
        
        return features
    
    def _extract_advanced_features(self, y, sr, hop_length):
        """Extract additional advanced features"""
        
        features = {}
        
        try:
            # Mel-scale spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
            features['mel_spec_mean'] = np.mean(mel_spec)
            features['mel_spec_std'] = np.std(mel_spec)
            
            # Log-power Mel spectrogram
            log_mel = librosa.power_to_db(mel_spec)
            features['log_mel_mean'] = np.mean(log_mel)
            features['log_mel_std'] = np.std(log_mel)
            
            # Delta features (first-order derivatives)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
            delta_mfcc = librosa.feature.delta(mfccs)
            features['delta_mfcc_mean'] = np.mean(delta_mfcc)
            features['delta_mfcc_std'] = np.std(delta_mfcc)
            
            # Delta-delta features (second-order derivatives)
            delta2_mfcc = librosa.feature.delta(mfccs, order=2)
            features['delta2_mfcc_mean'] = np.mean(delta2_mfcc)
            features['delta2_mfcc_std'] = np.std(delta2_mfcc)
            
            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
            features['onset_count'] = len(onset_frames)
            features['onset_rate'] = len(onset_frames) / (len(y) / sr)
            
            # Tempogram
            tempogram = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length)
            features['tempogram_mean'] = np.mean(tempogram)
            features['tempogram_std'] = np.std(tempogram)
            
            # Spectral flatness (measure of noisiness)
            spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)
            features['spectral_flatness_mean'] = np.mean(spectral_flatness)
            features['spectral_flatness_std'] = np.std(spectral_flatness)
            
            # Polyrhythm features
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
                beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
                if len(beat_times) > 1:
                    beat_intervals = np.diff(beat_times)
                    features['beat_interval_mean'] = np.mean(beat_intervals)
                    features['beat_interval_std'] = np.std(beat_intervals)
                else:
                    features['beat_interval_mean'] = 0
                    features['beat_interval_std'] = 0
            except:
                features['beat_interval_mean'] = 0
                features['beat_interval_std'] = 0
            
        except Exception as e:
            print(f"Warning: Error extracting advanced features: {str(e)}")
        
        return features
