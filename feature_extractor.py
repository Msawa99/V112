import numpy as np
import pandas as pd
from scipy.signal import find_peaks, spectrogram
from scipy.stats import entropy
import librosa
import python_speech_features

class FeatureExtractor:
    """Additional feature extraction utilities"""
    
    @staticmethod
    def extract_rhythm_features(y, sr, hop_length=512):
        """Extract rhythm and timing features"""
        
        features = {}
        
        try:
            # Beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            
            # Beat consistency
            if len(beats) > 2:
                beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
                beat_intervals = np.diff(beat_times)
                
                features['rhythm_regularity'] = 1.0 / (1.0 + np.std(beat_intervals))
                features['rhythm_complexity'] = np.std(beat_intervals) / np.mean(beat_intervals)
            else:
                features['rhythm_regularity'] = 0
                features['rhythm_complexity'] = 0
            
            # Onset strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            features['onset_strength_mean'] = np.mean(onset_env)
            features['onset_strength_std'] = np.std(onset_env)
            
            # Peak analysis in onset strength
            peaks, _ = find_peaks(onset_env, height=np.mean(onset_env))
            features['onset_peaks_count'] = len(peaks)
            
        except Exception as e:
            print(f"Warning: Error extracting rhythm features: {str(e)}")
        
        return features
    
    @staticmethod
    def extract_energy_features(y, sr, frame_length=2048, hop_length=512):
        """Extract energy-based features"""
        
        features = {}
        
        try:
            # Short-time energy
            frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
            energy = np.sum(frames ** 2, axis=0)
            
            features['energy_mean'] = np.mean(energy)
            features['energy_std'] = np.std(energy)
            features['energy_max'] = np.max(energy)
            features['energy_min'] = np.min(energy)
            
            # Energy entropy
            energy_normalized = energy / np.sum(energy)
            features['energy_entropy'] = entropy(energy_normalized + 1e-12)
            
            # Low energy ratio
            energy_threshold = np.mean(energy) * 0.5
            low_energy_frames = np.sum(energy < energy_threshold)
            features['low_energy_ratio'] = low_energy_frames / len(energy)
            
            # Energy variation
            energy_diff = np.diff(energy)
            features['energy_variation'] = np.mean(np.abs(energy_diff))
            
        except Exception as e:
            print(f"Warning: Error extracting energy features: {str(e)}")
        
        return features
    
    @staticmethod
    def extract_silence_features(y, sr, top_db=20):
        """Extract silence and voice activity features"""
        
        features = {}
        
        try:
            # Split audio into non-silent intervals
            intervals = librosa.effects.split(y, top_db=top_db)
            
            if len(intervals) > 0:
                # Voice activity statistics
                total_samples = len(y)
                voice_samples = np.sum([end - start for start, end in intervals])
                
                features['voice_activity_ratio'] = voice_samples / total_samples
                features['silence_ratio'] = 1.0 - features['voice_activity_ratio']
                features['voice_segments_count'] = len(intervals)
                
                # Segment length statistics
                segment_lengths = [(end - start) / sr for start, end in intervals]
                features['voice_segment_mean_duration'] = np.mean(segment_lengths)
                features['voice_segment_std_duration'] = np.std(segment_lengths)
                
                # Pause analysis
                if len(intervals) > 1:
                    pause_durations = []
                    for i in range(len(intervals) - 1):
                        pause_start = intervals[i][1]
                        pause_end = intervals[i + 1][0]
                        pause_durations.append((pause_end - pause_start) / sr)
                    
                    features['pause_count'] = len(pause_durations)
                    features['pause_mean_duration'] = np.mean(pause_durations)
                    features['pause_std_duration'] = np.std(pause_durations)
                else:
                    features['pause_count'] = 0
                    features['pause_mean_duration'] = 0
                    features['pause_std_duration'] = 0
            else:
                # No voice activity detected
                features['voice_activity_ratio'] = 0
                features['silence_ratio'] = 1.0
                features['voice_segments_count'] = 0
                features['voice_segment_mean_duration'] = 0
                features['voice_segment_std_duration'] = 0
                features['pause_count'] = 0
                features['pause_mean_duration'] = 0
                features['pause_std_duration'] = 0
            
        except Exception as e:
            print(f"Warning: Error extracting silence features: {str(e)}")
        
        return features
    
    @staticmethod
    def extract_harmonic_features(y, sr):
        """Extract harmonic and percussive features"""
        
        features = {}
        
        try:
            # Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Harmonic features
            features['harmonic_energy'] = np.sum(y_harmonic ** 2)
            features['harmonic_rms'] = np.sqrt(np.mean(y_harmonic ** 2))
            
            # Percussive features
            features['percussive_energy'] = np.sum(y_percussive ** 2)
            features['percussive_rms'] = np.sqrt(np.mean(y_percussive ** 2))
            
            # Harmonic-to-percussive ratio
            total_energy = features['harmonic_energy'] + features['percussive_energy']
            if total_energy > 0:
                features['harmonic_percussive_ratio'] = features['harmonic_energy'] / total_energy
            else:
                features['harmonic_percussive_ratio'] = 0.5
            
            # Spectral analysis of harmonic component
            if np.sum(np.abs(y_harmonic)) > 0:
                harmonic_centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)
                features['harmonic_centroid_mean'] = np.mean(harmonic_centroid)
            else:
                features['harmonic_centroid_mean'] = 0
            
            # Spectral analysis of percussive component
            if np.sum(np.abs(y_percussive)) > 0:
                percussive_centroid = librosa.feature.spectral_centroid(y=y_percussive, sr=sr)
                features['percussive_centroid_mean'] = np.mean(percussive_centroid)
            else:
                features['percussive_centroid_mean'] = 0
            
        except Exception as e:
            print(f"Warning: Error extracting harmonic features: {str(e)}")
        
        return features
    
    @staticmethod
    def extract_temporal_features(y, sr, frame_length=2048, hop_length=512):
        """Extract temporal dynamics features"""
        
        features = {}
        
        try:
            # Autocorrelation
            autocorr = np.correlate(y, y, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find first minimum (indicates periodicity)
            first_min_idx = 1
            for i in range(1, len(autocorr) - 1):
                if autocorr[i] < autocorr[i-1] and autocorr[i] < autocorr[i+1]:
                    first_min_idx = i
                    break
            
            features['autocorr_first_min'] = first_min_idx / sr
            features['autocorr_first_min_value'] = autocorr[first_min_idx]
            
            # Temporal centroid
            frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
            frame_energies = np.sum(frames ** 2, axis=0)
            
            if np.sum(frame_energies) > 0:
                time_axis = np.arange(len(frame_energies)) * hop_length / sr
                temporal_centroid = np.sum(time_axis * frame_energies) / np.sum(frame_energies)
                features['temporal_centroid'] = temporal_centroid
            else:
                features['temporal_centroid'] = 0
            
            # Attack time (time to reach maximum energy)
            max_energy_idx = np.argmax(frame_energies)
            features['attack_time'] = max_energy_idx * hop_length / sr
            
            # Decay characteristics
            if max_energy_idx < len(frame_energies) - 1:
                decay_energies = frame_energies[max_energy_idx:]
                # Find where energy drops to 50% of maximum
                half_max = frame_energies[max_energy_idx] * 0.5
                decay_idx = np.where(decay_energies <= half_max)[0]
                
                if len(decay_idx) > 0:
                    features['decay_time'] = decay_idx[0] * hop_length / sr
                else:
                    features['decay_time'] = (len(decay_energies) - 1) * hop_length / sr
            else:
                features['decay_time'] = 0
            
        except Exception as e:
            print(f"Warning: Error extracting temporal features: {str(e)}")
        
        return features
    
    @staticmethod
    def extract_advanced_spectral_features(y, sr, hop_length=512):
        """Extract advanced spectral features"""
        
        features = {}
        
        try:
            # Spectral roll-off percentiles
            rolloff_25 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.25)[0]
            rolloff_75 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.75)[0]
            rolloff_95 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)[0]
            
            features['rolloff_25_mean'] = np.mean(rolloff_25)
            features['rolloff_75_mean'] = np.mean(rolloff_75)
            features['rolloff_95_mean'] = np.mean(rolloff_95)
            
            # Spectral slope
            stft = librosa.stft(y, hop_length=hop_length)
            magnitude = np.abs(stft)
            freqs = librosa.fft_frequencies(sr=sr)
            
            spectral_slopes = []
            for frame in magnitude.T:
                # Calculate slope of log magnitude spectrum
                log_magnitude = np.log(frame + 1e-8)
                slope = np.polyfit(freqs[:len(log_magnitude)], log_magnitude, 1)[0]
                spectral_slopes.append(slope)
            
            features['spectral_slope_mean'] = np.mean(spectral_slopes)
            features['spectral_slope_std'] = np.std(spectral_slopes)
            
            # Spectral crest factor
            crest_factors = []
            for frame in magnitude.T:
                if np.mean(frame) > 0:
                    crest_factor = np.max(frame) / np.mean(frame)
                    crest_factors.append(crest_factor)
            
            features['spectral_crest_mean'] = np.mean(crest_factors) if crest_factors else 0
            features['spectral_crest_std'] = np.std(crest_factors) if crest_factors else 0
            
            # Spectral flux variation
            spectral_flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
            features['flux_variation'] = np.std(spectral_flux) / (np.mean(spectral_flux) + 1e-8)
            
        except Exception as e:
            print(f"Warning: Error extracting advanced spectral features: {str(e)}")
        
        return features
    
    @staticmethod
    def extract_psychoacoustic_features(y, sr, hop_length=512):
        """Extract psychoacoustic features"""
        
        features = {}
        
        try:
            # Loudness estimation using A-weighting
            stft = librosa.stft(y, hop_length=hop_length)
            freqs = librosa.fft_frequencies(sr=sr)
            magnitude = np.abs(stft)
            
            # A-weighting approximation
            a_weight = 12200**2 * freqs**4 / ((freqs**2 + 20.6**2) * 
                      (freqs**2 + 12200**2) * np.sqrt((freqs**2 + 107.7**2) * 
                      (freqs**2 + 737.9**2)))
            a_weight = a_weight / np.max(a_weight)  # Normalize
            
            # Apply weighting
            weighted_magnitude = magnitude * a_weight.reshape(-1, 1)
            loudness = np.mean(weighted_magnitude, axis=0)
            
            features['loudness_mean'] = np.mean(loudness)
            features['loudness_std'] = np.std(loudness)
            features['loudness_range'] = np.max(loudness) - np.min(loudness)
            
            # Roughness estimation (amplitude modulation detection)
            hop_time = hop_length / sr
            mod_freqs = np.fft.fftfreq(len(loudness), hop_time)
            mod_spectrum = np.abs(np.fft.fft(loudness - np.mean(loudness)))
            
            # Roughness is related to modulation in 15-300 Hz range
            roughness_range = (mod_freqs >= 15) & (mod_freqs <= 300)
            features['roughness'] = np.sum(mod_spectrum[roughness_range])
            
            # Sharpness (high frequency content)
            nyquist = sr // 2
            high_freq_range = freqs > nyquist * 0.5
            high_freq_energy = np.sum(magnitude[high_freq_range], axis=0)
            total_energy = np.sum(magnitude, axis=0)
            
            sharpness = high_freq_energy / (total_energy + 1e-8)
            features['sharpness_mean'] = np.mean(sharpness)
            features['sharpness_std'] = np.std(sharpness)
            
        except Exception as e:
            print(f"Warning: Error extracting psychoacoustic features: {str(e)}")
        
        return features
    
    @staticmethod
    def extract_all_enhanced_features(y, sr, hop_length=512):
        """Extract all enhanced features in one call"""
        
        all_features = {}
        
        # Extract all feature categories
        all_features.update(FeatureExtractor.extract_rhythm_features(y, sr, hop_length))
        all_features.update(FeatureExtractor.extract_energy_features(y, sr, hop_length=hop_length))
        all_features.update(FeatureExtractor.extract_silence_features(y, sr))
        all_features.update(FeatureExtractor.extract_harmonic_features(y, sr))
        all_features.update(FeatureExtractor.extract_temporal_features(y, sr, hop_length=hop_length))
        all_features.update(FeatureExtractor.extract_advanced_spectral_features(y, sr, hop_length))
        all_features.update(FeatureExtractor.extract_psychoacoustic_features(y, sr, hop_length))
        
        return all_features
