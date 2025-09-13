import os

def get_supported_formats():
    """Return list of supported audio formats"""
    return [
        "WAV - Waveform Audio File Format",
        "MP3 - MPEG Audio Layer III",
        "FLAC - Free Lossless Audio Codec",
        "M4A - MPEG-4 Audio",
        "AAC - Advanced Audio Coding",
        "OGG - Ogg Vorbis"
    ]

def format_file_size(size_bytes):
    """Convert file size in bytes to human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def validate_audio_file(file_path):
    """Validate if file is a supported audio format"""
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    file_extension = os.path.splitext(file_path)[1].lower()
    supported_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
    
    if file_extension not in supported_extensions:
        return False, f"Unsupported file format: {file_extension}"
    
    return True, "Valid audio file"

def create_feature_description():
    """Return detailed description of extracted features"""
    return {
        "Basic Features": [
            "duration - Audio file duration in seconds",
            "sample_rate - Audio sampling rate in Hz",
            "samples - Total number of audio samples",
            "channels - Number of audio channels"
        ],
        "Spectral Features": [
            "mfcc_mean/std_* - Mel-frequency cepstral coefficients statistics",
            "chroma_mean/std - Chroma features (pitch class profiles)",
            "spectral_centroid - Center of mass of the spectrum",
            "spectral_bandwidth - Spectral bandwidth",
            "spectral_rolloff - Frequency below which 85% of energy is concentrated",
            "spectral_contrast - Spectral valley-to-peak contrast",
            "spectral_flatness - Measure of spectral flatness (noisiness)"
        ],
        "Temporal Features": [
            "zcr_mean/std - Zero-crossing rate statistics",
            "rms_mean/std - Root mean square energy statistics",
            "tempo - Estimated tempo in BPM",
            "beats_count - Number of detected beats",
            "onset_count - Number of detected onsets"
        ],
        "Voice Quality Features": [
            "pitch_mean/std/min/max - Fundamental frequency statistics",
            "formant_*_mean/std - Formant frequency statistics",
            "jitter - Frequency perturbation",
            "shimmer - Amplitude perturbation",
            "hnr_mean/std - Harmonics-to-noise ratio statistics"
        ],
        "Energy Features": [
            "energy_mean/std/max/min - Energy statistics",
            "energy_entropy - Energy distribution entropy",
            "low_energy_ratio - Ratio of low-energy frames"
        ],
        "Rhythm Features": [
            "rhythm_regularity - Beat regularity measure",
            "rhythm_complexity - Rhythmic complexity measure",
            "beat_interval_mean/std - Beat interval statistics"
        ]
    }

def get_feature_categories():
    """Return feature categories for organizing results"""
    return {
        "basic": ["duration", "sample_rate", "samples", "channels"],
        "spectral": [
            "spectral_centroid_mean", "spectral_bandwidth_mean", 
            "spectral_rolloff_mean", "spectral_contrast_mean"
        ],
        "mfcc": [f"mfcc_mean_{i}" for i in range(1, 14)],
        "chroma": [f"chroma_{i}_mean" for i in range(1, 13)],
        "prosodic": [
            "pitch_mean", "pitch_std", "formant_1_mean", 
            "formant_2_mean", "jitter", "shimmer", "hnr_mean"
        ],
        "energy": ["rms_mean", "energy_mean", "dbfs"],
        "temporal": ["zcr_mean", "tempo", "onset_count"]
    }

def export_feature_documentation(output_path="feature_documentation.txt"):
    """Export detailed feature documentation to text file"""
    
    descriptions = create_feature_description()
    
    content = []
    content.append("AUDIO ANALYSIS FEATURE DOCUMENTATION")
    content.append("=" * 50)
    content.append("")
    
    for category, features in descriptions.items():
        content.append(f"{category}:")
        content.append("-" * len(category))
        for feature in features:
            content.append(f"  • {feature}")
        content.append("")
    
    content.append("NOTES:")
    content.append("- All spectral features are computed using Short-Time Fourier Transform")
    content.append("- MFCCs are computed on mel-scale filterbank")
    content.append("- Prosodic features require voice/speech content")
    content.append("- Statistics include mean, standard deviation, min, max where applicable")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))
    
    return output_path

def clean_feature_names(features_dict):
    """Clean and standardize feature names"""
    
    cleaned = {}
    
    for key, value in features_dict.items():
        # Remove special characters and normalize
        clean_key = key.replace(' ', '_').replace('-', '_').lower()
        cleaned[clean_key] = value
    
    return cleaned

def filter_features_by_category(features_dict, category):
    """Filter features by category"""
    
    categories = get_feature_categories()
    
    if category not in categories:
        return features_dict
    
    category_features = categories[category]
    filtered = {}
    
    for key, value in features_dict.items():
        if any(cat_feat in key for cat_feat in category_features):
            filtered[key] = value
    
    return filtered

def calculate_feature_statistics(df, feature_columns):
    """Calculate summary statistics for features across all files"""
    
    stats = {}
    
    for col in feature_columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            }
    
    return stats
