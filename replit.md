# Audio Analysis Tool

## Overview

This is a comprehensive audio analysis tool built with Streamlit that extracts detailed acoustic features from multiple audio file formats. The application provides a web-based interface for uploading audio files and generating CSV reports with extensive feature analysis including spectral analysis, voice quality metrics, and signal processing characteristics. It leverages multiple specialized audio processing libraries to perform deep acoustic analysis suitable for research, music analysis, or audio classification tasks.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Interface**: Single-page application with file upload capabilities and real-time analysis display
- **Layout Design**: Wide layout with sidebar for feature information and supported formats
- **Session State Management**: Maintains AudioAnalyzer instance across user interactions
- **Progressive Disclosure**: Sidebar provides detailed feature descriptions and supported format information

### Backend Architecture
- **Modular Component Design**: Separate classes for different analysis aspects
  - `AudioAnalyzer`: Main analysis orchestrator handling multiple audio processing libraries
  - `FeatureExtractor`: Specialized rhythm and energy feature extraction utilities
  - `utils`: Helper functions for file validation and formatting
- **Multi-Library Integration**: Combines librosa, parselmouth, pydub, scipy, and soundfile for comprehensive analysis
- **Error Handling**: Graceful degradation when specific analysis components fail

### Data Processing Pipeline
- **File Format Support**: Handles WAV, MP3, FLAC, M4A, AAC, and OGG formats through format-specific loaders
- **Feature Categories**:
  - Spectral Analysis: MFCCs, chroma features, spectral centroid/bandwidth, zero-crossing rate
  - Voice Quality: Pitch, formants, jitter, shimmer, harmonics-to-noise ratio
  - Signal Processing: Energy analysis, FFT spectrum, log filterbank energies
  - Rhythm Analysis: Tempo, beat tracking, onset detection
- **Output Format**: CSV generation for batch analysis results

### Audio Processing Strategy
- **Standardized Preprocessing**: Default 22050 Hz sample rate with configurable hop length (1024 samples)
- **Multi-Domain Analysis**: Time-domain, frequency-domain, and cepstral-domain feature extraction
- **Statistical Aggregation**: Mean, standard deviation, skewness, and kurtosis for time-varying features
- **Prosodic Analysis**: Optional Parselmouth integration for detailed voice quality metrics

## External Dependencies

### Core Audio Processing Libraries
- **librosa**: Primary library for music and audio analysis, spectral feature extraction
- **parselmouth**: Python wrapper for Praat speech analysis toolkit, prosodic feature extraction
- **pydub**: Audio file format handling and basic audio manipulation
- **soundfile**: High-quality audio file I/O operations
- **scipy**: Signal processing functions, statistical analysis, and peak detection

### Web Framework and Data Handling
- **streamlit**: Web application framework for the user interface
- **pandas**: Data manipulation and CSV export functionality
- **numpy**: Numerical computing foundation for all audio processing operations

### System Dependencies
- **librosa backend requirements**: FFmpeg or similar audio codec libraries for format support
- **Praat integration**: System-level Praat installation may be required for advanced prosodic analysis
- **Audio codec support**: Platform-specific audio decoders for MP3, AAC, and other compressed formats