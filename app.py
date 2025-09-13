import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from audio_analyzer import AudioAnalyzer
from utils import get_supported_formats, format_file_size
import traceback

def main():
    st.set_page_config(
        page_title="Audio Analysis Tool",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Comprehensive Audio Analysis Tool")
    st.markdown("Upload multiple audio files to extract detailed acoustic features and generate CSV reports.")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = AudioAnalyzer()
    
    # Sidebar with information
    with st.sidebar:
        st.header("📋 Supported Features")
        st.markdown("""
        **Spectral Analysis:**
        - MFCCs, Chroma Features
        - Spectral Centroid, Bandwidth
        - Zero-crossing Rate
        - Tempo & Rhythm
        
        **Voice Quality:**
        - Pitch, Formants
        - Jitter, Shimmer
        - Harmonics-to-Noise Ratio
        
        **Signal Processing:**
        - Energy Analysis
        - FFT Spectrum
        - Log Filterbank Energies
        """)
        
        st.header("📁 Supported Formats")
        formats = get_supported_formats()
        for fmt in formats:
            st.markdown(f"• {fmt}")
    
    # File upload section
    st.header("📤 Upload Audio Files")
    uploaded_files = st.file_uploader(
        "Choose audio files",
        type=['wav', 'mp3', 'flac', 'm4a', 'aac', 'ogg'],
        accept_multiple_files=True,
        help="Upload one or more audio files for analysis"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
        
        # Display file information
        with st.expander("📊 File Information", expanded=True):
            file_data = []
            for file in uploaded_files:
                file_data.append({
                    "Filename": file.name,
                    "Size": format_file_size(file.size),
                    "Type": file.type or "Unknown"
                })
            st.dataframe(pd.DataFrame(file_data), use_container_width=True)
        
        # Analysis configuration
        st.header("⚙️ Analysis Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            sample_rate = st.selectbox(
                "Sample Rate (Hz)",
                [22050, 44100, 48000],
                index=0,
                help="Higher sample rates provide more detail but take longer to process"
            )
            
            hop_length = st.selectbox(
                "Hop Length",
                [512, 1024, 2048],
                index=1,
                help="Smaller hop lengths provide more temporal resolution"
            )
        
        with col2:
            n_mfcc = st.slider(
                "Number of MFCC coefficients",
                min_value=12,
                max_value=20,
                value=13,
                help="Number of Mel-frequency cepstral coefficients to extract"
            )
            
            extract_prosody = st.checkbox(
                "Extract Prosodic Features",
                value=True,
                help="Include pitch, jitter, shimmer, and HNR analysis"
            )
        
        # Analysis button
        if st.button("🔍 Start Analysis", type="primary", use_container_width=True):
            analyze_files(uploaded_files, sample_rate, hop_length, n_mfcc, extract_prosody)

def analyze_files(uploaded_files, sample_rate, hop_length, n_mfcc, extract_prosody):
    """Process uploaded files and perform audio analysis"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    total_files = len(uploaded_files)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (idx + 1) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name} ({idx + 1}/{total_files})")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Analyze audio file
            features = st.session_state.analyzer.analyze_file(
                tmp_file_path,
                sample_rate=sample_rate,
                hop_length=hop_length,
                n_mfcc=n_mfcc,
                extract_prosody=extract_prosody
            )
            
            # Add filename to features
            features['filename'] = uploaded_file.name
            results.append(features)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            st.code(traceback.format_exc())
            continue
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text("✅ Analysis complete!")
    
    if results:
        display_results(results)
    else:
        st.error("❌ No files were successfully processed.")

def display_results(results):
    """Display analysis results and provide download options"""
    
    st.header("📊 Analysis Results")
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Move filename to first column
    cols = ['filename'] + [col for col in df.columns if col != 'filename']
    df = df[cols]
    
    # Display summary statistics
    st.subheader("📈 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Files Processed", len(df))
    
    with col2:
        if 'duration' in df.columns:
            total_duration = df['duration'].sum()
            st.metric("Total Duration", f"{total_duration:.1f}s")
    
    with col3:
        mfcc_columns = [col for col in df.columns if 'mfcc_mean' in col]
        if mfcc_columns and len(mfcc_columns) > 0:
            mfcc_means = df[mfcc_columns].mean()
            avg_mfcc = mfcc_means.mean() if hasattr(mfcc_means, 'mean') else np.mean(mfcc_means)
            st.metric("Avg MFCC", f"{avg_mfcc:.3f}")
        else:
            st.metric("Avg MFCC", "N/A")
    
    with col4:
        if 'pitch_mean' in df.columns:
            avg_pitch = df['pitch_mean'].mean()
            st.metric("Avg Pitch", f"{avg_pitch:.1f} Hz")
    
    # Display detailed results
    st.subheader("🔍 Detailed Results")
    st.dataframe(df, use_container_width=True, height=400)
    
    # Feature visualization
    if len(df) > 1:
        st.subheader("📊 Feature Visualization")
        
        # Select numeric columns for visualization
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'filename' in numeric_cols:
            numeric_cols.remove('filename')
        
        if numeric_cols:
            selected_features = st.multiselect(
                "Select features to visualize",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            if selected_features:
                try:
                    chart_data = df[['filename'] + selected_features].copy()
                    chart_data = chart_data.set_index('filename')
                    st.line_chart(chart_data)
                except Exception as e:
                    st.error(f"Error creating chart: {str(e)}")
                    st.dataframe(df[['filename'] + selected_features])
    
    # Download section
    st.subheader("💾 Download Results")
    
    # Generate CSV
    csv_data = df.to_csv(index=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download CSV Report",
            data=csv_data,
            file_name="audio_analysis_report.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        # Generate summary report
        summary_data = generate_summary_report(df)
        st.download_button(
            label="📋 Download Summary Report",
            data=summary_data,
            file_name="audio_analysis_summary.txt",
            mime="text/plain",
            use_container_width=True
        )

def generate_summary_report(df):
    """Generate a text summary report"""
    
    report = []
    report.append("AUDIO ANALYSIS SUMMARY REPORT")
    report.append("=" * 40)
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total files analyzed: {len(df)}")
    report.append("")
    
    # Basic statistics
    report.append("BASIC STATISTICS:")
    if 'duration' in df.columns:
        report.append(f"Total duration: {df['duration'].sum():.2f} seconds")
        report.append(f"Average duration: {df['duration'].mean():.2f} seconds")
        report.append(f"Duration range: {df['duration'].min():.2f} - {df['duration'].max():.2f} seconds")
    
    # Feature statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        report.append("")
        report.append("FEATURE STATISTICS:")
        for col in numeric_cols[:10]:  # Show first 10 features
            if col != 'duration':
                mean_val = df[col].mean()
                std_val = df[col].std()
                report.append(f"{col}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    # File list
    report.append("")
    report.append("PROCESSED FILES:")
    for filename in df['filename']:
        report.append(f"- {filename}")
    
    return "\n".join(report)

if __name__ == "__main__":
    main()
