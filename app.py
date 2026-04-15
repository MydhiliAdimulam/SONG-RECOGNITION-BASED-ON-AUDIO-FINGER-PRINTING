import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
import tempfile
import re

# Set page configuration
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('Trained_model.keras')
        st.success("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

# Fixed preprocessing with correct shapes
def preprocess_audio(file_path, target_shape=(150, 150)):
    try:
        # Load audio with same parameters as training
        audio_data, sr = librosa.load(file_path, sr=22050, duration=30)
        
        # Mel spectrogram
        S = librosa.feature.melspectrogram(y=audio_data, sr=sr,
                                        n_mels=128,
                                        fmax=8000,
                                        n_fft=2048,
                                        hop_length=512)
        
        # Convert to dB
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Normalize
        S_normalized = (S_dB - np.min(S_dB)) / (np.max(S_dB) - np.min(S_dB))
        
        # Add channel dimension and resize
        S_expanded = np.expand_dims(S_normalized, axis=-1)  # Now (n_mels, time, 1)
        S_resized = resize(S_expanded, target_shape)  # (150, 150, 1)
        
        # Add batch dimension to make it (1, 150, 150, 1)
        final_input = np.expand_dims(S_resized, axis=0)
        
        if final_input.shape != (1, *target_shape, 1):
            st.error(f"Shape mismatch! Got {final_input.shape}, expected {(1, *target_shape, 1)}")
            return None
            
        return final_input
    
    except Exception as e:
        st.error(f"Audio processing failed: {str(e)}")
        return None

# Define the genre classes
genre_classes = [
    'blues', 'classical', 'country', 'jazz', 'hiphop',
    'disco', 'metal', 'pop', 'reggae', 'rock'
]

# Main content
st.title("🎵 Music Genre Classification")


# Function to extract genre from filename
def extract_genre_from_filename(filename):
    
    
    for genre in genre_classes:
        if filename.startswith(genre):
            return genre
    
    # If not found, check if genre is part of the filename
    for genre in genre_classes:
        if genre in filename:
            return genre
    
    return "Unknown"

# Function to extract mel spectrogram
def extract_mel_spectrogram(file_path, target_shape=(128, 128)):
    """Extract Mel-Spectrogram from audio file and resize to target shape."""
    # Load audio (resampled to 22.05kHz, mono)
    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30.0)
    
    # Extract Mel-Spectrogram (128 frequency bins)
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr,
        n_mels=128,  # Number of Mel bands
        fmax=8000    # Maximum frequency
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Resize to target shape if needed
    if mel_spec_db.shape[1] != target_shape[1]:
        # Either pad or crop time dimension to match target
        if mel_spec_db.shape[1] < target_shape[1]:
            # Pad if too short
            pad_width = target_shape[1] - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Crop if too long
            mel_spec_db = mel_spec_db[:, :target_shape[1]]
    
    # Normalize to [0, 1] range
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    # Reshape for model input (adding channel dimension)
    return mel_spec_db.reshape((*target_shape, 1))

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name
    
    # Get the original filename
    original_filename = uploaded_file.name
    
    # Extract genre from original filename
    genre = extract_genre_from_filename(original_filename)
    
    # Display audio file info
   # st.subheader("Audio File Information")
   
    st.subheader(f"Detected Genre: {genre}")
    
    # Process the audio file
    try:
        # Load and display audio waveform
        y, sr = librosa.load(temp_file_path, sr=22050, duration=30.0)
        st.subheader("Audio Waveform")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.linspace(0, len(y)/sr, len(y)), y)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
        
        # Extract and display the spectrogram
        spectrogram = extract_mel_spectrogram(temp_file_path)
        st.subheader("Mel-Spectrogram")
        fig, ax = plt.subplots(figsize=(10, 4))
        img = ax.imshow(spectrogram[:,:,0], aspect='auto', origin='lower', cmap='viridis')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Mel-Spectrogram')
        st.pyplot(fig)
        
        # Audio playback
        st.subheader("Audio Preview")
        st.audio(temp_file_path, format='audio/wav')
        
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
    
    # Clean up the temporary file
    os.unlink(temp_file_path)
else:
    st.info("Please upload a .wav file to analyze its genre.")


# Display supported genres
st.sidebar.title("Supported Genres")
st.sidebar.write(", ".join(genre_classes))