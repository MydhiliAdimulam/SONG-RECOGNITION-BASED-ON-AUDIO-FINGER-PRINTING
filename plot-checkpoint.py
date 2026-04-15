import librosa.display
import matplotlib.pyplot as plt
import numpy as np
def plots(filename):
# Load audio file
    path = "mp3/"
    y, sr = librosa.load(path+filename)

    # Extract Mel-Frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)

    # Extract Chroma feature
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    # Extract Spectral contrast feature
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    # Plot the MFCCs
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs'+filename)
    plt.tight_layout()


    # Plot the Chroma feature
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
    plt.colorbar()
    plt.title('Chroma'+filename)
    plt.tight_layout()


    # Plot the Spectral contrast feature
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectral_contrast, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('Spectral Contrast'+filename)
    plt.tight_layout()
    plt.show()