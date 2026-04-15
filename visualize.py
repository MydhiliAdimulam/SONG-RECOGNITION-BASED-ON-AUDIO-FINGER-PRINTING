from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np

# Load the mp3 file
audio = AudioSegment.from_file("mp3/Brad-Sucks--Total-Breakdown.mp3")

# Convert the mp3 to a numpy array
data = np.array(audio.get_array_of_samples())

# normalize the data
data = data/np.max(np.abs(data))

# Calculate the FFT of the audio data
fft_data = np.fft.fft(data)

# Calculate the spectral peaks
spectral_peaks = np.argmax(np.abs(fft_data))

# Calculate the energy of the audio data
energy = np.sum(np.square(data))

# Plot the spectral peaks
plt.plot(spectral_peaks)
plt.title("Spectral Peaks")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

# Plot the energy
plt.figure()
plt.plot(energy)
plt.title("Energy")
plt.xlabel("Sample")
plt.ylabel("Energy")

plt.show()
