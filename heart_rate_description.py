# Import all the libraries we need
import cv2                          # For webcam and face detection
import numpy as np                  # For math calculations
from scipy.signal import butter, filtfilt  # For filtering the signal
from scipy.fft import fft, fftfreq  # For finding heart rate frequency

# ─── This function cleans up the signal by removing unwanted noise ───
# It only keeps frequencies between 0.75-4 Hz (which is 45-240 BPM)
def bandpass_filter(signal, lowcut=0.75, highcut=4.0, fs=30, order=3):
    nyq = fs / 2                    # Nyquist frequency (half of sample rate)
    low, high = lowcut / nyq, highcut / nyq  # Normalize the frequencies
    b, a = butter(order, [low, high], btype='band')  # Create the filter
    return filtfilt(b, a, signal)   # Apply the filter and return clean signal

# ─── This function calculates the actual heart rate in BPM ───
# It uses FFT (Fast Fourier Transform) to find the dominant frequency
def get_heart_rate(signal, fs=30):
    n = len(signal)                 # Get length of signal
    freqs = fftfreq(n, d=1.0/fs)   # Calculate all frequencies
    fft_vals = np.abs(fft(signal))  # Get strength of each frequency
    
    # Only look at frequencies in the heartbeat range (45-240 BPM)
    valid = (freqs >= 0.75) & (freqs <= 4.0)
    if not np.any(valid):
        return 0                    # Return 0 if no valid frequency found
    
    # Find the strongest frequency and convert to BPM
    peak_freq = freqs[valid][np.argmax(fft_vals[valid])]
    return peak_freq * 60           # Convert Hz to BPM

# ─── Main function — this is where everything happens ───
def main():
    # Open the webcam (0 = default webcam)
    cap = cv2.VideoCapture(0)
    
    # Load the face detection model that comes with OpenCV
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    green_signal = []       # List to store green light values over time
    fps = 30                # Frames per second of webcam
    window_size = fps * 10  # Use 10 seconds of data for calculation
    bpm = 0                 # Starting BPM value

    print("📷 Starting... Hold still for 10 seconds")

    # Keep running until user presses Q
    while True:
        ret, frame = cap.read()  # Read one frame from webcam
        if not ret:
            break                # Stop if webcam fails

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        # If at least one face is detected
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Get position of first face

            # Crop just the forehead area (top 1/3 of face)
            # Forehead has the strongest pulse signal
            forehead = frame[y:y + h//3, x:x + w]

            # Get the average green color value from forehead
            # Green channel changes most with blood flow
            green_mean = np.mean(forehead[:, :, 1])
            green_signal.append(green_mean)  # Save the value

            # Draw green rectangle around the whole face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw smaller rectangle around forehead area
            cv2.rectangle(frame, (x, y), (x+w, y+h//3), (0, 200, 100), 2)
            
            # Label the forehead region
            cv2.putText(frame, "Forehead ROI", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1)

        # Once we have 10 seconds of data start calculating BPM
        if len(green_signal) >= window_size:
            signal_array = np.array(green_signal[-window_size:])
            signal_array -= np.mean(signal_array)  # Remove average to center signal
            filtered = bandpass_filter(signal_array, fs=fps)  # Clean the signal
            bpm = get_heart_rate(filtered, fs=fps)  # Calculate BPM
            green_signal = green_signal[-window_size:]  # Keep only recent data

        # Show BPM in green if normal, red if abnormal
        color = (0, 255, 0) if 50 < bpm < 180 else (0, 0, 255)
        
        # Display heart rate on screen
        cv2.putText(frame, f"Heart Rate: {bpm:.1f} BPM",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Show how many samples collected vs needed
        cv2.putText(frame, f"Samples: {len(green_signal)}/{window_size}",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Show the frame on screen
        cv2.imshow("rPPG Heart Rate Monitor", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()