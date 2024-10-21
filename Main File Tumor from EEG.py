import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Bandpass Filter (Butterworth)
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Function to denoise EEG signal using wavelet transform
def denoise_signal(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)  # Decompose signal
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745       # Estimate noise standard deviation from detail coefficients
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))  # Universal threshold
    
    # Apply soft thresholding to detail coefficients
    denoised_coeffs = [coeffs[0]]  # Approximation coefficients remain unchanged
    for detail_coeff in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(detail_coeff, threshold, mode='soft'))

    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)  # Reconstruct the signal
    return denoised_signal

# Feature extraction 
def extract_features(signal):
    # Denoise the signal before extracting features
    denoised_signal = denoise_signal(signal)
    
    coeffs = pywt.swt(denoised_signal, 'db4', level=3)  # Use SWT for feature extraction
    features = []
    
    for (cA, cD) in coeffs:
        # Compute statistical features for approximation and detail coefficients
        features.append(np.mean(cA))  # Mean of approximation
        features.append(np.var(cA))   # Variance of approximation
        features.append(np.mean(cD))  # Mean of detail
        features.append(np.var(cD))   # Variance of detail

    return features

# GUI 
class TumorDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠Tumor-from-EEG - Copyright 2024 | github.com/soheilkooklan")
        self.root.geometry('900x750')
        self.root.configure(bg='darkblue')  
        
        self.eeg_data_list = []
        self.features = None
        self.labels = None
        self.model = None

        # Default range for signal points (can be customized by the user)
        self.start_point = 0
        self.end_point = 3000

        # Sampling frequency and filter settings
        self.fs = 250  # Sampling frequency in Hz (adjustable)
        self.lowcut = 1.0   # Low cutoff frequency (Hz)
        self.highcut = 30.0  # High cutoff frequency (Hz)

        self.setup_gui()

    def setup_gui(self):
        # Define GUI layout
        self.button_frame = tk.Frame(self.root, bg='darkblue')  
        self.button_frame.pack(pady=20)

        # Load EEG Data button
        self.load_button = tk.Button(self.button_frame, text="Load EEG Data", command=self.load_eeg_data, bg="lightblue", font=("Arial", 12, "bold"), padx=10, pady=5)
        self.load_button.grid(row=0, column=0, padx=10)

        # Train Model button
        self.train_button = tk.Button(self.button_frame, text="Train Model", command=self.train_model, bg="lightblue", font=("Arial", 12, "bold"), padx=10, pady=5)
        self.train_button.grid(row=0, column=1, padx=10)

        # Detect Tumor button
        self.detect_button = tk.Button(self.button_frame, text="Detect Tumor", command=self.detect_tumor, bg="lightblue", font=("Arial", 12, "bold"), padx=10, pady=5)
        self.detect_button.grid(row=0, column=2, padx=10)

        # Plot EEG Data button
        self.plot_button = tk.Button(self.button_frame, text="Plot EEG Data", command=self.plot_eeg_data, bg="lightblue", font=("Arial", 12, "bold"), padx=10, pady=5)
        self.plot_button.grid(row=0, column=3, padx=10)

        # Reset Application button (Simulated Reset)
        self.reset_button = tk.Button(self.button_frame, text="Reset App", command=self.reset_app, bg="lightblue", font=("Arial", 12, "bold"), padx=10, pady=5)
        self.reset_button.grid(row=0, column=4, padx=10)

        # Help button to show tips and instructions
        self.help_button = tk.Button(self.button_frame, text="Help", command=self.show_help, bg="lightblue", font=("Arial", 12, "bold"), padx=10, pady=5)
        self.help_button.grid(row=0, column=5, padx=10)

        # Textbox to display results
        self.result_box = tk.Text(self.root, height=10, width=100, bg='white', font=("Arial", 12))
        self.result_box.pack(pady=20, padx=10)

        # User input for selecting range of points to analyze
        self.range_frame = tk.Frame(self.root, bg='darkblue')
        self.range_frame.pack(pady=20)

        # Start point label and entry
        tk.Label(self.range_frame, text="Start Point", font=("Arial", 12, "bold"), bg='darkblue', fg='white').grid(row=0, column=0, padx=5)
        self.start_point_entry = tk.Entry(self.range_frame, width=10, font=("Arial", 12))
        self.start_point_entry.grid(row=0, column=1, padx=5)
        self.start_point_entry.insert(0, '0')  # Default start point is 0

        # End point label and entry
        tk.Label(self.range_frame, text="End Point", font=("Arial", 12, "bold"), bg='darkblue', fg='white').grid(row=0, column=2, padx=5)
        self.end_point_entry = tk.Entry(self.range_frame, width=10, font=("Arial", 12))
        self.end_point_entry.grid(row=0, column=3, padx=5)
        self.end_point_entry.insert(0, '3000')  # Default end point is 3000

    def load_eeg_data(self):
        # Load EEG data 
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if file_paths:
            for file_path in file_paths:
                eeg_data = pd.read_csv(file_path)
                self.eeg_data_list.append(eeg_data)
            messagebox.showinfo("Success", "EEG data loaded successfully.")
        else:
            messagebox.showwarning("Error", "No files selected.")

    def preprocess_data(self):
        # Preprocess the EEG data (denoising, filtering, etc.)
        try:
            # Get user-defined range of points for analysis
            self.start_point = int(self.start_point_entry.get())
            self.end_point = int(self.end_point_entry.get())

            if self.start_point >= self.end_point:
                raise ValueError("Start point must be less than end point")

            if not self.eeg_data_list:
                raise ValueError("Please load EEG data first.")

            all_features = []
            for eeg_data in self.eeg_data_list:
                signal_columns = eeg_data.iloc[:, 1:]
                for col in signal_columns.columns:
                    signal = pd.to_numeric(signal_columns[col], errors='coerce').dropna().values.astype(float)
                    signal = signal[self.start_point:self.end_point]  # Use user-defined range of points
                    
                    # Apply bandpass filter (denoising with wavelet will happen later)
                    filtered_signal = apply_bandpass_filter(signal, self.lowcut, self.highcut, self.fs)
                    
                    features = extract_features(filtered_signal)  # Extract features using SWT after filtering
                    all_features.append(features)

            self.features = np.array(all_features)
            self.labels = np.random.randint(0, 2, size=self.features.shape[0])  # Dummy labels (0 for normal, 1 for tumor)

            # Feature scaling
            scaler = StandardScaler()
            self.features = scaler.fit_transform(self.features)

        except Exception as e:
            messagebox.showwarning("Error", f"Preprocessing error: {str(e)}")

    def train_model(self):
        # Train the ANN model with early stopping
        self.preprocess_data()
        if self.features is not None:
            try:
                X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.3, random_state=42)

                # ANN with early stopping to prevent overtraining
                self.model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42, early_stopping=True)
                self.model.fit(X_train, y_train)

                # Evaluate the model
                y_pred = self.model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                # Add the zero_division parameter to handle precision issues
                report = classification_report(y_test, y_pred, zero_division=1)

                self.result_box.insert(tk.END, f"Model Accuracy: {acc * 100:.2f}%\n")
                self.result_box.insert(tk.END, f"Classification Report:\n{report}\n")
            except Exception as e:
                messagebox.showwarning("Error", f"Model training error: {str(e)}")
        else:
            messagebox.showwarning("Error", "Please preprocess EEG data first.")

    def detect_tumor(self):
        # Use the trained model to detect brain tumors and identify specific points
        if self.model is not None and self.features is not None:
            predictions = self.model.predict(self.features)
            
            tumor_points = np.where(predictions == 1)[0]  # Get the points where a tumor is detected
            tumor_count = len(tumor_points)

            if tumor_count > 0:
                tumor_message = f"Tumor detected at the following points: {tumor_points}\n"
            else:
                tumor_message = "No tumor detected.\n"
            
            self.result_box.insert(tk.END, f"{tumor_message}Total Tumor Samples Detected: {tumor_count}\n")
        else:
            messagebox.showwarning("Error", "Please train the model first.")

    def plot_eeg_data(self):
        # Plot both original and denoised EEG data
        if self.eeg_data_list:
            for eeg_data in self.eeg_data_list:
                try:
                    time = np.arange(eeg_data.shape[0])  # Time series data
                    fig, ax = plt.subplots(figsize=(10, 6))

                    for col in eeg_data.columns[1:]:
                        signal = pd.to_numeric(eeg_data[col], errors='coerce').dropna().values.astype(float)
                        original_signal = signal[self.start_point:self.end_point]  # Original signal
                        denoised_signal = denoise_signal(original_signal)  # Denoised signal

                        # Plot original signal
                        ax.plot(time[:len(original_signal)], original_signal, label=f'Original {col}', alpha=0.6)

                        # Plot denoised signal
                        ax.plot(time[:len(denoised_signal)], denoised_signal, label=f'Denoised {col}', linestyle='dashed')

                    ax.set_xlabel("Time (samples)")
                    ax.set_ylabel("Amplitude")
                    ax.set_title("Original vs. Denoised EEG Signals")
                    ax.legend()
                    plt.show()

                except Exception as e:
                    messagebox.showwarning("Error", f"Plotting error: {str(e)}")
        else:
            messagebox.showwarning("Error", "Please load EEG data first.")

    def reset_app(self):
        """Simulated reset of the application."""
        self.eeg_data_list = []
        self.features = None
        self.labels = None
        self.model = None
        self.result_box.delete(1.0, tk.END)  # Clear the result box
        messagebox.showinfo("Reset", "Application has been reset. Load new EEG data to continue.")

    def show_help(self):
        """Display help and tips for using the application."""
        help_text = (
            "Tips for Using the Brain Tumor Detection App:\n\n"
            "1. Load EEG Data:\n"
            "   - Click the 'Load EEG Data' button to import your EEG data.\n"
            "   - The data should be in CSV format, with columns representing the EEG channels.\n\n"
            "2. Set EEG Signal Range:\n"
            "   - Use the 'Start Point' and 'End Point' fields to specify the range of EEG signal points you want to analyze.\n"
            "   - The default range is from 0 to 3000, but you can adjust this to focus on a different part of the signal.\n\n"
            "3. Train Model:\n"
            "   - After loading the data and specifying the range, click 'Train Model' to train the system using an Artificial Neural Network (ANN).\n"
            "   - The model is trained using features extracted from the EEG signals (via Stationary Wavelet Transform).\n\n"
            "4. Detect Tumor:\n"
            "   - Once the model is trained, click 'Detect Tumor' to classify the EEG data.\n"
            "   - The system will predict if a tumor is present based on the EEG signals and report the detected points.\n\n"
            "5. Plot EEG Data:\n"
            "   - Click 'Plot EEG Data' to visualize the loaded EEG signals over time.\n\n"
            "6. Reset App:\n"
            "   - Click 'Reset App' to clear the current data and model, and reset the application.\n"
            "   - For more information or troubleshooting, please go to https://github.com/soheilkooklan/Tumor-from-EEG"
        )
        messagebox.showinfo("Help", help_text)

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = TumorDetectionApp(root)
    root.mainloop()