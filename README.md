# 🧠Tumor-from-EEG
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

This project detects brain tumors from EEG (electroencephalogram) data. The system utilizes various signal processing techniques to extract features from the EEG data, including wavelet-based denoising and bandpass filtering. These features are then analyzed with an Artificial Neural Network (ANN) classifier to determine brain tumors.

The graphical user interface (GUI) enables users to load EEG data, train a model, and classify EEG signals to detect brain tumors. The code incorporates various signal processing methods and machine learning techniques using scientific libraries within a user-friendly interface.
## ⚠️ Warning

**This software is not intended for medical diagnosis, treatment, or patient advice.** The code is designed for research purposes only, specifically for students and researchers to develop and contribute to brain tumor detection algorithms. 

**Important:** 
- The model is incomplete and has low accuracy for real-world medical diagnosis.
- Do not rely on this software for any clinical decisions or patient care.

Use it as a learning tool to explore EEG signal processing and machine learning applications in brain tumor research.
## Features

- **EEG Data Loading**: Import EEG signals from CSV files.
- **Bandpass Filter**: Apply a Butterworth bandpass filter to the EEG data.
- **Wavelet Denoising**: Reduce noise in the EEG signals using wavelet transforms.
- **Feature Extraction**: Extract statistical features from the EEG data using Stationary Wavelet Transform (SWT).
- **Model Training**: Train an ANN classifier with early stopping to prevent overfitting.
- **Tumor Detection**: Classify EEG signals to predict whether a brain tumor is present.
- **Data Visualization**: Plot both original and denoised EEG signals for comparison.
- **Interactive GUI**: User-friendly interface for data handling and model operations.

## Reference Article

**The concept behind this project was inspired by signal processing techniques studied in a Master's course on signal processing. Using wavelet transforms for EEG denoising and ANN classifiers was based on a combination of ideas from academic research articles.**
-  **Wavelet Transform for EEG Signal Denoising:**
Murugesan, M., & Sukanesh, R. (2009). Automated Detection of Brain Tumor in EEG Signals Using Artificial Neural Networks. 2009 International Conference on Advances in Computing, Control, and Telecommunication Technologies, Bangalore, India. https://doi.org/10.1109/ACT.2009.77

**This article explores the use of wavelet transforms to extract features from EEG signals and classify brain tumors using artificial neural networks (ANNs). This work inspires the wavelet-based denoising approach.**
- **EEG Signal Processing using Bandpass Filters:**
Karameh, F. N., & Dahleh, M. A. (2000). Automated classification of EEG signals in brain tumor diagnostics. Proceedings of the 2000 American Control Conference (IEEE), Chicago, IL, USA. https://doi.org/10.1109/ACC.2000.877006

**This article discusses the filtering of EEG signals to remove noise using bandpass filters, which is a key part of EEG preprocessing in the code. The bandpass filter implementation in this code is based on this concept.**
- **Preprocessing and Filtering in EEG Signals:**
Sharanreddy, M., & Kulkarni, P. K. (2013). Automated EEG signal analysis for identification of epilepsy seizures and brain tumor. Journal of Medical Engineering & Technology, 37(8), 511–519. https://doi.org/10.3109/03091902.2013.837530

**This article outlines several EEG preprocessing steps, including filtering and artifact removal, which inform the filtering and feature extraction methods used in the code.**
- **Machine Learning in EEG Classification:**
Amin, J., Sharif, M., Raza, M. et al. (2024). Detection of Brain Tumor based on Features Fusion and Machine Learning. Journal of Ambient Intelligence and Humanized Computing, 15, 983–999. https://doi.org/10.1007/s12652-018-1092-9

**This paper uses machine learning models such as Artificial Neural Networks (ANNs) for brain tumor classification from EEG data. The code's use of the MLPClassifier from scikit-learn is inspired by this article’s approach to classification.**

## AI Assistance

Several AI-based tools were utilized to complete and debug the code, including tools for automating model tuning, error detection, and optimizing signal processing functions.

## Screenshots

![Tumor from EEG GUI With Sample ECG](Tumor%20from%20Sample%20EEG.jpg).

## Technologies Used

- **Python Libraries**:
  - `numpy`, `pandas`: Data manipulation and handling.
  - `matplotlib`: Visualization of EEG signals.
  - `pywt`: Wavelet transforms for signal denoising and feature extraction.
  - `scipy`: Signal filtering using Butterworth bandpass filters.
  - `scikit-learn`: Machine learning (ANN classifier), feature scaling, and model evaluation.
  - `tkinter`: GUI framework for user interaction.

## Installation

- Install the required dependencies:
    ```bash
    pip install numpy pandas matplotlib pywt scipy scikit-learn tkinter
    ```
- or a Newer version of Python written in CMD
   ```bash
   py -m pip install numpy pandas matplotlib pywt scipy scikit-learn tkinter
   ```
## Usage Instructions

### 1. Loading EEG Data
Click the **Load EEG Data** button to select CSV files containing EEG signals. The CSV file should have rows representing time points and columns for EEG channels. It's better to use several EEG data for the best result. For memory limited, you can define the signal range (e.g., 0 to 3000).

### 2. Preprocessing Data
The system automatically applies a bandpass filter and wavelet denoising to clean the EEG signals. Use the GUI to define the range of EEG signal points to analyze.

### 3. Training the Model
After loading the data, click the **Train Model** button to train the ANN using the extracted features.

### 4. Tumor Detection
Once the model is trained, click **Detect Tumor** to classify EEG signals and detect tumor points.

### 5. Visualizing EEG Signals
Click **Plot EEG Data** to compare the original and denoised signals.

### 6. Resetting the Application
Use **Reset App** to clear the data and start a new analysis.

## License
- This project is licensed under the MIT License. See the LICENSE file for details.
- This project was inspired by tutorials on working with EEG data in Python. Thanks to the Python community and scientific libraries like `numpy`, `scipy`, `tkinter`, `pandas`, `matplotlib`, `pywt` and `scikit-learn` for making such projects possible.

## Contributing

I invite researchers, students, and developers to contribute to this project and help improve the basic framework for brain tumor detection using EEG signals. This code provides a foundation for signal processing and classification, but it is far from complete and currently offers limited accuracy.

### How You Can Contribute:
- **Improve the Model**: Enhance the accuracy of the current ANN model or experiment with other machine learning models.
- **Expand Feature Extraction**: Implement more advanced signal processing and feature extraction techniques to boost model performance.
- **Incorporate Real Data**: Use real-world EEG datasets to validate and refine the detection process.
- **Enhance GUI and Usability**: Improve the GUI to make it more user-friendly and accessible to non-technical users.
- **Multiclass Classification**: Add support for detecting multiple types of brain conditions beyond tumor vs. non-tumor classification.
- **Optimize Preprocessing**: Explore and optimize different filtering, denoising, or transformation methods to improve the signal quality before classification.

### How to Get Started:
1. Fork the repository.
2. Make your changes and improvements.
3. Submit a pull request with a detailed explanation of your contributions.

I appreciate all contributions and encourage collaborative development to advance this project further as a research tool. Let's work together to build a more robust and accurate system for brain tumor diagnosis based on EEG signals.
