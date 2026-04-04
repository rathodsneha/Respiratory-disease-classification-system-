import librosa
import numpy as np
import scipy.signal
from scipy import stats
import soundfile as sf
from python_speech_features import mfcc, logfbank, delta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from pydub import AudioSegment
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AudioProcessor:
    """Comprehensive audio processing for respiratory sounds"""
    
    def __init__(self, sample_rate=22050, n_mfcc=13, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def load_audio(self, file_path, target_sr=None):
        """
        Load audio file and return audio signal and sample rate
        
        Args:
            file_path (str): Path to audio file
            target_sr (int): Target sample rate for resampling
            
        Returns:
            tuple: (audio_signal, sample_rate)
        """
        try:
            if target_sr is None:
                target_sr = self.sample_rate
                
            # Load with librosa (handles most formats)
            audio, sr = librosa.load(file_path, sr=target_sr)
            
            return audio, sr
            
        except Exception as e:
            # Try with pydub for additional format support
            try:
                audio_segment = AudioSegment.from_file(file_path)
                audio_segment = audio_segment.set_frame_rate(target_sr).set_channels(1)
                audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                audio = audio / np.max(np.abs(audio))  # Normalize
                return audio, target_sr
            except:
                raise Exception(f"Could not load audio file: {file_path}. Error: {str(e)}")
    
    def preprocess_audio(self, audio, sr):
        """
        Preprocess audio signal with noise reduction and normalization
        
        Args:
            audio (np.array): Audio signal
            sr (int): Sample rate
            
        Returns:
            np.array: Preprocessed audio signal
        """
        # Remove DC component
        audio = audio - np.mean(audio)
        
        # Apply bandpass filter for respiratory sounds (100Hz - 2000Hz)
        nyquist = sr / 2
        low_freq = 100 / nyquist
        high_freq = 2000 / nyquist
        
        if high_freq < 1.0:  # Ensure frequency is valid
            b, a = scipy.signal.butter(4, [low_freq, high_freq], btype='band')
            audio = scipy.signal.filtfilt(b, a, audio)
        
        # Normalize amplitude
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Apply noise reduction using spectral subtraction
        audio = self._spectral_subtraction(audio, sr)
        
        return audio
    
    def _spectral_subtraction(self, audio, sr, alpha=2.0, beta=0.01):
        """
        Simple spectral subtraction for noise reduction
        
        Args:
            audio (np.array): Audio signal
            sr (int): Sample rate
            alpha (float): Over-subtraction factor
            beta (float): Spectral floor parameter
            
        Returns:
            np.array: Noise-reduced audio signal
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from first 0.5 seconds
        noise_frames = int(0.5 * sr / self.hop_length)
        noise_magnitude = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Spectral subtraction
        enhanced_magnitude = magnitude - alpha * noise_magnitude
        enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
        
        # Reconstruct signal
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
        
        return enhanced_audio
    
    def extract_features(self, audio, sr):
        """
        Extract comprehensive features from audio signal
        
        Args:
            audio (np.array): Audio signal
            sr (int): Sample rate
            
        Returns:
            dict: Dictionary containing all extracted features
        """
        features = {}
        
        # Basic audio properties
        features['duration'] = len(audio) / sr
        features['sample_rate'] = sr
        
        # Time-domain features
        features.update(self._extract_time_domain_features(audio))
        
        # Frequency-domain features
        features.update(self._extract_frequency_domain_features(audio, sr))
        
        # MFCC features
        features.update(self._extract_mfcc_features(audio, sr))
        
        # Spectral features
        features.update(self._extract_spectral_features(audio, sr))
        
        # Chroma features
        features.update(self._extract_chroma_features(audio, sr))
        
        # Additional respiratory-specific features
        features.update(self._extract_respiratory_features(audio, sr))
        
        return features
    
    def _extract_time_domain_features(self, audio):
        """Extract time-domain features"""
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(audio)
        features['std'] = np.std(audio)
        features['var'] = np.var(audio)
        features['rms'] = np.sqrt(np.mean(audio**2))
        
        # Amplitude features
        features['max_amplitude'] = np.max(np.abs(audio))
        features['min_amplitude'] = np.min(np.abs(audio))
        features['amplitude_range'] = features['max_amplitude'] - features['min_amplitude']
        
        # Zero crossing rate
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
        
        # Energy features
        frame_length = 2048
        energy = []
        for i in range(0, len(audio) - frame_length, frame_length // 2):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame**2))
        
        if energy:
            features['energy_mean'] = np.mean(energy)
            features['energy_std'] = np.std(energy)
            features['energy_max'] = np.max(energy)
            features['energy_min'] = np.min(energy)
        
        return features
    
    def _extract_frequency_domain_features(self, audio, sr):
        """Extract frequency-domain features"""
        features = {}
        
        # FFT
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(fft), 1/sr)[:len(fft)//2]
        
        # Spectral centroid
        features['spectral_centroid'] = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
        
        # Spectral bandwidth
        centroid = features['spectral_centroid']
        features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude)) if np.sum(magnitude) > 0 else 0
        
        # Spectral rolloff
        cumsum = np.cumsum(magnitude)
        rolloff_point = 0.85 * cumsum[-1]
        rolloff_idx = np.where(cumsum >= rolloff_point)[0]
        features['spectral_rolloff'] = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        
        # Spectral flatness
        geometric_mean = stats.gmean(magnitude + 1e-10)
        arithmetic_mean = np.mean(magnitude)
        features['spectral_flatness'] = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        
        return features
    
    def _extract_mfcc_features(self, audio, sr):
        """Extract MFCC features"""
        features = {}
        
        # Convert to 16-bit PCM format for python_speech_features
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # MFCC coefficients
        mfcc_features = mfcc(audio_int16, sr, numcep=self.n_mfcc, nfilt=26, nfft=self.n_fft)
        
        # Statistical measures of MFCCs
        features['mfcc_mean'] = np.mean(mfcc_features, axis=0).tolist()
        features['mfcc_std'] = np.std(mfcc_features, axis=0).tolist()
        features['mfcc_max'] = np.max(mfcc_features, axis=0).tolist()
        features['mfcc_min'] = np.min(mfcc_features, axis=0).tolist()
        
        # Delta and Delta-Delta features
        delta_mfcc = delta(mfcc_features, 2)
        delta_delta_mfcc = delta(delta_mfcc, 2)
        
        features['delta_mfcc_mean'] = np.mean(delta_mfcc, axis=0).tolist()
        features['delta_delta_mfcc_mean'] = np.mean(delta_delta_mfcc, axis=0).tolist()
        
        return features
    
    def _extract_spectral_features(self, audio, sr):
        """Extract spectral features using librosa"""
        features = {}
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1).tolist()
        features['spectral_contrast_std'] = np.std(spectral_contrast, axis=1).tolist()
        
        return features
    
    def _extract_chroma_features(self, audio, sr):
        """Extract chroma features"""
        features = {}
        
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
        features['chroma_std'] = np.std(chroma, axis=1).tolist()
        
        return features
    
    def _extract_respiratory_features(self, audio, sr):
        """Extract respiratory-specific features"""
        features = {}
        
        # Breathing rate estimation
        features['estimated_breathing_rate'] = self._estimate_breathing_rate(audio, sr)
        
        # Wheeze detection features
        features.update(self._detect_wheeze_features(audio, sr))
        
        # Crackle detection features
        features.update(self._detect_crackle_features(audio, sr))
        
        return features
    
    def _estimate_breathing_rate(self, audio, sr):
        """Estimate breathing rate from audio signal"""
        # Apply envelope detection
        envelope = np.abs(scipy.signal.hilbert(audio))
        
        # Low-pass filter to smooth envelope
        b, a = scipy.signal.butter(4, 2, btype='low', fs=sr)
        envelope = scipy.signal.filtfilt(b, a, envelope)
        
        # Find peaks (breathing cycles)
        peaks, _ = scipy.signal.find_peaks(envelope, height=np.max(envelope)*0.3, distance=sr//2)
        
        if len(peaks) > 1:
            # Calculate breathing rate in breaths per minute
            duration = len(audio) / sr
            breathing_rate = (len(peaks) - 1) * 60 / duration
            return min(breathing_rate, 60)  # Cap at reasonable maximum
        
        return 0
    
    def _detect_wheeze_features(self, audio, sr):
        """Detect wheeze-related features"""
        features = {}
        
        # High-frequency energy ratio (wheeze indicator)
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Define frequency bands
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        high_freq_mask = freqs > 400  # Wheeze typically above 400Hz
        low_freq_mask = freqs <= 400
        
        high_freq_energy = np.sum(magnitude[high_freq_mask, :], axis=0)
        low_freq_energy = np.sum(magnitude[low_freq_mask, :], axis=0)
        
        # High-frequency ratio
        hf_ratio = high_freq_energy / (low_freq_energy + 1e-10)
        features['wheeze_hf_ratio_mean'] = np.mean(hf_ratio)
        features['wheeze_hf_ratio_std'] = np.std(hf_ratio)
        features['wheeze_hf_ratio_max'] = np.max(hf_ratio)
        
        return features
    
    def _detect_crackle_features(self, audio, sr):
        """Detect crackle-related features"""
        features = {}
        
        # Crackles are characterized by short, explosive sounds
        # Use variance in short-time energy
        frame_length = int(0.01 * sr)  # 10ms frames
        energy_variance = []
        
        for i in range(0, len(audio) - frame_length, frame_length):
            frame = audio[i:i + frame_length]
            energy_variance.append(np.var(frame))
        
        if energy_variance:
            features['crackle_energy_variance_mean'] = np.mean(energy_variance)
            features['crackle_energy_variance_std'] = np.std(energy_variance)
            features['crackle_energy_variance_max'] = np.max(energy_variance)
        
        return features
    
    def generate_spectrogram(self, audio, sr, output_path=None):
        """
        Generate mel-spectrogram for CNN input
        
        Args:
            audio (np.array): Audio signal
            sr (int): Sample rate
            output_path (str): Path to save spectrogram image
            
        Returns:
            np.array: Mel-spectrogram
        """
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=128, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Save visualization if requested
        if output_path:
            plt.figure(figsize=(12, 8))
            librosa.display.specshow(
                log_mel_spec, 
                sr=sr, 
                hop_length=self.hop_length, 
                x_axis='time', 
                y_axis='mel'
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel-Spectrogram')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return log_mel_spec
    
    def segment_breathing_cycles(self, audio, sr):
        """
        Segment audio into individual breathing cycles
        
        Args:
            audio (np.array): Audio signal
            sr (int): Sample rate
            
        Returns:
            list: List of audio segments (breathing cycles)
        """
        # Apply envelope detection
        envelope = np.abs(scipy.signal.hilbert(audio))
        
        # Smooth envelope
        b, a = scipy.signal.butter(4, 2, btype='low', fs=sr)
        envelope = scipy.signal.filtfilt(b, a, envelope)
        
        # Find valleys (end of breathing cycles)
        valleys, _ = scipy.signal.find_peaks(-envelope, height=-np.max(envelope)*0.2, distance=sr//3)
        
        segments = []
        for i in range(len(valleys) - 1):
            start = valleys[i]
            end = valleys[i + 1]
            segment = audio[start:end]
            if len(segment) > sr * 0.5:  # Minimum 0.5 seconds
                segments.append(segment)
        
        return segments
    
    def assess_audio_quality(self, audio, sr):
        """
        Assess audio quality and provide recommendations
        
        Args:
            audio (np.array): Audio signal
            sr (int): Sample rate
            
        Returns:
            dict: Quality assessment results
        """
        quality = {
            'overall_quality': 'Good',
            'snr_db': 0,
            'clipping_detected': False,
            'noise_level': 'Low',
            'recommendations': []
        }
        
        # Signal-to-noise ratio estimation
        # Use first and last 0.5 seconds as noise estimate
        noise_duration = int(0.5 * sr)
        if len(audio) > 2 * noise_duration:
            noise_start = audio[:noise_duration]
            noise_end = audio[-noise_duration:]
            noise_power = np.mean([np.var(noise_start), np.var(noise_end)])
            signal_power = np.var(audio)
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                quality['snr_db'] = snr
                
                if snr < 10:
                    quality['noise_level'] = 'High'
                    quality['overall_quality'] = 'Poor'
                    quality['recommendations'].append('High noise level detected. Consider re-recording in quieter environment.')
                elif snr < 20:
                    quality['noise_level'] = 'Medium'
                    quality['overall_quality'] = 'Fair'
                    quality['recommendations'].append('Moderate noise level. Results may be affected.')
        
        # Clipping detection
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0.95:
            quality['clipping_detected'] = True
            quality['overall_quality'] = 'Poor'
            quality['recommendations'].append('Audio clipping detected. Reduce recording volume.')
        
        # Duration check
        duration = len(audio) / sr
        if duration < 10:
            quality['recommendations'].append('Recording is shorter than recommended 10 seconds.')
        elif duration > 60:
            quality['recommendations'].append('Recording is longer than recommended 60 seconds. Consider segmenting.')
        
        # Dynamic range check
        dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (np.std(audio) + 1e-10))
        if dynamic_range < 20:
            quality['recommendations'].append('Low dynamic range. Check microphone positioning.')
        
        return quality
    
    def save_features_to_file(self, features, output_path):
        """
        Save extracted features to JSON file
        
        Args:
            features (dict): Extracted features
            output_path (str): Output file path
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_features = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                serializable_features[key] = value.tolist()
            elif isinstance(value, np.floating):
                serializable_features[key] = float(value)
            elif isinstance(value, np.integer):
                serializable_features[key] = int(value)
            else:
                serializable_features[key] = value
        
        # Add metadata
        serializable_features['extraction_timestamp'] = datetime.now().isoformat()
        serializable_features['processor_version'] = '1.0.0'
        
        with open(output_path, 'w') as f:
            json.dump(serializable_features, f, indent=2)
    
    def process_audio_file(self, file_path, output_dir=None):
        """
        Complete audio processing pipeline
        
        Args:
            file_path (str): Path to audio file
            output_dir (str): Directory to save processed outputs
            
        Returns:
            dict: Processing results including features and quality assessment
        """
        try:
            # Load audio
            audio, sr = self.load_audio(file_path)
            
            # Preprocess
            processed_audio = self.preprocess_audio(audio, sr)
            
            # Extract features
            features = self.extract_features(processed_audio, sr)
            
            # Generate spectrogram
            spectrogram_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                spectrogram_path = os.path.join(output_dir, 'spectrogram.png')
                
            mel_spec = self.generate_spectrogram(processed_audio, sr, spectrogram_path)
            
            # Assess quality
            quality = self.assess_audio_quality(processed_audio, sr)
            
            # Save features if output directory provided
            if output_dir:
                features_path = os.path.join(output_dir, 'features.json')
                self.save_features_to_file(features, features_path)
            
            results = {
                'success': True,
                'audio_properties': {
                    'duration': len(processed_audio) / sr,
                    'sample_rate': sr,
                    'channels': 1,
                    'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
                },
                'features': features,
                'mel_spectrogram': mel_spec,
                'quality_assessment': quality,
                'spectrogram_path': spectrogram_path
            }
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'audio_properties': None,
                'features': None,
                'mel_spectrogram': None,
                'quality_assessment': None
            }