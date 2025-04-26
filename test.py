from flask import Flask, request, render_template, send_from_directory, session, send_file, redirect, url_for # type: ignore
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import librosa.display
import io
import base64
from scipy.io import wavfile
import soundfile as sf
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['IMAGE_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/images')


# 确保上传文件夹和图片文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)

# 设置一个安全密钥，启用 session
app.secret_key = 'your_secret_key'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('results.html', error='No file part')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('results.html', error='No selected file')
    
    if file:
        # 保存上传的音频文件
        original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(original_file_path)
        
        # 判断文件格式是否为 m4a，如果是则转换为 wav
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext == '.m4a':
            try:
                converted_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(file.filename)[0]}.wav")
                # 使用 ffmpeg 转换 m4a 为 wav
                subprocess.run(['ffmpeg', '-i', original_file_path, converted_file_path], check=True)
                file_path = converted_file_path
            except subprocess.CalledProcessError as e:
                return render_template('results.html', error=f"Error converting m4a to wav: {str(e)}")
        else:
            file_path = original_file_path
        
        try:
            # 加载音频文件
            y, sr = librosa.load(file_path, sr=None)
            
            # 基本音频特征提取
            duration = librosa.get_duration(y=y, sr=sr)  # 音频时长
            rms = librosa.feature.rms(y=y).mean()  # 均方根能量
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y).mean()  # 过零率
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()  # 频谱质心
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()  # 频谱带宽
            
            # 新增音频特征提取
            pitches, _ = librosa.piptrack(y=y, sr=sr)  # 音调
            pitch = np.max(pitches) if pitches.size > 0 else 0
            hnr = librosa.effects.harmonic(y=y).mean()  # 谐波频谱比 (HNR)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)  # 梅尔频率倒谱系数 (MFCCs)
            
            # 生成波形图
            plt.figure(figsize=(14, 5), dpi=100)
            librosa.display.waveshow(y, sr=sr)
            waveform_image_path = os.path.join(app.config['IMAGE_FOLDER'], 'waveform.png')
            plt.savefig(waveform_image_path)
            plt.close()
            
            # 生成频谱图
            X = librosa.stft(y)
            Xdb = librosa.amplitude_to_db(abs(X))
            plt.figure(figsize=(14, 5), dpi=100)
            librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar()
            spectrogram_image_path = os.path.join(app.config['IMAGE_FOLDER'], 'spectrogram.png')
            plt.savefig(spectrogram_image_path)
            plt.close()
            
            # 快速傅里叶变换 (FFT)
            fft = np.abs(np.fft.fft(y))
            fft_freq = np.fft.fftfreq(len(fft), 1/sr)
            plt.figure(figsize=(14, 5))
            plt.plot(fft_freq[:len(fft)//2], fft[:len(fft)//2])
            plt.title('FFT Magnitude')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            fft_image_path = os.path.join(app.config['IMAGE_FOLDER'], 'fft.png')
            plt.savefig(fft_image_path)
            plt.close()
            
            # 生成基次谐波的时域图像
            fundamental_frequency = fft_freq[np.argmax(fft[:len(fft)//2])]
            t = np.linspace(0, duration, len(y))
            fundamental_wave = np.sin(2 * np.pi * fundamental_frequency * t)
            plt.figure(figsize=(14, 5))
            plt.plot(t[:len(y)], fundamental_wave[:len(y)], lw=1.5, alpha=0.8, label='Fundamental Harmonic')
            plt.title('Time Domain Fundamental Harmonic')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            fundamental_image_path = os.path.join(app.config['IMAGE_FOLDER'], 'fundamental.png')
            plt.savefig(fundamental_image_path)
            plt.close()
            
            # 计算短时傅里叶变换 (STFT)
            stft = np.abs(librosa.stft(y))
            
            # 计算功率谱 (Power Spectrogram)
            power_spectrogram = stft**2
            
            # 计算频率轴上的平均能量分布（能量谱）
            frequencies = np.linspace(0, sr / 2, power_spectrogram.shape[0])
            power_spectrum = np.mean(power_spectrogram, axis=1)
            
            # 对数变换，确保不会对零或负值取对数
            valid_indices = np.where((power_spectrum > 0) & (frequencies > 0))
            log_frequencies = np.log(frequencies[valid_indices])
            log_power = np.log(power_spectrum[valid_indices])
            
            # 二次拟合，计算二次方程的系数
            if len(log_frequencies) > 0 and len(log_power) > 0:
                coefficients = np.polyfit(log_frequencies, log_power, 2)
                a, b, c = coefficients
            else:
                a, b, c = 0, 0, 0


            # 生成奇次谐波单独图像
            harmonics = [3, 5, 7, 9, 11]  # 前5个奇次谐波
            colors = ['b', 'g', 'orange', 'purple', 'cyan']  # 为每个谐波指定不同的颜色
            plt.figure(figsize=(14, 10))
            for i, harmonic in enumerate(harmonics):
                harmonic_fft = np.abs(np.fft.fft(y * np.sin(2 * np.pi * harmonic * fundamental_frequency * t)))
                plt.plot(fft_freq[:len(fft)//2], harmonic_fft[:len(fft)//2], lw=1.5, alpha=0.8, color=colors[i], label=f'{harmonic}th Harmonic')
            plt.title('FFT 5 Odd Harmonics')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.legend()
            individual_harmonics_image_path = os.path.join(app.config['IMAGE_FOLDER'], 'individual_harmonics.png')
            plt.savefig(individual_harmonics_image_path)
            plt.close()

            # 生成奇次谐波和叠加图像
            plt.figure(figsize=(14, 5))
            summed_harmonics = np.zeros(len(fft)//2)
            for i, harmonic in enumerate(harmonics):
                harmonic_fft = np.abs(np.fft.fft(y * np.sin(2 * np.pi * harmonic * fundamental_frequency * t)))
                summed_harmonics += harmonic_fft[:len(fft)//2]
            plt.plot(fft_freq[:len(fft)//2], summed_harmonics, lw=2, color='r', label='Summed Harmonics')
            plt.title('FFT Summed Harmonics')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.legend()
            harmonics_image_path = os.path.join(app.config['IMAGE_FOLDER'], 'harmonics.png')
            plt.savefig(harmonics_image_path)
            plt.close()
            
            return render_template('results.html', images=['/static/images/waveform.png', '/static/images/spectrogram.png', '/static/images/fft.png', '/static/images/individual_harmonics.png', '/static/images/harmonics.png','/static/images/fundamental.png'], coefficients={'a': a, 'b': b, 'c': c}, audio_features={
                                       'pitch': pitch,
                                       'pitch_description': '音调表示信号中的主要频率成分，通常用于分析音高和音乐特征。' if session.get('language', '中文') == '中文' else 'Pitch represents the dominant frequency component in the signal, often used to analyze musical and tonal characteristics.',
                                       'duration': duration,
                                       'rms': rms,
                                       'zero_crossing_rate': zero_crossing_rate,
                                       'spectral_centroid': spectral_centroid,
                                       'spectral_bandwidth': spectral_bandwidth,
                                       'hnr': hnr,
                                       'mfcc': mfccs.tolist(),
                                       'filename': file.filename
                                   },
                                   language=session.get('language', '中文'))
        
        except Exception as e:
            return render_template('results.html', error=str(e))

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(os.path.abspath(app.config['IMAGE_FOLDER']), filename)

@app.route('/audio/<filename>')
def audio_file(filename):
    return send_from_directory(os.path.abspath(app.config['UPLOAD_FOLDER']), filename)

@app.route('/set_language', methods=['POST'])
def set_language():
    language = request.form.get('language', '中文')  # 默认为中文
    session['language'] = language
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)