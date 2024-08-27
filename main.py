import os
import numpy as np
import matplotlib.pyplot as plt

def read_iq_file_in_chunks(filename, dtype=np.int16, chunk_size=2048 * 200):
    sample_size = np.dtype(dtype).itemsize
    read_len = chunk_size * sample_size * 2
    with open(filename, 'rb') as f:
        while True:
            data = f.read(read_len)  # 每个样本占 4 字节 (2 字节 I + 2 字节 Q)
            if not data:
                break
            iq_data = np.frombuffer(data, dtype=np.int16)
            yield iq_data


def compute_spectrum(iq_data, fs, bandwidth, fft_size=2048):
    # magnitude_spectrum_list = []
    # step = fs // fft_size
    samples = iq_data[::2] + 1j * iq_data[1::2]
    segments = range(len(samples) // fft_size)
    for i in segments:
        fft_window = np.hanning(fft_size)
        windowed_data = samples[fft_size * i:fft_size * (i + 1)] * fft_window
        # fft_result = np.fft.fft(windowed_data, n=fft_size)
        # 做完 fft 后将 oHz 频率挪至中心
        fft_result = np.fft.fftshift(np.fft.fft(windowed_data, n=fft_size))
        fft_freq = np.fft.fftshift(np.fft.fftfreq(fft_size, 1 / fs))
        # 截取真实带宽频域数据
        center_idx = fft_size // 2
        half_bw_idx = int(bandwidth / 2 * fft_size / fs)
        start_idx = center_idx - half_bw_idx
        end_idx = center_idx + half_bw_idx
        # 频域截取
        signal_fft = np.zeros_like(fft_result)
        signal_fft[start_idx:end_idx] = fft_result[start_idx:end_idx]
        # 将复数转为实数
        # magnitude_spectrum = np.abs(signal_fft)
        magnitude_spectrum = np.abs(fft_result)

        # 画图测试
        # 计算功率
        PSD = magnitude_spectrum ** 2 / (fft_size * fs)
        PSD_log = 10.0 * np.log10(PSD)
        # 绘制 x 频率轴
        # f = np.arange(fs / -2.0, fs / 2.0, fs / fft_size)  # 起始值，结束值，步长
        # 画图
        plt.plot(fft_freq, PSD_log)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        plt.grid(True)
        plt.show()
        print("success")

        # 转换回时域
        extracted_signal_time = np.fft.ifft(np.fft.ifftshift(signal_fft))
        real_part = np.real(extracted_signal_time)
        imag_part = np.imag(extracted_signal_time)
        # 交错合并为实数数组
        iq_real = np.empty(extracted_signal_time.size * 2, dtype=np.float32)
        iq_real[0::2] = real_part
        iq_real[1::2] = imag_part
        iq_data = np.real(iq_real) # 实数 IQ 数据
        # iq_list.append()
        # magnitude_spectrum_list.append(magnitude_spectrum)
    # return step, magnitude_spectrum_list


def main():
    fft_size = 2048
    dtype = np.int16
    file = r'F:\SignalFiles\1000000000_800000_1280000.iq'
    file_name = os.path.basename(file)
    names = os.path.splitext(file_name)[0].split('_')
    center_freq = float(names[0])
    bandwidth = float(names[1])
    sample_rate = float(names[2])
    for iq_data in read_iq_file_in_chunks(file, dtype):
        compute_spectrum(iq_data, sample_rate, bandwidth, fft_size)


if __name__ == "__main__":
    main()
