# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/19 上午11:28
@Author  : Kend
@FileName: get_time_audio.py
@Software: PyCharm
@modifier:
"""

# -*- coding: utf-8 -*-
"""
统计音频文件时长
"""

import os
import soundfile as sf


def get_audio_duration(file_path):
    """
    获取音频时长（秒）
    """
    try:
        data, samplerate = sf.read(file_path)
        duration = len(data) / samplerate
        return duration
    except Exception as e:
        print(f"无法读取 {file_path}: {e}")
        return None


def scan_directory(dir_path, extensions=None):
    """
    扫描目录下所有音频文件，并打印时长
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac']

    audio_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                audio_files.append(os.path.join(root, file))

    for file_path in audio_files:
        duration = get_audio_duration(file_path)
        if duration is not None:
            print(f"{file_path} → {duration:.3f} 秒")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python check_audio_duration.py <音频目录>")
        sys.exit(1)

    audio_dir = sys.argv[1]
    scan_directory(audio_dir)
"""
python utils\get_time_audio.py D:\\work\\datasets\\tinyML\\bark_origion
"""