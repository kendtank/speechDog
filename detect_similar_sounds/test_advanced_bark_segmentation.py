# -*- coding: utf-8 -*-
"""
@Time    : 2024/9/18
@Author  : AI Assistant
@FileName: test_advanced_bark_segmentation.py
@Software: PyCharm
"""

"""
高级狗吠分割系统测试脚本
用于验证advanced_bark_segmentation.py的功能和性能
"""

import os
import sys
import time
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# 添加项目根目录到Python路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# 导入高级狗吠分割器
from detect_similar_sounds.advanced_bark_segmentation import AdvancedBarkSegmenter

class BarkSegmentationTester:
    def __init__(self):
        """初始化测试器"""
        self.default_params = {
            'target_sr': 16000,
            'frame_length': 0.025,
            'hop_length': 0.01,
            'energy_window_ms': 200,
            'min_bark_duration': 0.1,
            'max_bark_duration': 2.0,
            'merge_gap_threshold': 0.2,
            'short_gap_threshold': 0.05,
            'confidence_threshold': 0.6
        }
        
    def load_audio_files(self, audio_dir):
        """加载目录中的所有音频文件"""
        supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.aiff']
        audio_files = []
        
        if os.path.isfile(audio_dir):
            # 如果输入是单个文件
            ext = os.path.splitext(audio_dir)[1].lower()
            if ext in supported_formats:
                audio_files.append(audio_dir)
        elif os.path.isdir(audio_dir):
            # 如果输入是目录
            for root, _, files in os.walk(audio_dir):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in supported_formats:
                        audio_files.append(os.path.join(root, file))
        
        return audio_files
        
    def test_single_file(self, audio_path, params=None, visualize=True, output_dir=None):
        """测试单个音频文件的狗吠分割"""
        if params is None:
            params = self.default_params.copy()
            
        # 创建分割器实例
        segmenter = AdvancedBarkSegmenter(**params)
        
        print(f"\n=== 测试文件: {os.path.basename(audio_path)} ===")
        print(f"参数设置: {params}")
        
        try:
            # 加载音频并记录时间
            start_time = time.time()
            y, sr = librosa.load(audio_path, sr=None)
            load_time = time.time() - start_time
            print(f"音频加载完成，耗时: {load_time:.3f}s")
            print(f"音频信息: 采样率={sr}Hz, 时长={len(y)/sr:.2f}s, 通道数={'单声道' if len(y.shape) == 1 else '立体声'}")
            
            # 执行分割并记录时间
            start_time = time.time()
            segments = segmenter.detect_bark_segments(y, sr)
            process_time = time.time() - start_time
            
            print(f"狗吠检测完成，耗时: {process_time:.3f}s")
            print(f"检测到 {len(segments)} 个狗吠片段")
            
            # 统计片段信息
            if segments:
                durations = [end - start for start, end in segments]
                print(f"片段时长统计: 最小={min(durations):.3f}s, 平均={np.mean(durations):.3f}s, 最大={max(durations):.3f}s")
                
                # 计算检测率（检测到的狗吠时长占总音频时长的比例）
                detection_ratio = sum(durations) / (len(y) / sr) * 100
                print(f"检测率: {detection_ratio:.2f}% (检测到的狗吠时长/总音频时长)")
                
                # 提取并保存片段
                if output_dir:
                    saved_files, _ = segmenter.extract_and_save_segments(audio_path, output_dir, visualize=visualize)
                    print(f"已保存 {len(saved_files)} 个狗吠片段到 {output_dir}")
                elif visualize:
                    # 仅可视化不保存
                    segmenter.visualize_results(y, sr, segments, title=f"{os.path.basename(audio_path)} 的狗吠检测结果")
                    
            return {
                'audio_path': audio_path,
                'success': True,
                'segment_count': len(segments),
                'segments': segments,
                'load_time': load_time,
                'process_time': process_time,
                'audio_duration': len(y) / sr
            }
            
        except Exception as e:
            print(f"处理文件 {audio_path} 时出错: {str(e)}")
            return {
                'audio_path': audio_path,
                'success': False,
                'error': str(e)
            }
            
    def parameter_sweep(self, audio_path, param_name, param_values, visualize=False):
        """参数扫描：测试不同参数值的效果"""
        print(f"\n=== 参数扫描: {param_name} ===")
        print(f"测试值: {param_values}")
        
        results = []
        for value in param_values:
            params = self.default_params.copy()
            params[param_name] = value
            
            print(f"\n测试参数值: {param_name}={value}")
            result = self.test_single_file(audio_path, params, visualize=False)
            if result['success']:
                results.append({
                    'param_value': value,
                    'segment_count': result['segment_count'],
                    'process_time': result['process_time']
                })
        
        # 可视化参数扫描结果
        if results:
            self.visualize_parameter_sweep(results, param_name)
            
        return results
        
    def visualize_parameter_sweep(self, results, param_name):
        """可视化参数扫描结果"""
        plt.figure(figsize=(12, 6))
        
        param_values = [r['param_value'] for r in results]
        segment_counts = [r['segment_count'] for r in results]
        process_times = [r['process_time'] for r in results]
        
        # 绘制片段数量曲线
        ax1 = plt.subplot(121)
        ax1.plot(param_values, segment_counts, 'o-', color='blue')
        ax1.set_xlabel(f'{param_name} 值')
        ax1.set_ylabel('检测到的片段数量')
        ax1.set_title(f'{param_name} 对片段数量的影响')
        ax1.grid(True)
        
        # 绘制处理时间曲线
        ax2 = plt.subplot(122)
        ax2.plot(param_values, process_times, 's-', color='red')
        ax2.set_xlabel(f'{param_name} 值')
        ax2.set_ylabel('处理时间 (秒)')
        ax2.set_title(f'{param_name} 对处理时间的影响')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def batch_test(self, audio_dir, output_dir=None, max_workers=4):
        """批量测试多个音频文件"""
        # 加载所有音频文件
        audio_files = self.load_audio_files(audio_dir)
        if not audio_files:
            print(f"在 {audio_dir} 中未找到支持的音频文件")
            return []
            
        print(f"找到 {len(audio_files)} 个音频文件，准备进行批量测试...")
        
        # 创建输出目录
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 记录开始时间
        batch_start_time = time.time()
        
        # 使用多进程进行批量测试
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = []
            for i, audio_file in enumerate(audio_files):
                file_output_dir = None
                if output_dir:
                    # 为每个文件创建单独的输出子目录
                    base_name = os.path.splitext(os.path.basename(audio_file))[0]
                    file_output_dir = os.path.join(output_dir, base_name)
                    
                futures.append(executor.submit(self.test_single_file, audio_file, 
                                              self.default_params.copy(), 
                                              visualize=False, 
                                              output_dir=file_output_dir))
            
            # 收集结果
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)
                    print(f"文件 {i+1}/{len(audio_files)} 处理完成")
                except Exception as e:
                    print(f"文件 {i+1} 处理异常: {str(e)}")
                    results.append({
                        'audio_path': audio_files[i],
                        'success': False,
                        'error': str(e)
                    })
        
        # 统计批量测试结果
        total_time = time.time() - batch_start_time
        success_count = sum(1 for r in results if r['success'])
        total_segments = sum(r['segment_count'] for r in results if r['success'])
        
        print(f"\n=== 批量测试结果汇总 ===")
        print(f"总文件数: {len(audio_files)}")
        print(f"成功处理: {success_count}")
        print(f"失败处理: {len(audio_files) - success_count}")
        print(f"总共检测到狗吠片段: {total_segments}")
        print(f"总处理时间: {total_time:.3f}s")
        print(f"平均处理速度: {(total_time/len(audio_files)):.3f}s/文件")
        
        # 保存批量测试报告
        if output_dir:
            report_path = os.path.join(output_dir, 'batch_test_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# 批量测试报告\n")
                f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总文件数: {len(audio_files)}\n")
                f.write(f"成功处理: {success_count}\n")
                f.write(f"失败处理: {len(audio_files) - success_count}\n")
                f.write(f"总共检测到狗吠片段: {total_segments}\n")
                f.write(f"总处理时间: {total_time:.3f}s\n")
                f.write(f"平均处理速度: {(total_time/len(audio_files)):.3f}s/文件\n")
                f.write("\n# 文件详细结果\n")
                
                for result in results:
                    if result['success']:
                        f.write(f"文件: {os.path.basename(result['audio_path'])}\n")
                        f.write(f"  音频时长: {result['audio_duration']:.2f}s\n")
                        f.write(f"  片段数量: {result['segment_count']}\n")
                        f.write(f"  加载时间: {result['load_time']:.3f}s\n")
                        f.write(f"  处理时间: {result['process_time']:.3f}s\n")
                    else:
                        f.write(f"文件: {os.path.basename(result['audio_path'])}\n")
                        f.write(f"  状态: 失败\n")
                        f.write(f"  错误: {result['error']}\n")
            
            print(f"批量测试报告已保存到: {report_path}")
            
        return results

# 主函数
def main():
    parser = argparse.ArgumentParser(description='高级狗吠分割系统测试工具')
    parser.add_argument('audio_path', help='音频文件或目录路径')
    parser.add_argument('--output', '-o', help='输出目录路径')
    parser.add_argument('--batch', '-b', action='store_true', help='批量测试模式')
    parser.add_argument('--sweep', '-s', metavar='PARAM', help='参数扫描模式，指定要扫描的参数名')
    parser.add_argument('--visualize', '-v', action='store_true', help='可视化检测结果')
    parser.add_argument('--workers', '-w', type=int, default=4, help='批量测试时的进程数量')
    
    args = parser.parse_args()
    
    # 创建测试器实例
    tester = BarkSegmentationTester()
    
    # 确保输出目录存在
    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # 根据不同模式执行测试
    if args.sweep:
        # 参数扫描模式
        # 为不同参数准备默认的扫描值
        param_sweeps = {
            'min_bark_duration': [0.05, 0.1, 0.15, 0.2, 0.3],
            'max_bark_duration': [1.0, 1.5, 2.0, 2.5, 3.0],
            'merge_gap_threshold': [0.1, 0.15, 0.2, 0.25, 0.3],
            'confidence_threshold': [0.4, 0.5, 0.6, 0.7, 0.8],
            'energy_window_ms': [100, 150, 200, 250, 300],
        }
        
        if args.sweep in param_sweeps:
            tester.parameter_sweep(args.audio_path, args.sweep, param_sweeps[args.sweep])
        else:
            print(f"不支持的参数名: {args.sweep}")
            print(f"支持的参数名: {list(param_sweeps.keys())}")
            
    elif args.batch:
        # 批量测试模式
        tester.batch_test(args.audio_path, args.output, args.workers)
        
    else:
        # 单文件测试模式
        tester.test_single_file(args.audio_path, visualize=args.visualize, output_dir=args.output)

if __name__ == "__main__":
    main()

"""
使用示例:

# 单文件测试
python detect_similar_sounds/test_advanced_bark_segmentation.py D:\kend\myPython\speechDog-master\youtube_wav\bark_segmentation\processed_template_preserving\outdoor_braking_clean_preserving.wav -o ./output -v

# 批量测试
python detect_similar_sounds/test_advanced_bark_segmentation.py D:\audio_folder -o D:\batch_output -b -w 4

# 参数扫描
python detect_similar_sounds/test_advanced_bark_segmentation.py D:\test_audio.wav -s min_bark_duration

"""