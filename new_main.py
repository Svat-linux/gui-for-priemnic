import serial
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from collections import deque
import re
from datetime import datetime
import json
import os

class SerialDataReader:
    def __init__(self, port='COM7', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.data = {
            'temperature': 0.0,
            'current': 0.0,
            'latitude': 0.0,
            'longitude': 0.0,
            'altitude': 0.0,
            'magnetometer': [0, 0, 0],
            'accelerometer': [0, 0, 0],
            'gyroscope': [0, 0, 0],
            'pressure': 0.0,
            'color_temp': 0.0,
            'color_raw': [0, 0, 0, 0, 0, 0],
            'uv_intensity': 0.0
        }
        self.lock = threading.Lock()
        self.last_save_time = time.time()
        
    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            return True
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            return False
    
    def read_data(self):
        buffer = ""
        while self.running:
            try:
                if self.ser and self.ser.in_waiting > 0:
                    raw_data = self.ser.read(self.ser.in_waiting)
                    try:
                        buffer += raw_data.decode('utf-8', errors='ignore')
                    except UnicodeDecodeError:
                        continue
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            self.parse_data(line)
                    
                    # Сохранение данных каждые 10 секунд
                    current_time = time.time()
                    if current_time - self.last_save_time >= 10:
                        self.save_serial_data()
                        self.last_save_time = current_time
                    
            except Exception as e:
                print(f"Error reading data: {e}")
                time.sleep(0.1)
    
    def save_serial_data(self):
        """Сохраняет данные из serial порта в файл"""
        try:
            with self.lock:
                data_copy = self.data.copy()
            
            current_time = datetime.now()
            filename = f"serial_{current_time.hour:02d}_{current_time.minute:02d}_{current_time.second:02d}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Время сохранения: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n")
                f.write("ДАННЫЕ ИЗ SERIAL ПОРТА:\n")
                f.write("="*50 + "\n")
                
                f.write(f"Температура батареи: {data_copy['temperature']:.2f} °C\n")
                f.write(f"Ток батареи: {data_copy['current']:.2f} A\n")
                f.write(f"Температура среды: {data_copy['color_temp']:.2f} °C\n\n")
                
                f.write(f"Широта: {data_copy['latitude']:.6f}\n")
                f.write(f"Долгота: {data_copy['longitude']:.6f}\n")
                f.write(f"Высота: {data_copy['altitude']:.2f} m\n\n")
                
                f.write(f"Магнитометр: {data_copy['magnetometer']}\n")
                f.write(f"Акселерометр: {data_copy['accelerometer']}\n")
                f.write(f"Гироскоп: {data_copy['gyroscope']}\n")
                f.write(f"Давление: {data_copy['pressure']:.2f} Pa\n")
                f.write(f"UV Интенсивность: {data_copy['uv_intensity']:.2f}\n\n")
                
                f.write("Спектральные данные:\n")
                colors = ['Violet', 'Blue', 'Green', 'Yellow', 'Orange', 'Red']
                for i, color in enumerate(colors):
                    f.write(f"{color}: {data_copy['color_raw'][i]}\n")
            
            print(f"Данные сохранены в файл: {filename}")
            
        except Exception as e:
            print(f"Ошибка при сохранении serial данных: {e}")
    
    def parse_data(self, line):
        patterns = {
            'temperature': r'Temperature: (\d+\.\d+) C',
            'current': r'Current: (\d+\.\d+) A',
            'latitude': r'Latitude: (\d+\.\d+)',
            'longitude': r'Longitude: (\d+\.\d+)',
            'altitude': r'Altitude: (\d+\.\d+)',
            'magnetometer': r'Magnetometer: (-?\d+), (-?\d+), (-?\d+)',
            'accelerometer': r'Accelerometer: (-?\d+), (-?\d+), (-?\d+)',
            'gyroscope': r'Gyroscope: (-?\d+), (-?\d+), (-?\d+)',
            'pressure': r'Pressure: (\d+\.\d+) Pa',
            'color_temp': r'Color Temp: (\d+) C',
            'color_raw': r'Color Raw: Violet: (\d+), Blue: (\d+), Green: (\d+), Yellow: (\d+), Orange: (\d+), Red: (\d+)',
            'uv_intensity': r'UV Intensity: (\d+\.\d+)'
        }
        
        with self.lock:
            for key, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    try:
                        if key in ['magnetometer', 'accelerometer', 'gyroscope', 'color_raw']:
                            self.data[key] = [int(x) for x in match.groups()]
                        else:
                            self.data[key] = float(match.group(1))
                    except (ValueError, IndexError):
                        pass
    
    def get_data(self):
        with self.lock:
            return self.data.copy()
    
    def disconnect(self):
        self.running = False
        if self.ser:
            self.ser.close()

class AdvancedSpectralAnalyzer:
    def __init__(self):
        self.colors = ['Violet', 'Blue', 'Green', 'Yellow', 'Orange', 'Red']
        self.wavelengths = np.array([400, 450, 510, 570, 600, 650])
        
        # Расширенная база данных элементов и соединений
        self.element_signatures = {
            'Литий (Li)': [2, 1, 0, 15, 3, 8],
            'Натрий (Na)': [1, 2, 3, 25, 5, 2],
            'Калий (K)': [1, 3, 4, 20, 8, 3],
            'Кальций (Ca)': [3, 6, 8, 15, 20, 6],
            'Железо (Fe)': [8, 12, 15, 18, 14, 10],
            'Медь (Cu)': [6, 10, 25, 15, 8, 6],
            'Гелий (He)': [25, 8, 3, 1, 1, 5],
            'Неон (Ne)': [15, 20, 8, 4, 2, 3],
            'Водород (H)': [30, 5, 2, 1, 1, 15],
            'Кислород (O)': [5, 25, 15, 8, 5, 3],
            'Азот (N)': [8, 20, 12, 10, 6, 4],
            'Углерод (C)': [10, 15, 25, 12, 8, 5],
            'Вода (H2O)': [3, 18, 22, 12, 8, 5],
            'Углекислый газ (CO2)': [4, 16, 28, 10, 6, 4],
            'Метан (CH4)': [8, 12, 20, 15, 10, 8],
        }
    
    def calculate_average_spectrum(self, spectrum_history):
        """Усреднение спектра за 15 секунд"""
        if not spectrum_history:
            return np.zeros(6)
        
        spectra = np.array(spectrum_history)
        return np.mean(spectra, axis=0)
    
    def analyze_spectrum(self, spectrum, additional_data=None):
        """Полный спектральный анализ с усреднением"""
        spectrum_avg = self.calculate_average_spectrum(additional_data.get('spectrum_history_15s', [spectrum]))
        
        return {
            'chemical_composition': self.analyze_chemical_composition(spectrum_avg),
            'temperature': self.estimate_temperature(spectrum_avg, additional_data),
            'object_size': self.estimate_object_size(spectrum_avg, additional_data),
            'distance': self.estimate_distance(spectrum_avg, additional_data),
            'movement': self.analyze_movement(spectrum_avg, additional_data),
            'magnetic_field': self.analyze_magnetic_field(spectrum_avg, additional_data),
            'physical_properties': self.analyze_physical_properties(spectrum_avg, additional_data),
            'luminosity': self.estimate_luminosity(spectrum_avg, additional_data),
            'spectral_quality': self.analyze_spectral_quality(spectrum_avg),
            'element_probabilities': self.get_element_probabilities(spectrum_avg),
            'average_spectrum': spectrum_avg.tolist(),
            'measurement_count': len(additional_data.get('spectrum_history_15s', [])),
            'time_window': len(additional_data.get('spectrum_history_15s', [])) * 0.3
        }
    
    def analyze_chemical_composition(self, spectrum):
        """Детальный анализ химического состава"""
        spectrum_norm = spectrum / (np.max(spectrum) + 0.001)
        
        matches = []
        for element, signature in self.element_signatures.items():
            signature_norm = np.array(signature) / (np.max(signature) + 0.001)
            score = np.dot(spectrum_norm, signature_norm) / (
                np.linalg.norm(spectrum_norm) * np.linalg.norm(signature_norm) + 0.001
            )
            matches.append((element, score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def get_element_probabilities(self, spectrum):
        """Вероятности элементов"""
        matches = self.analyze_chemical_composition(spectrum)
        total_score = sum(score for _, score in matches[:10])
        
        probabilities = []
        for element, score in matches[:10]:
            probability = (score / total_score * 100) if total_score > 0 else 0
            probabilities.append((element, probability))
        
        return probabilities
    
    def estimate_temperature(self, spectrum, additional_data):
        """Точная оценка температуры"""
        if np.sum(spectrum) == 0:
            return "Недостаточно данных"
        
        weighted_avg_wavelength = np.sum(self.wavelengths * spectrum) / np.sum(spectrum)
        wien_constant = 2898000
        temperature_k = wien_constant / weighted_avg_wavelength
        
        blue_red_ratio = np.sum(spectrum[1:3]) / (np.sum(spectrum[4:6]) + 0.001)
        if blue_red_ratio > 2.5:
            temperature_k *= 1.3
        elif blue_red_ratio > 1.5:
            temperature_k *= 1.15
        
        return f"{temperature_k:.0f} K ({temperature_k - 273:.0f} °C)"
    
    def estimate_object_size(self, spectrum, additional_data):
        """Оценка размера объекта"""
        total_intensity = np.sum(spectrum)
        
        if total_intensity < 5:
            size = "Очень малый"
        elif total_intensity < 15:
            size = "Малый"
        elif total_intensity < 30:
            size = "Средний"
        elif total_intensity < 50:
            size = "Крупный"
        else:
            size = "Очень крупный"
            
        return size
    
    def estimate_distance(self, spectrum, additional_data):
        """Оценка расстояния"""
        total_intensity = np.sum(spectrum)
        
        if total_intensity > 60:
            return "Очень близко (<1 м)"
        elif total_intensity > 40:
            return "Близко (1-5 м)"
        elif total_intensity > 20:
            return "Средняя дистанция (5-20 м)"
        elif total_intensity > 10:
            return "Далеко (20-100 м)"
        else:
            return "Очень далеко (>100 м)"
    
    def analyze_movement(self, spectrum, additional_data):
        """Анализ движения"""
        return "Стабильное положение"
    
    def analyze_magnetic_field(self, spectrum, additional_data):
        """Анализ магнитного поля"""
        line_variation = np.std(spectrum) / (np.mean(spectrum) + 0.001)
        
        if line_variation > 1.0:
            return "Сильное поле"
        elif line_variation > 0.3:
            return "Среднее поле"
        else:
            return "Слабое поле"
    
    def analyze_physical_properties(self, spectrum, additional_data):
        """Анализ физических свойств"""
        return "Средняя плотность, нормальное давление"
    
    def estimate_luminosity(self, spectrum, additional_data):
        """Оценка светимости"""
        total_intensity = np.sum(spectrum)
        
        if total_intensity > 30:
            return "Высокая светимость"
        elif total_intensity > 15:
            return "Средняя светимость"
        else:
            return "Низкая светимость"
    
    def analyze_spectral_quality(self, spectrum):
        """Качество спектральных данных"""
        snr = np.mean(spectrum) / (np.std(spectrum) + 0.001)
        
        if snr > 3.0:
            return "Хорошее качество"
        elif snr > 1.5:
            return "Удовлетворительное качество"
        else:
            return "Низкое качество"
    
    def save_analysis_to_file(self, analysis_data, raw_data):
        """Сохраняет результаты анализа в файл"""
        try:
            current_time = datetime.now()
            filename = f"analiz_{current_time.hour:02d}_{current_time.minute:02d}_{current_time.second:02d}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Время анализа: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n")
                f.write("ДЕТАЛЬНЫЙ СПЕКТРАЛЬНЫЙ АНАЛИЗ (ЗА 15 СЕКУНД)\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Количество измерений: {analysis_data['measurement_count']}\n")
                f.write(f"Временное окно: {analysis_data['time_window']:.1f} секунд\n\n")
                
                f.write("1. КАЧЕСТВО ДАННЫХ:\n")
                f.write(f"   {analysis_data['spectral_quality']}\n\n")
                
                f.write("2. ХИМИЧЕСКИЙ СОСТАВ (ВЕРОЯТНОСТИ):\n")
                for element, probability in analysis_data['element_probabilities']:
                    f.write(f"   {element:<20} {probability:6.1f}%\n")
                f.write("\n")
                
                f.write("3. ТЕМПЕРАТУРНЫЕ ХАРАКТЕРИСТИКИ:\n")
                f.write(f"   Температура объекта: {analysis_data['temperature']}\n")
                f.write(f"   Температура среды: {raw_data['color_temp']:.1f} °C\n\n")
                
                f.write("4. ФИЗИЧЕСКИЕ СВОЙСТВА:\n")
                f.write(f"   Размер объекта: {analysis_data['object_size']}\n")
                f.write(f"   Светимость: {analysis_data['luminosity']}\n")
                f.write(f"   {analysis_data['physical_properties']}\n\n")
                
                f.write("5. ПОЗИЦИОННЫЕ ДАННЫЕ:\n")
                f.write(f"   Расстояние до объекта: {analysis_data['distance']}\n")
                f.write(f"   Движение: {analysis_data['movement']}\n\n")
                
                f.write("6. МАГНИТНЫЕ ХАРАКТЕРИСТИКИ:\n")
                f.write(f"   Магнитное поле: {analysis_data['magnetic_field']}\n\n")
                
                f.write("7. УСРЕДНЕННЫЕ ДАННЫЕ СПЕКТРА:\n")
                colors = ['Violet', 'Blue', 'Green', 'Yellow', 'Orange', 'Red']
                for i, color in enumerate(colors):
                    f.write(f"   {color}: {analysis_data['average_spectrum'][i]:6.1f}\n")
                f.write(f"   Общая интенсивность: {sum(analysis_data['average_spectrum']):.1f}\n\n")
                
                f.write("8. ТЕКУЩИЕ ДАННЫЕ:\n")
                for i, color in enumerate(colors):
                    f.write(f"   {color}: {raw_data['color_raw'][i]:3d}\n")
                f.write("\n")
                
                f.write("9. ДОПОЛНИТЕЛЬНЫЕ ДАТЧИКИ:\n")
                f.write(f"   Давление: {raw_data['pressure']:.2f} Pa\n")
                f.write(f"   UV Интенсивность: {raw_data['uv_intensity']:.2f}\n")
                f.write(f"   Высота: {raw_data['altitude']:.2f} m\n")
                f.write(f"   Координаты: {raw_data['latitude']:.6f}, {raw_data['longitude']:.6f}\n")
            
            print(f"Анализ сохранен в файл: {filename}")
            
        except Exception as e:
            print(f"Ошибка при сохранении анализа: {e}")

class SerialMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Монитор данных с Serial порта")
        self.root.geometry("1200x800")
        
        self.reader = SerialDataReader()
        self.analyzer = AdvancedSpectralAnalyzer()
        self.spectrum_history = deque(maxlen=50)  # 50 измерений = 15 секунд
        self.last_analysis_time = 0
        self.analysis_results = None
        
        self.setup_ui()
        self.connect_serial()
    
    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Основные данные
        self.main_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text="Основные данные")
        
        # Спектральный анализ (график)
        self.spectrum_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.spectrum_frame, text="Спектральный анализ")
        
        # Детальный анализ (каждые 15 сек)
        self.detail_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.detail_frame, text="Детальный анализ (15 сек)")
        
        self.setup_main_tab()
        self.setup_spectrum_tab()
        self.setup_detail_tab()
    
    def setup_main_tab(self):
        main_container = ttk.Frame(self.main_frame)
        main_container.pack(fill='both', expand=True)
        
        # Левый фрейм - температура и батарея
        left_frame = ttk.LabelFrame(main_container, text="Энергетика")
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        
        ttk.Label(left_frame, text="Температура батареи:").grid(row=0, column=0, sticky='w', pady=2)
        self.temp_label = ttk.Label(left_frame, text="0.00 °C", font=('Arial', 10, 'bold'))
        self.temp_label.grid(row=0, column=1, sticky='w', pady=2)
        
        ttk.Label(left_frame, text="Ток батареи:").grid(row=1, column=0, sticky='w', pady=2)
        self.current_label = ttk.Label(left_frame, text="0.00 A", font=('Arial', 10, 'bold'))
        self.current_label.grid(row=1, column=1, sticky='w', pady=2)
        
        ttk.Label(left_frame, text="Температура среды:").grid(row=2, column=0, sticky='w', pady=2)
        self.color_temp_label = ttk.Label(left_frame, text="0.00 °C", font=('Arial', 10, 'bold'))
        self.color_temp_label.grid(row=2, column=1, sticky='w', pady=2)
        
        # Центральный фрейм - координаты
        center_frame = ttk.LabelFrame(main_container, text="Позиция")
        center_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        
        ttk.Label(center_frame, text="Широта:").grid(row=0, column=0, sticky='w', pady=2)
        self.lat_label = ttk.Label(center_frame, text="0.000000", font=('Arial', 10))
        self.lat_label.grid(row=0, column=1, sticky='w', pady=2)
        
        ttk.Label(center_frame, text="Долгота:").grid(row=1, column=0, sticky='w', pady=2)
        self.lon_label = ttk.Label(center_frame, text="0.000000", font=('Arial', 10))
        self.lon_label.grid(row=1, column=1, sticky='w', pady=2)
        
        ttk.Label(center_frame, text="Высота:").grid(row=2, column=0, sticky='w', pady=2)
        self.alt_label = ttk.Label(center_frame, text="0.00 m", font=('Arial', 10))
        self.alt_label.grid(row=2, column=1, sticky='w', pady=2)
        
        # Правый фрейм - датчики
        right_frame = ttk.LabelFrame(main_container, text="Датчики")
        right_frame.grid(row=0, column=2, padx=5, pady=5, sticky='nsew')
        
        sensors = [
            ("Магнитометр:", "self.mag_label"),
            ("Акселерометр:", "self.acc_label"),
            ("Гироскоп:", "self.gyro_label"),
            ("Давление:", "self.pressure_label"),
            ("UV Интенсивность:", "self.uv_label")
        ]
        
        for i, (text, _) in enumerate(sensors):
            ttk.Label(right_frame, text=text).grid(row=i, column=0, sticky='w', pady=2)
        
        self.mag_label = ttk.Label(right_frame, text="0, 0, 0", font=('Arial', 9))
        self.mag_label.grid(row=0, column=1, sticky='w', pady=2)
        
        self.acc_label = ttk.Label(right_frame, text="0, 0, 0", font=('Arial', 9))
        self.acc_label.grid(row=1, column=1, sticky='w', pady=2)
        
        self.gyro_label = ttk.Label(right_frame, text="0, 0, 0", font=('Arial', 9))
        self.gyro_label.grid(row=2, column=1, sticky='w', pady=2)
        
        self.pressure_label = ttk.Label(right_frame, text="0.00 Pa", font=('Arial', 9))
        self.pressure_label.grid(row=3, column=1, sticky='w', pady=2)
        
        self.uv_label = ttk.Label(right_frame, text="0.00", font=('Arial', 9))
        self.uv_label.grid(row=4, column=1, sticky='w', pady=2)
        
        # Нижний фрейм - спектральные данные
        bottom_frame = ttk.LabelFrame(main_container, text="Спектральные данные")
        bottom_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky='ew')
        
        colors = ['Violet', 'Blue', 'Green', 'Yellow', 'Orange', 'Red']
        for i, color in enumerate(colors):
            ttk.Label(bottom_frame, text=f"{color}:").grid(row=0, column=i*2, padx=2)
            label = ttk.Label(bottom_frame, text="0", font=('Arial', 9, 'bold'))
            label.grid(row=0, column=i*2+1, padx=2)
            setattr(self, f'{color.lower()}_label', label)
        
        for i in range(3):
            main_container.columnconfigure(i, weight=1)
        main_container.rowconfigure(1, weight=1)
    
    def setup_spectrum_tab(self):
        # График спектра
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, self.spectrum_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        info_frame = ttk.Frame(self.spectrum_frame)
        info_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(info_frame, text="Текущий спектр (мгновенные значения)").pack()
    
    def setup_detail_tab(self):
        # Верхняя часть - информация об анализе
        top_frame = ttk.LabelFrame(self.detail_frame, text="Анализ за последние 15 секунд")
        top_frame.pack(fill='x', padx=5, pady=5)
        
        info_frame = ttk.Frame(top_frame)
        info_frame.pack(fill='x', padx=5, pady=5)
        
        self.analysis_time_label = ttk.Label(info_frame, text="Последний анализ: не проводился", font=('Arial', 10))
        self.analysis_time_label.pack()
        
        self.next_analysis_label = ttk.Label(info_frame, text="Следующий анализ через: 15 сек", font=('Arial', 10, 'bold'))
        self.next_analysis_label.pack()
        
        # Детальный анализ
        analysis_frame = ttk.LabelFrame(self.detail_frame, text="Результаты анализа")
        analysis_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Создаем фрейм с прокруткой
        scroll_frame = ttk.Frame(analysis_frame)
        scroll_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Холст и скроллбар
        self.canvas_analysis = tk.Canvas(scroll_frame)
        scrollbar = ttk.Scrollbar(scroll_frame, orient='vertical', command=self.canvas_analysis.yview)
        self.scrollable_frame = ttk.Frame(self.canvas_analysis)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas_analysis.configure(scrollregion=self.canvas_analysis.bbox("all"))
        )
        
        self.canvas_analysis.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_analysis.configure(yscrollcommand=scrollbar.set)
        
        # Текстовое поле для анализа
        self.analysis_text = tk.Text(self.scrollable_frame, height=25, width=100, font=('Courier', 9), wrap='word')
        self.analysis_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Упаковка скроллбара и холста
        self.canvas_analysis.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Привязка колеса мыши к прокрутке
        def _on_mousewheel(event):
            self.canvas_analysis.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.canvas_analysis.bind_all("<MouseWheel>", _on_mousewheel)
    
    def connect_serial(self):
        if self.reader.connect():
            self.thread = threading.Thread(target=self.reader.read_data, daemon=True)
            self.thread.start()
            self.last_analysis_time = time.time()
            self.schedule_update()
        else:
            messagebox.showerror("Ошибка", "Не удалось подключиться к serial порту")
    
    def schedule_update(self):
        current_time = time.time()
        self.update_display()
        
        # Обновляем счетчик до следующего анализа
        time_until_next = 15 - (current_time - self.last_analysis_time)
        if time_until_next < 0:
            time_until_next = 0
        
        self.next_analysis_label.config(text=f"Следующий анализ через: {time_until_next:.0f} сек")
        
        # Запускаем анализ каждые 15 секунд
        if current_time - self.last_analysis_time >= 15:
            self.perform_detailed_analysis()
            self.last_analysis_time = current_time
        
        self.root.after(1000, self.schedule_update)  # Обновление каждую секунду
    
    def update_display(self):
        try:
            data = self.reader.get_data()
            color_data = data['color_raw']
            self.spectrum_history.append(color_data.copy())
            
            # Обновляем основные метки
            self.temp_label.config(text=f"{data['temperature']:.2f} °C")
            self.current_label.config(text=f"{data['current']:.2f} A")
            self.color_temp_label.config(text=f"{data['color_temp']:.2f} °C")
            
            self.lat_label.config(text=f"{data['latitude']:.6f}")
            self.lon_label.config(text=f"{data['longitude']:.6f}")
            self.alt_label.config(text=f"{data['altitude']:.2f} m")
            
            self.mag_label.config(text=f"{data['magnetometer'][0]}, {data['magnetometer'][1]}, {data['magnetometer'][2]}")
            self.acc_label.config(text=f"{data['accelerometer'][0]}, {data['accelerometer'][1]}, {data['accelerometer'][2]}")
            self.gyro_label.config(text=f"{data['gyroscope'][0]}, {data['gyroscope'][1]}, {data['gyroscope'][2]}")
            self.pressure_label.config(text=f"{data['pressure']:.2f} Pa")
            self.uv_label.config(text=f"{data['uv_intensity']:.2f}")
            
            # Обновляем спектральные метки
            colors = ['violet', 'blue', 'green', 'yellow', 'orange', 'red']
            for i, color in enumerate(colors):
                label = getattr(self, f'{color}_label')
                label.config(text=str(data['color_raw'][i]))
            
            # Обновляем график
            self.update_spectrum_graph(data['color_raw'])
            
        except Exception as e:
            print(f"Error updating display: {e}")
    
    def perform_detailed_analysis(self):
        """Выполнение детального анализа каждые 15 секунд"""
        try:
            data = self.reader.get_data()
            additional_data = {
                'spectrum_history_15s': list(self.spectrum_history),
                'pressure': data['pressure'],
                'uv_intensity': data['uv_intensity'],
                'magnetometer': data['magnetometer']
            }
            
            analysis = self.analyzer.analyze_spectrum(data['color_raw'], additional_data)
            self.update_detailed_analysis(data, analysis)
            
            # Сохраняем результаты анализа в файл
            self.analyzer.save_analysis_to_file(analysis, data)
            
        except Exception as e:
            print(f"Error in detailed analysis: {e}")
    
    def update_spectrum_graph(self, color_data):
        self.ax.clear()
        colors = ['violet', 'blue', 'green', 'yellow', 'orange', 'red']
        x_pos = np.arange(len(colors))
        
        bars = self.ax.bar(x_pos, color_data, color=colors, alpha=0.7, edgecolor='black')
        self.ax.set_xlabel('Цвета')
        self.ax.set_ylabel('Интенсивность')
        self.ax.set_title('Текущий спектр (мгновенные значения)')
        self.ax.set_xticks(x_pos)
        self.ax.set_xticklabels(colors)
        self.ax.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, color_data):
            self.ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value}', ha='center', va='bottom')
        
        self.canvas.draw()
    
    def update_detailed_analysis(self, data, analysis):
        """Обновление детального анализа"""
        # Сохраняем позицию прокрутки
        scroll_position = self.canvas_analysis.yview()
        
        # Обновляем время последнего анализа
        self.analysis_time_label.config(
            text=f"Последний анализ: {datetime.now().strftime('%H:%M:%S')}"
        )
        
        # Формируем отчет
        analysis_report = self.generate_detailed_report(data, analysis)
        
        # Обновляем текстовое поле
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(1.0, analysis_report)
        
        # Восстанавливаем позицию прокрутки
        self.root.after(100, lambda: self.canvas_analysis.yview_moveto(scroll_position[0]))
    
    def generate_detailed_report(self, data, analysis):
        report = f"""
{'='*80}
ДЕТАЛЬНЫЙ СПЕКТРАЛЬНЫЙ АНАЛИЗ (ЗА 15 СЕКУНД)
{'='*80}
Время анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Количество измерений: {analysis['measurement_count']}
Временное окно: {analysis['time_window']:.1f} секунд

1. КАЧЕСТВО ДАННЫХ:
   {analysis['spectral_quality']}

2. ХИМИЧЕСКИЙ СОСТАВ (ВЕРОЯТНОСТИ):
"""
        # Топ-10 наиболее вероятных элементов
        for element, probability in analysis['element_probabilities']:
            report += f"   {element:<20} {probability:6.1f}%\n"

        report += f"""
3. ТЕМПературные характеристики:
   Температура объекта: {analysis['temperature']}
   Температура среды: {data['color_temp']:.1f} °C

4. ФИЗИЧЕСКИЕ СВОЙСТВА:
   Размер объекта: {analysis['object_size']}
   Светимость: {analysis['luminosity']}
   {analysis['physical_properties']}

5. ПОЗИЦИОННЫЕ ДАННЫЕ:
   Расстояние до объекта: {analysis['distance']}
   Движение: {analysis['movement']}

6. МАГНИТНЫЕ ХАРАКТЕРИСТИКИ:
   Магнитное поле: {analysis['magnetic_field']}

7. УСРЕДНЕННЫЕ ДАННЫЕ СПЕКТРА:
   Violet: {analysis['average_spectrum'][0]:6.1f}
   Blue:  {analysis['average_spectrum'][1]:6.1f}
   Green: {analysis['average_spectrum'][2]:6.1f}
   Yellow: {analysis['average_spectrum'][3]:6.1f}
   Orange: {analysis['average_spectrum'][4]:6.1f}
   Red:   {analysis['average_spectrum'][5]:6.1f}
   Общая интенсивность: {sum(analysis['average_spectrum']):.1f}

8. ТЕКУЩИЕ ДАННЫЕ:
   Violet: {data['color_raw'][0]:3d}
   Blue:  {data['color_raw'][1]:3d}
   Green: {data['color_raw'][2]:3d}
   Yellow: {data['color_raw'][3]:3d}
   Orange: {data['color_raw'][4]:3d}
   Red:   {data['color_raw'][5]:3d}

9. ДОПОЛНИТЕЛЬНЫЕ ДАТЧИКИ:
   Давление: {data['pressure']:.2f} Pa
   UV Интенсивность: {data['uv_intensity']:.2f}
   Высота: {data['altitude']:.2f} m
   Координаты: {data['latitude']:.6f}, {data['longitude']:.6f}

{'='*80}
"""
        return report

def main():
    root = tk.Tk()
    app = SerialMonitorApp(root)
    
    def on_closing():
        app.reader.disconnect()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
