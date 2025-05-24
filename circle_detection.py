import cv2
import numpy as np
import os
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import json

frame_queue = queue.Queue(maxsize=5)  # 增加缓冲大小
result_queue = queue.Queue(maxsize=5)  # 增加缓冲大小
diameter_data = []
time_data = []
detection_active = False
file_list = []  # 存储所有结果文件路径（按编号排序）
current_index = -1  # 当前选中的文件索引

def process_frame():
    global start_time
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
            
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 二值化处理
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # 创建彩色图层用于显示
        display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # 仅在检测激活时执行圆检测
        if detection_active:
            # 使用霍夫圆变换检测圆
            circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, dp=1, minDist=500,
                                      param1=50, param2=5, minRadius=10, maxRadius=100)
            
            # 只处理检测到的一个圆
            if circles is not None and len(circles[0]) > 0:
                circles = np.uint16(np.around(circles))
                circle = circles[0][0]  # 只取第一个圆
                
                # 绘制圆和圆心
                cv2.circle(display, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                cv2.circle(display, (circle[0], circle[1]), 2, (0, 255, 0), 3)
                
                # 显示直径(像素)
                diameter = circle[2] * 2
                cv2.putText(display, f"Diameter: {diameter}px", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 阈值过滤和平滑处理
                # 初始化滑动窗口（保留最近5个有效数据）和前一个有效直径
                if not hasattr(process_frame, 'window'):
                    process_frame.window = []
                    process_frame.prev_diameter = diameter  # 初始化为第一个有效直径
                elif process_frame.prev_diameter is None:
                    process_frame.prev_diameter = diameter
                
                # 动态阈值判断（基于前一个直径的10%变化范围）
                threshold = max(5, process_frame.prev_diameter * 0.1) if process_frame.prev_diameter else 5
                if abs(diameter - process_frame.prev_diameter) <= threshold:
                    valid_diameter = diameter
                    process_frame.prev_diameter = valid_diameter
                else:
                    valid_diameter = process_frame.prev_diameter  # 使用前一个有效直径
                
                # 滑动平均（窗口大小增加到5）
                process_frame.window.append(valid_diameter)
                if len(process_frame.window) > 5:
                    process_frame.window.pop(0)
                smoothed_diameter = sum(process_frame.window) / len(process_frame.window)  # 使用浮点数计算
                
                # 添加调试输出
                print(f"Original diameter: {diameter}px,Effective diameter: {valid_diameter}px, Smooth diameter: {smoothed_diameter:.2f}px")
                
                # 记录平滑后的数据
                diameter_data.append(smoothed_diameter)
                time_data.append(time.time() - start_time)
        
        result_queue.put(display)

def detect_circle(update_preview, canvas, ax):
    # 初始化摄像头
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Can not open camera")
        return
    
    # 启动处理线程
    processor = threading.Thread(target=process_frame)
    processor.start()
    
    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("Unable to read camera frames")
            break
            
        # 将帧放入队列
        if frame_queue.empty():
            frame_queue.put(frame)
        
        # 获取处理结果
        if not result_queue.empty():
            display = result_queue.get()
            update_preview(display)
            
            # 更新图表
            if detection_active:
                ax.cla()
                ax.plot(time_data, diameter_data, 'b-')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Diameter (px)')
                ax.set_title('Real-time Diameter Measurement')
                ax.grid(True)
                canvas.draw()
        
        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 清理线程
    frame_queue.put(None)
    processor.join()
    
    cap.release()

def update_plot(frame):
    plt.cla()
    plt.plot(time_data, diameter_data, 'b-')
    plt.xlabel('Time (s)')
    plt.ylabel('Diameter (px)')
    plt.title('Real-time Diameter Measurement')
    plt.grid(True)

def start_detection():
    global detection_active, start_time
    detection_active = True
    file_label.pack_forget()  # 检测时隐藏标签
    diameter_data.clear()
    time_data.clear()
    start_time = time.time()
    # 重置滑动窗口数据
    if hasattr(process_frame, 'window'):
        process_frame.window = []
        process_frame.prev_diameter = None
    
    # 重置图表
    ax.cla()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Diameter (px)')
    ax.set_title('Real-time Diameter Measurement')
    ax.grid(True)
    canvas.draw()
    
    # 在单独线程中运行检测循环，避免阻塞主窗口
    def detection_loop():
        global detection_active
        while time.time() - start_time < 2 and detection_active:
            if not result_queue.empty():
                display = result_queue.get()
                update_preview(display)  # 使用主窗口预览标签显示
            
            # 降低循环频率避免CPU占用过高
            time.sleep(0.01)
        
        # 检测结束后一次性更新图表
        if diameter_data:
            ax.cla()
            ax.plot(time_data, diameter_data, 'b-')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Diameter (px)')
            ax.set_title('Real-time Diameter Measurement')
            ax.grid(True)
            if diameter_data:  # 自动调整Y轴范围
                ax.set_ylim(min(diameter_data)*0.9, max(diameter_data)*1.1)
                ax.relim()
                ax.autoscale_view()
            canvas.draw()
        
        detection_active = False
    
    threading.Thread(target=detection_loop, daemon=True).start()

def save_results():
    # 查找当前最大的文件编号
    max_num = 0
    for file in os.listdir('.'):
        if file.endswith('.json') and file[:-5].isdigit():
            num = int(file[:-5])
            if num > max_num:
                max_num = num
    new_num = max_num + 1
    save_file = f"{new_num}.json"
    # 将numpy类型转换为Python原生类型
    data = {'time': time_data, 'diameter': [int(d) for d in diameter_data]}
    with open(save_file, 'w') as f:
        json.dump(data, f)
    print(f"Data saved to {save_file}")
    # 更新文件列表
    update_file_list()

def update_file_list():
    global file_list, current_index
    # 获取所有数字命名的JSON文件
    file_list = []
    for file in os.listdir('.'):
        if file.endswith('.json') and file[:-5].isdigit():
            file_list.append(file)
    # 按编号排序
    file_list.sort(key=lambda x: int(x[:-5]))
    # 如果有文件，current_index设为最后一个（最新保存的）
    if file_list:
        current_index = len(file_list) - 1
    else:
        current_index = -1

def load_previous():
    global current_index
    if current_index > 0:
        current_index -= 1
        load_file(file_list[current_index])

def load_next():
    global current_index
    if current_index < len(file_list) - 1:
        current_index += 1
        load_file(file_list[current_index])

def load_file(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            global time_data, diameter_data
            time_data = data['time']
            diameter_data = data['diameter']
            update_plot(0)
            canvas.draw()
            print(f"Data loaded from {filename}")
            file_label.config(text=f"          Now File:{filename}")  # 更新标签内容
            file_label.pack()  # 显示标签
    except FileNotFoundError:
        print("No saved data found")
    except json.JSONDecodeError as e:
        print(f"ERROR: {filename} ,INCORRECT FORMAT,UNABLE TO PARSE.ERROR INFO:{e}")

def exit_program():
    root.destroy()
    os._exit(0)

if __name__ == "__main__":
    update_file_list()  # 初始化文件列表
    import tkinter as tk
    from tkinter import ttk, Label
    from PIL import Image, ImageTk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    
    # 创建主窗口
    root = tk.Tk()
    root.title('Opticeye Circle Detection System')
    root.geometry('1240x1080')  # 设置固定尺寸为1024x1024正方形
    # 初始化按钮样式
    style = ttk.Style()
    style.configure('TButton', font=('Arial', 30))
    win_width = 1240  # 窗口宽度
    win_height = 1080  # 窗口高度
    left_width = int(win_width * 6/10)  # 左侧占6/10宽度
    right_width = win_width - left_width  # 右侧宽度
    # 移除全屏相关计算，使用固定窗口尺寸
    # 创建布局框架并使用绝对定位
    preview_frame = ttk.Frame(root, padding=10)
    preview_frame.place(x=0, y=0, width=left_width, height=win_height)  # 左侧全屏显示
    
    plot_frame = ttk.Frame(root, padding=10)
    plot_frame.place(x=left_width + right_width/2, y=win_height * 0.35, anchor='center', width=right_width, height=int(win_height * 0.5))  # 调整高度适应新宽度
    plot_frame.grid_propagate(False)  # 禁止自动调整大小
    
    control_frame = ttk.Frame(root, padding=10)
    control_frame.place(x=left_width + right_width/2, y=win_height * 0.83, anchor='center', width=right_width, height=int(win_height * 0.5))  # 调整高度适应新宽度
    control_frame.grid_propagate(False)  # 禁止自动调整大小
    
    # 控制按钮
    start_btn = ttk.Button(control_frame, text='Start Detection', command=start_detection, style='TButton')
    start_btn.pack(pady=5, fill='x', anchor='center')
    
    save_btn = ttk.Button(control_frame, text='Save Results', command=save_results, style='TButton')
    save_btn.pack(pady=5, fill='x', anchor='center')
    
    load_prev_btn = ttk.Button(control_frame, text='Load Previous', command=load_previous, style='TButton')
    load_prev_btn.pack(pady=5, fill='x', anchor='center')

    load_next_btn = ttk.Button(control_frame, text='Load Next', command=load_next, style='TButton')
    load_next_btn.pack(pady=5, fill='x', anchor='center')

    exit_btn = ttk.Button(control_frame, text='Exit System', command=exit_program, style='TButton')
    exit_btn.pack(pady=5, fill='x', anchor='center')
    
    file_label = ttk.Label(plot_frame, text="", font=('Arial',30))  # 添加文件提示标签
    file_label.pack(side='bottom', pady=15, fill='x', anchor='center')
    
    # 图像预览标签
    preview_label = Label(preview_frame)
    preview_label.pack(fill='both', expand=True, anchor='center')  # 使预览标签在左侧区域居中显示
    
    # 初始化图表
    fig, ax = plt.subplots(figsize=(10, 3))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def update_preview(image):
        def _update():
            # 转换颜色空间
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            preview_label.img = img
            preview_label.config(image=img)
        root.after(0, _update)
    
    # 启动主检测循环
    def run_detection():
        detect_circle(update_preview, canvas, ax)
    
    threading.Thread(target=run_detection).start()
    root.mainloop()