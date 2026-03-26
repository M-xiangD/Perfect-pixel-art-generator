import os
import sys
import numpy as np
import cv2
from PIL import Image
import gradio as gr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "src"
))

try:
    from perfect_pixel.perfect_pixel import get_perfect_pixel
    perfect_pixel_available = True
except ImportError:
    perfect_pixel_available = False

def pixelate_image(image, pixel_size, num_colors):
    """
    步骤1：将图片转换为像素画风格
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)
    
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    
    if num_colors > 0 and num_colors < 256:
        Z = pixelated.reshape((-1, 3))
        Z = np.float32(Z)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(
            Z, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        center = np.uint8(center)
        res = center[label.flatten()]
        pixelated = res.reshape((pixelated.shape))
    
    return pixelated

def optimize_with_perfect_pixel(image, pixel_count, sample_method, export_scale, add_grid):
    """
    步骤2：使用 PerfectPixel 优化像素画
    pixel_count: 控制最终输出的像素数量（越大像素越多越精细，越小像素越少越粗糙）
    """
    if not perfect_pixel_available or image is None:
        return image
    
    if " (" in sample_method:
        sample_method = sample_method.split(" (")[0]
    
    try:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        h, w = image.shape[:2]
        grid_w = max(4, int(w * pixel_count / 100))
        grid_h = max(4, int(h * pixel_count / 100))
        
        refined_w, refined_h, optimized = get_perfect_pixel(
            image, 
            sample_method=sample_method,
            grid_size=(grid_w, grid_h)
        )
        
        scale = int(export_scale)
        if scale > 1:
            optimized = cv2.resize(
                optimized, 
                (optimized.shape[1] * scale, optimized.shape[0] * scale),
                interpolation=cv2.INTER_NEAREST
            )
        
        if add_grid:
            h, w = optimized.shape[:2]
            grid_px = scale
            for x in range(0, w + grid_px, grid_px):
                cv2.line(optimized, (x, 0), (x, h), (0, 0, 0), 1)
            for y in range(0, h + grid_px, grid_px):
                cv2.line(optimized, (0, y), (w, y), (0, 0, 0), 1)
        
        return optimized
    except:
        return image

def process_image(image, pixel_size, num_colors, pixel_count, sample_method, export_scale, add_grid):
    """
    处理图片的主函数
    """
    if image is None:
        return None, None
    
    # 步骤1：像素化
    step1_result = pixelate_image(image, pixel_size, num_colors)
    
    # 步骤2：PerfectPixel优化
    step2_result = optimize_with_perfect_pixel(
        step1_result, pixel_count, sample_method, export_scale, add_grid
    )
    
    return step1_result, step2_result

def realtime_process(frame, pixel_size, num_colors, pixel_count, sample_method, export_scale, add_grid, is_processing):
    """
    实时摄像头处理
    """
    if frame is None or not is_processing:
        return frame, frame
    
    try:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        step1_result = pixelate_image(frame, pixel_size, num_colors)
        
        step2_result = optimize_with_perfect_pixel(
            step1_result, pixel_count, sample_method, export_scale, add_grid
        )
        
        return step1_result, step2_result
    except:
        return frame, frame

def start_processing():
    return True

def stop_processing():
    return False

def handle_open():
    return gr.update(streaming=True, sources=["webcam"], type="numpy"), False

def handle_close():
    return gr.update(streaming=False, value=None), False

with gr.Blocks(title="完美像素画生成器") as demo:
    gr.Markdown("# 完美像素画生成器")
    gr.Markdown("将任意图片转换为像素画风格，并使用 PerfectPixel 优化")
    
    with gr.Tab("图片处理"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 参数设置")
                gr.Markdown("- **像素大小**：控制像素块的大小，值越大，像素块越大")
                gr.Markdown("- **颜色数量**：控制颜色的数量，值越小，颜色越少")
                gr.Markdown("- **像素数量**：控制PerfectPixel优化后的像素密度，值越大越精细")
                gr.Markdown("- **采样方法**：选择网格采样的方式")
                gr.Markdown("- **放大倍数**：控制输出图片的放大倍数")
                gr.Markdown("- **添加网格线**：是否在输出图片上添加网格线")
                
                pixel_size = gr.Slider(
                    minimum=4, 
                    maximum=32, 
                    value=8, 
                    step=1, 
                    label="像素大小"
                )
                num_colors = gr.Slider(
                    minimum=2, 
                    maximum=32, 
                    value=8, 
                    step=1, 
                    label="颜色数量"
                )
                pixel_count = gr.Slider(
                    minimum=5, 
                    maximum=100, 
                    value=20, 
                    step=1, 
                    label="像素数量（控制精细度）"
                )
                sample_method = gr.Dropdown(
                    choices=[
                        "majority (多数采样 - 取网格中最常见像素)",
                        "center (中心采样 - 取网格中心像素)",
                        "median (中值采样 - 取网格中值像素)"
                    ],
                    value="majority (多数采样 - 取网格中最常见像素)",
                    label="采样方法"
                )
                export_scale = gr.Slider(
                    minimum=1, 
                    maximum=16, 
                    value=4, 
                    step=1, 
                    label="放大倍数"
                )
                add_grid = gr.Checkbox(
                    label="添加网格线", 
                    value=False
                )
            
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 上传图片")
                input_image = gr.Image(
                    label="", 
                    type="numpy",
                    height=300,
                    show_label=False,
                    sources=["upload"]
                )
                
                process_button = gr.Button(
                    "生成像素画", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 输出效果")
                output_image1 = gr.Image(
                    label="像素画效果",
                    height=300,
                    show_label=True
                )
                output_image2 = gr.Image(
                    label="PerfectPixel 优化效果",
                    height=300,
                    show_label=True
                )
        
        process_button.click(
            fn=process_image,
            inputs=[input_image, pixel_size, num_colors, pixel_count, sample_method, export_scale, add_grid],
            outputs=[output_image1, output_image2]
        )
        
        gr.Markdown("## 操作流程")
        gr.Markdown('1. **上传图片**：点击中间的"上传图片"区域，选择要处理的图片')
        gr.Markdown('2. **调整参数**：在左侧调整像素大小、颜色数量等参数')
        gr.Markdown('3. **生成像素画**：点击"生成像素画"按钮，等待处理完成')
        gr.Markdown('4. **查看结果**：在右侧查看像素画效果和PerfectPixel优化效果')
    
    with gr.Tab("实时摄像头"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 参数设置")
                gr.Markdown("- **像素大小**：控制像素块的大小，值越大，像素块越大")
                gr.Markdown("- **颜色数量**：控制颜色的数量，值越小，颜色越少")
                gr.Markdown("- **像素数量**：控制PerfectPixel优化后的像素密度，值越大越精细")
                gr.Markdown("- **采样方法**：选择网格采样的方式")
                gr.Markdown("- **放大倍数**：控制输出图片的放大倍数")
                gr.Markdown("- **添加网格线**：是否在输出图片上添加网格线")
                
                webcam_pixel_size = gr.Slider(
                    minimum=4, 
                    maximum=32, 
                    value=8, 
                    step=1, 
                    label="像素大小"
                )
                webcam_num_colors = gr.Slider(
                    minimum=2, 
                    maximum=32, 
                    value=8, 
                    step=1, 
                    label="颜色数量"
                )
                webcam_pixel_count = gr.Slider(
                    minimum=5, 
                    maximum=100, 
                    value=20, 
                    step=1, 
                    label="像素数量（控制精细度）"
                )
                webcam_sample_method = gr.Dropdown(
                    choices=[
                        "majority (多数采样 - 取网格中最常见像素)",
                        "center (中心采样 - 取网格中心像素)",
                        "median (中值采样 - 取网格中值像素)"
                    ],
                    value="majority (多数采样 - 取网格中最常见像素)",
                    label="采样方法"
                )
                webcam_export_scale = gr.Slider(
                    minimum=1, 
                    maximum=16, 
                    value=4, 
                    step=1, 
                    label="放大倍数"
                )
                webcam_add_grid = gr.Checkbox(
                    label="添加网格线", 
                    value=False
                )
            
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 摄像头")
                webcam_input = gr.Image(
                    label="", 
                    streaming=False, 
                    sources=["webcam"], 
                    type="numpy",
                    height=300,
                    show_label=False
                )
                
                # 处理状态
                is_processing = gr.State(False)
                
                # 控制按钮
                with gr.Row(elem_id="control-buttons"):
                    start_btn = gr.Button(
                        "▶️ 开始转换", 
                        variant="primary",
                        size="md"
                    )
                    stop_btn = gr.Button(
                        "⏸️ 停止转换",
                        size="md"
                    )
                
                with gr.Row(elem_id="camera-buttons"):
                    open_btn = gr.Button(
                        "📷 打开摄像头",
                        size="md"
                    )
                    close_btn = gr.Button(
                        "🔴 关闭摄像头",
                        size="md"
                    )
            
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 输出效果")
                webcam_output1 = gr.Image(
                    label="像素画效果",
                    height=300,
                    show_label=True
                )
                webcam_output2 = gr.Image(
                    label="PerfectPixel 优化效果",
                    height=300,
                    show_label=True
                )
        
        # 开始转换
        start_btn.click(
            fn=start_processing,
            inputs=[],
            outputs=[is_processing]
        )
        
        # 停止转换
        stop_btn.click(
            fn=stop_processing,
            inputs=[],
            outputs=[is_processing]
        )
        
        # 打开摄像头
        open_btn.click(
            fn=handle_open,
            inputs=[],
            outputs=[webcam_input, is_processing]
        )
        
        # 关闭摄像头
        close_btn.click(
            fn=handle_close,
            inputs=[],
            outputs=[webcam_input, is_processing]
        )
        
        # 实时处理
        webcam_input.stream(
            fn=realtime_process,
            inputs=[webcam_input, webcam_pixel_size, webcam_num_colors, webcam_pixel_count, webcam_sample_method, webcam_export_scale, webcam_add_grid, is_processing],
            outputs=[webcam_output1, webcam_output2],
            stream_every=1
        )
        
        gr.Markdown("## 操作流程")
        gr.Markdown('1. **打开摄像头**：点击"Click to Access Webcam"框内（会请求摄像头权限）同意后即可打开摄像头/直接打开摄像头。')
        gr.Markdown('2. **打开录制**：点击"开始录制"按钮开始录制。')
        gr.Markdown('3. **开始转换**：点击"▶️ 开始转换"按钮开始实时处理')
        gr.Markdown('4. **调整参数**：在左侧调整参数查看实时效果')
        gr.Markdown('5. **停止转换**：点击"⏸️ 停止转换"按钮暂停处理')
        gr.Markdown('6. **关闭摄像头**：点击"🔴 关闭摄像头"按钮完全关闭摄像头')
        gr.Markdown('7. **注意**：在转换过程中，摄像头会持续运行，建议在转换完成后关闭摄像头。')
        gr.Markdown('8. **注意**：如果关闭摄像头，在打开摄像头时没有录制权限，需要点击"打开摄像头"按钮重新打开摄像头。')

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7896,
        share=False,
        inbrowser=True
    )