import os
import sys
import subprocess
import json
import shutil
import threading
import queue
import time
from pathlib import Path
import gradio as gr

# Setup default paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WAN_REPO = "/teamspace/studios/this_studio/Wan2.2"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_KNOWLEDGE_DIR = SCRIPT_DIR / "knowledge"

# Task queue for log streaming
log_queue = queue.Queue()
active_process = None
is_paused = False
paused_expected_path = None

def log_reader_thread(proc):
    """Reads stdout and stderr of the pipeline process and puts them in a queue."""
    global is_paused, paused_expected_path
    
    # Read both streams line-by-line
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        
        # Put raw log line to queue
        log_queue.put(line)
        
        # Detect pause state
        if "⏸️  PIPELINE PAUSED" in line:
            is_paused = True
        elif "Please place your generated image at:" in line:
            # Extract expected file path from log line
            try:
                parts = line.split("Please place your generated image at:")
                if len(parts) > 1:
                    paused_expected_path = Path(parts[1].strip())
            except Exception:
                pass

def start_pipeline(query, knowledge_file, preset, experiment_name, custom_args):
    """Spawns the main.py pipeline subprocess."""
    global active_process, is_paused, paused_expected_path
    
    if active_process is not None and active_process.poll() is None:
        return "Pipeline is already running!", gr.update(visible=False), gr.update(visible=False)
        
    is_paused = False
    paused_expected_path = None
    
    # Clear queue
    while not log_queue.empty():
        try:
            log_queue.get_nowait()
        except queue.Empty:
            break
            
    # Build command
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "main.py"),
        "--generator-script", "wan_local_wrapper.py",
        "--wan-repo-dir", DEFAULT_WAN_REPO,
        "--wan-model-preset", preset,
        "--experiment-name", experiment_name
    ]
    
    if query:
        cmd.extend(["--query", query])
    elif knowledge_file:
        cmd.extend(["--knowledge-file", knowledge_file.name])
        
    if custom_args:
        cmd.extend(custom_args.split())
        
    log_queue.put(f"🚀 Launching command: {' '.join(cmd)}\n\n")
    
    try:
        active_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Redirect stderr to stdout to catch all logs
            text=True,
            bufsize=1,
            cwd=str(SCRIPT_DIR)
        )
        
        # Start thread to read outputs asynchronously
        threading.Thread(target=log_reader_thread, args=(active_process,), daemon=True).start()
        
        return "Starting generation...", gr.update(visible=False), gr.update(visible=False)
    except Exception as e:
        return f"Failed to start pipeline: {e}", gr.update(visible=False), gr.update(visible=False)

def stop_pipeline():
    """Terminates the pipeline and cleans up."""
    global active_process, is_paused
    if active_process is not None:
        try:
            active_process.terminate()
            active_process.wait(timeout=2)
        except Exception:
            try:
                active_process.kill()
            except Exception:
                pass
        active_process = None
    is_paused = False
    return "Pipeline terminated.", gr.update(visible=False), gr.update(visible=False)

def poll_logs(current_logs):
    """Gradio generator to stream logs and monitor pause/resume triggers."""
    global is_paused, paused_expected_path, active_process
    
    logs = current_logs or ""
    
    # Stream from queue
    while not log_queue.empty():
        try:
            line = log_queue.get_nowait()
            logs += line
        except queue.Empty:
            break
            
    # Check if process finished
    status_msg = "Running..."
    is_finished = False
    if active_process is not None:
        ret = active_process.poll()
        if ret is not None:
            status_msg = f"Completed (Exit Code: {ret})"
            active_process = None
            is_finished = True
            is_paused = False
            
    # Show/hide file uploader based on pause state
    uploader_update = gr.update(visible=is_paused)
    resume_btn_update = gr.update(visible=is_paused)
    
    # Check for final video file if completed
    video_update = gr.update()
    if is_finished:
        # Scan for final_video.mp4 inside the outputs folder
        for root, dirs, files in os.walk(str(DEFAULT_OUTPUT_DIR)):
            if "final_video.mp4" in files:
                video_path = Path(root) / "final_video.mp4"
                video_update = gr.update(value=str(video_path), visible=True)
                break
                
    return logs, status_msg, uploader_update, resume_btn_update, video_update

def handle_resume(uploaded_file):
    """Copies uploaded manual image to the expected path and resumes the pipeline."""
    global is_paused, paused_expected_path, active_process
    
    if not is_paused or paused_expected_path is None or active_process is None:
        return "Not currently paused or no path specified.", gr.update(visible=is_paused), gr.update(visible=is_paused)
        
    if uploaded_file is None:
        return "Please upload an image first!", gr.update(visible=True), gr.update(visible=True)
        
    try:
        # Copy file to expected destination
        dest = Path(paused_expected_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(uploaded_file.name, dest)
        log_queue.put(f"\n[GUI] Manually placed image at: {dest}\n")
        
        # Send Enter (\n) to subprocess stdin to trigger resume
        active_process.stdin.write("\n")
        active_process.stdin.flush()
        
        is_paused = False
        return "Resuming generation...", gr.update(visible=False), gr.update(visible=False)
    except Exception as e:
        log_queue.put(f"\n[GUI Error] Failed to place image: {e}\n")
        return f"Error: {e}", gr.update(visible=True), gr.update(visible=True)

# Define Custom Sleek Dark-Mode CSS for presentation
theme_css = """
body {
    background-color: #0d0d11 !important;
    color: #e4e4e7 !important;
    font-family: 'Outfit', sans-serif !important;
}
.gradio-container {
    background: radial-gradient(circle at top, #1e1e2d 0%, #0d0d11 100%) !important;
    border: 1px solid #2d2d3d !important;
    border-radius: 16px !important;
    padding: 24px !important;
}
button.primary {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4) !important;
    transition: all 0.2s ease !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.6) !important;
}
.console-box textarea {
    background-color: #08080a !important;
    color: #00ff66 !important;
    font-family: 'Fira Code', monospace !important;
    border: 1px solid #1f2937 !important;
}
"""

with gr.Blocks(title="Scientific Video Generation Simulator") as demo:
    gr.HTML(
        """
        <div style="text-align: center; margin-bottom: 24px;">
            <h1 style="color: #6366f1; font-size: 2.5rem; margin-bottom: 8px; font-weight: 800;">🔬 AI Scientific Laboratory Simulator</h1>
            <p style="color: #a1a1aa; font-size: 1.1rem;">Generate high-continuity multi-scene chemistry videos with Wan 2.2 and dynamic offloading.</p>
        </div>
        """
    )
    
    with gr.Row():
        # Column 1: Configuration & Control
        with gr.Column(scale=2):
            gr.Markdown("### ⚙️ Generation Control Panel")
            
            with gr.Tab("RAG Query (Auto)"):
                query_input = gr.Textbox(
                    label="Natural Language Experiment Query",
                    placeholder="e.g., Flame test with platinum loop showing crimson red flame",
                    value="Newton's Cradle"
                )
                
            with gr.Tab("Structured JSON (Manual)"):
                json_file_input = gr.File(
                    label="Upload Manual Experiment JSON Config",
                    file_types=[".json"]
                )
                
            model_preset = gr.Dropdown(
                choices=["i2v-a14b", "ti2v-5b"],
                value="i2v-a14b",
                label="Wan 2.2 Model Preset (Dynamic Offloading)"
            )
            
            exp_name = gr.Textbox(
                label="Experiment Name",
                value="salt_analysis_gpu",
                placeholder="Folder classification name"
            )
            
            extra_args_input = gr.Textbox(
                label="Custom CLI Arguments (Optional)",
                placeholder="e.g. --initial-image path/to/image.png"
            )
            
            with gr.Row():
                run_btn = gr.Button("🚀 Run Pipeline", variant="primary")
                stop_btn = gr.Button("⏹️ Stop Generation", variant="stop")
                
            gr.Markdown("---")
            status_display = gr.Label(value="Idle", label="System Status")

        # Column 2: Terminal Logs & Interrupt Handlers
        with gr.Column(scale=3):
            gr.Markdown("### 🖥️ Real-time Processing Console")
            log_display = gr.Textbox(
                label="Output Logs",
                value="",
                lines=20,
                max_lines=30,
                elem_classes=["console-box"],
                interactive=False
            )
            
            # Interactive manual override components (initially hidden)
            uploader_comp = gr.File(
                label="⏸️ Upload Manual Starting Image for Paused Clip",
                file_types=["image"],
                visible=False
            )
            resume_btn = gr.Button("▶️ Place Image & Resume Pipeline", variant="primary", visible=False)
            action_status = gr.Markdown("")

        # Column 3: Outputs Preview
        with gr.Column(scale=2):
            gr.Markdown("### 🎬 Stitched Compilation Result")
            video_display = gr.Video(label="Final Stitched Output", visible=False)
            
    # Event wiring
    run_btn.click(
        fn=start_pipeline,
        inputs=[query_input, json_file_input, model_preset, exp_name, extra_args_input],
        outputs=[status_display, uploader_comp, resume_btn]
    )
    
    stop_btn.click(
        fn=stop_pipeline,
        outputs=[status_display, uploader_comp, resume_btn]
    )
    
    resume_btn.click(
        fn=handle_resume,
        inputs=[uploader_comp],
        outputs=[status_display, uploader_comp, resume_btn]
    )
    
    # Auto-polling loop to stream logs to terminal (polls every 1 second using Gradio 6 Timer)
    timer = gr.Timer(1.0)
    timer.tick(
        fn=poll_logs,
        inputs=[log_display],
        outputs=[log_display, status_display, uploader_comp, resume_btn, video_display]
    )

if __name__ == "__main__":
    demo.queue()
    # Share set to False in standard studio, runs locally on port 7860
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=theme_css
    )
