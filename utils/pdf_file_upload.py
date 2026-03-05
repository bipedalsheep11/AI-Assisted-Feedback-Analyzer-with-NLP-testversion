import ipywidgets as widgets
import io
import pandas as pd
import time

_loaded_file = None

def load_file(on_complete):
    uploader = widgets.FileUpload(accept=".pdf", multiple=False)
    button = widgets.Button(description="Load File", button_style="primary")
    output = widgets.Output()

    def on_button_click(b):
        with output:
            output.clear_output()
            if not uploader.value:
                print("⚠️ Please upload a file before proceeding.")
                return
            uploaded_file = uploader.value[0]
            try:
                file = io.BytesIO(bytes(uploaded_file["content"]))
                print("✅ File loaded successfully.")
                on_complete(file)  # trigger next step here
            except Exception as e:
                print(f"⚠️ Failed to read file: {e}")

    button.on_click(on_button_click)
    display(uploader, button, output)

# Usage:
def process_file(file):
    global _loaded_file
    _loaded_file = file
    print(f"Processing file of size: {len(file.read())} bytes")

def get_file():
    print("Start Function")
    _loaded_file.seek(0)
    # Read everything and measure it
    content = _loaded_file.read()
    
    # This will tell us if the BytesIO is empty or has data
    print(f"Number of bytes: {len(content)}")
     # Rewind again so the file is ready for the next operation
    _loaded_file.seek(0)
    return _loaded_file