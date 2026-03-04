import ipywidgets as widgets
import io
import pandas as pd
import time

def load_survey_data() -> pd.DataFrame:
    uploader = widgets.FileUpload(accept=".csv", multiple=False)
    button = widgets.Button(description="Load File", button_style="primary")
    output = widgets.Output()
    result = {"df": None}

    def on_button_click(b):
        with output:
            output.clear_output()

            # Guard: check if file was uploaded
            if not uploader.value:
                print("⚠️ Please upload a file before proceeding.")
                return

            uploaded_file = uploader.value[0]

            # # Guard: check file extension
            # filename = uploaded_file["name"]
            # if not filename.lower().endswith(".csv"):
            #     print(f"⚠️ Invalid file type: '{filename}'. Please upload a CSV file.")
            #     return

            # # Guard: check MIME type as secondary validation
            # mime_type = uploaded_file.get("type", "")
            # if mime_type and "csv" not in mime_type and "text" not in mime_type:
            #     print(f"⚠️ Unexpected file type: '{mime_type}'. Please upload a CSV UTF-8 file.")
            #     return

            try:
                file = io.BytesIO(bytes(uploaded_file["content"]))
                dataframe = pd.read_csv(file, encoding="utf-8")
                print(f"✅ Loaded {dataframe.shape[0]} rows and {dataframe.shape[1]} columns.")
                dataframe = dataframe.iloc[:, 1:]
                display(dataframe.head(3))
                result["df"] = dataframe
            except UnicodeDecodeError:
                print("⚠️ Encoding error: Please ensure the file is saved as CSV UTF-8.")
            except Exception as e:
                print(f"⚠️ Failed to read file: {e}")

    button.on_click(on_button_click)
    display(uploader, button, output)

    # Block execution until the dataframe is loaded
    while result["df"] is None:
        time.sleep(0.5)

    return result["df"]