import gradio as gr
import os
from PIL import Image
from dots_ocr.parser import DotsOCRParser
import json

# Initialize the parser with Mac GPU support
parser = DotsOCRParser(use_hf=True)

def process_document_detailed(file_path):
    if not file_path:
        return None, "Please upload a file.", None
    
    try:
        # Parse the file (handles PDF and images)
        results = parser.parse_file(file_path, prompt_mode="prompt_layout_all_en", fitz_preprocess=True)
        
        if not results:
            return None, "No results found.", None
        
        # Take the first page for the simple demo
        result = results[0]
        
        # Load the visualization image
        layout_image = Image.open(result['layout_image_path'])
        
        # Load the markdown content
        with open(result['md_content_path'], 'r') as f:
            md_content = f.read()
            
        # Load the JSON data
        with open(result['layout_info_path'], 'r') as f:
            json_data = json.load(f)
            
        return layout_image, md_content, json_data
    except Exception as e:
        return None, f"Error: {str(e)}", None

# Custom CSS for a premium look
css = """
.container { max-width: 1200px; margin: auto; }
.header { text-align: center; margin-bottom: 2rem; }
.result-box { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; background: #fafafa; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.HTML("<div class='header'><h1>🔍 DotsOCR: Quick Document Parser</h1><p>Upload a document to see OCR and Layout analysis in action</p></div>")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload Image or PDF", file_types=[".pdf", ".jpg", ".jpeg", ".png"])
            run_btn = gr.Button("🚀 Run OCR", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Visual Layout"):
                    image_output = gr.Image(label="Detected Layout", interactive=False)
                with gr.Tab("Extracted Text (Markdown)"):
                    text_output = gr.Markdown(label="OCR Result")
                with gr.Tab("Raw JSON Data"):
                    json_output = gr.JSON(label="Detailed Analysis")

    run_btn.click(
        fn=process_document_detailed,
        inputs=[file_input],
        outputs=[image_output, text_output, json_output]
    )
    
    gr.Examples(
        examples=["demo/demo_image1.jpg"],
        inputs=file_input
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8081)
