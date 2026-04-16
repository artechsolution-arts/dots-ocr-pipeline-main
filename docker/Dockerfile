FROM vllm/vllm-openai:v0.9.1

COPY setup.py /workspace/dots.ocr/setup.py
COPY requirements.txt /workspace/dots.ocr/requirements.txt
COPY tools/download_model.py /workspace/dots.ocr/tools/download_model.py

# Set working directory
WORKDIR /workspace/dots.ocr

# Install dots.ocr package
RUN pip install -e .
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Copy the dots.ocr source code
COPY . /workspace/dots.ocr/

# Set environment variables
ENV PYTHONPATH=/workspace/dots.ocr:/workspace/dots.ocr/weights:$PYTHONPATH
ENV hf_model_path=/workspace/dots.ocr/weights/DotsOCR

# Copy and setup startup script
COPY start_server.sh /workspace/start_server.sh
RUN chmod +x /workspace/start_server.sh

# Expose port
EXPOSE 8000

RUN pip install fastapi uvicorn requests

# Set entrypoint
ENTRYPOINT ["/workspace/start_server.sh"]