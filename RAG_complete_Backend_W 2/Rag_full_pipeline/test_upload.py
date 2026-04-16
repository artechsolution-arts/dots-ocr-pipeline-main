import httpx
import json
import asyncio
import sys
import os

API_URL = "http://localhost:8000"

async def test_upload(pdf_path: str):
    print(f"Testing RAG Pipeline with file: {pdf_path}")
    
    # Upload file
    with open(pdf_path, 'rb') as f:
        files = {'files': (os.path.basename(pdf_path), f, 'application/pdf')}
        data = {'user_id': 'test-user', 'dept_id': 'test-dept'}
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(f"{API_URL}/upload/pdfs", files=files, data=data, timeout=30.0)
                response.raise_for_status()
            except Exception as e:
                print(f"Error connecting to API: {e}")
                return

    result = response.json()
    session_id = result.get('session_id')
    print(f"Upload successful. Session ID: {session_id}")
    print("Listening to Server-Sent Events for progress...")
    print("-" * 50)

    # Listen to progress
    async with httpx.AsyncClient() as client:
        try:
            async with client.stream('GET', f"{API_URL}/upload/progress/{session_id}") as response:
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        data_str = line[6:]
                        try:
                            msg = json.loads(data_str)
                            msg_type = msg.get('type')
                            msg_data = msg.get('data', {})
                            
                            if msg_type == 'file_progress':
                                stage = msg_data.get('stage', 'unknown')
                                pct = msg_data.get('pct', 0)
                                print(f"[{pct:>3}%] \t Stage: {stage.upper()}")
                            elif msg_type == 'session_complete':
                                print("-" * 50)
                                print(f"Session Complete! Status: {msg_data.get('status')}")
                                print(f"✅ Extracted ~{msg_data.get('total_chunks')} chunks successfully.")
                                return
                            elif msg_type == 'ping':
                                continue
                            else:
                                print(f"Unknown message: {msg}")
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Error reading SSE stream: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_upload.py <path_to_pdf>")
        sys.exit(1)
    
    asyncio.run(test_upload(sys.argv[1]))
