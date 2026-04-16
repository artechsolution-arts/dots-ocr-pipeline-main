import os
import sys
from dots_ocr.parser import DotsOCRParser
from PIL import Image

def main():
    # Model path is default to ./weights/DotsOCR which we confirmed exists
    # Use HF mode for local inference on Mac (MPS)
    print("Initializing DotsOCRParser...")
    parser = DotsOCRParser(use_hf=True)
    
    test_image_path = "assets/showcase_origin/formula_1.jpg"
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}")
        return

    print(f"Running inference on {test_image_path}...")
    # parse_image returns a list of results
    results = parser.parse_image(
        input_path=test_image_path,
        filename="test_result",
        prompt_mode="prompt_layout_all_en",
        save_dir="./output/test_run"
    )
    
    if results:
        print("Inference successful!")
        result = results[0]
        if 'md_content_path' in result:
            with open(result['md_content_path'], 'r') as f:
                content = f.read()
                print("\nMetadata content preview:")
                print("-" * 20)
                print(content[:500] + "...")
                print("-" * 20)
    else:
        print("Inference failed to return results.")

if __name__ == "__main__":
    main()
