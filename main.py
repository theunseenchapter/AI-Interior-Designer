from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import torch
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import io
import cv2
from PIL import Image
import base64

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required but not available!")

# Set device to CUDA and use mixed precision for performance
device = "cuda"
dtype = torch.float16
torch.backends.cudnn.benchmark = True

# Load Stable Diffusion inpainting model on GPU
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=dtype
).to(device)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware (optional but helpful for cross-domain requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="C:/Users/Vivek Yadav/Desktop/pr/static"), name="static")

# Replace the frontend HTML with your provided code
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Home Alchemy</title>
    <style>
            * {
         margin: 0;
         padding: 0;
         box-sizing: border-box;
         font-family: Arial, sans-serif;
       }
       body {
               display: flex;
               align-items: center;
               justify-content: center;
               height: 100vh;
               background: url('/static/pexels-santesson89-10260287.jpg') no-repeat center center;
               background-size: cover;
           }
           .container {
         width: 80%;
         max-width: 1000px;
         background: rgba(255, 255, 255, 0.1);
         padding: 20px;
         border-radius: 15px;
         box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
         text-align: center;
         backdrop-filter: blur(10px);
         display: flex;
         justify-content: space-between;
         align-items: flex-start;
       }
       .form-container {
         width: 30%;
         text-align: left;
         padding-right: 20px;
       }
       h1 {
         text-align: center;
         font-size: 28px;
         color: #f9f9f9;
         margin-bottom: 20px;
         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
       }
       .form-group {
         margin-bottom: 15px;
       }
       label {
         display: block;
         font-weight: bold;
         margin-bottom: 5px;
         color: #f0f0f0;
         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
       }
       select,
       input[type="file"] {
         width: 100%;
         padding: 10px;
         border-radius: 5px;
         border: none;
         font-size: 16px;
       }
       button {
         width: 100%;
         padding: 12px;
         font-size: 18px;
         border: none;
         border-radius: 5px;
         cursor: pointer;
         background: linear-gradient(to right, #6a11cb, #2575fc);
         color: #fff;
         transition: 0.3s;
       }
       button:hover {
         opacity: 0.8;
       }
       .preview-container {
         width: 65%;
         display: flex;
         justify-content: space-between;
       }
       .image-section {
         width: 48%;
         text-align: center;
         padding: 20px;
         background: rgba(255, 255, 255, 0.2);
         border-radius: 10px;
         backdrop-filter: blur(8px);
       }
       .image-section h3 {
         margin-bottom: 10px;
         color: #f9f9f9;
         font-weight: bold;
         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
       }
       .image-section img {
         width: 100%;
         border-radius: 5px;
         margin-top: 10px;
         display: none;
       }
    </style>
</head>
<body>
    <div class="container">
        <div class="form-container">
            <h1>Home Alchemy</h1>
            <form id="uploadForm">
                <div class="form-group">
                    <label for="roomImage">Upload Room Image:</label>
                    <input type="file" id="roomImage" accept="image/*" required>
                </div>
                <div class="form-group">
                    <label for="style">Choose Style:</label>
                    <select id="style" required>
                        <option value="modern minimalist">Modern Minimalist</option>
                        <option value="scandinavian">Scandinavian</option>
                        <option value="industrial">Industrial</option>
                        <option value="contemporary">Contemporary</option>
                        <option value="bohemian">Bohemian</option>
                    </select>
                </div>
                <button type="submit">Generate Design</button>
            </form>
        </div>

        <div class="preview-container">
            <div class="image-section">
                <h3>Original Image</h3>
                <img id="uploadedPreview">
            </div>
            <div class="image-section">
                <h3>Generated Image</h3>
                <img id="generatedPreview">
            </div>
        </div>
    </div>

    <script>
        document.getElementById('roomImage').addEventListener('change', function(e) {
            const preview = document.getElementById('uploadedPreview');
            const file = e.target.files[0];
            if (file) {
                preview.style.display = 'block';
                preview.src = URL.createObjectURL(file);
            }
        });

        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData();
            const fileInput = document.getElementById('roomImage');
            const style = document.getElementById('style').value;

            if (!fileInput.files[0]) {
                alert('Please select an image first');
                return;
            }

            formData.append('file', fileInput.files[0]);
            formData.append('style', style);

            try {
                const response = await fetch('/upload/', { method: 'POST', body: formData });
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                if (data.error) {
                    document.getElementById('generatedPreview').style.display = 'none';
                    alert(`Error: ${data.error}`);
                } else {
                    const generatedPreview = document.getElementById('generatedPreview');
                    generatedPreview.src = 'data:image/png;base64,' + data.design;
                    generatedPreview.style.display = 'block';
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        });
    </script>
</body>
</html>

"""

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the HTML frontend."""
    return HTML_CONTENT

def create_detailed_mask(image):
    """Creates a mask that focuses on the center of the room."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Create a circular mask in the center
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 3
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    return mask

def generate_design(image, mask, style_prompt):
    """Generates a new interior design using Stable Diffusion Inpainting."""
    prompt = f"interior design, {style_prompt}, professional, photorealistic, high quality"
    negative_prompt = "low quality, blurry, bad architecture"

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)

    target_size = (512, 512)
    image = image.resize(target_size, Image.LANCZOS).convert("RGB")
    mask = mask.resize(target_size, Image.NEAREST).convert("L")

    # Perform inference with mixed precision
    with torch.no_grad(), torch.autocast(device):
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=100,  # More steps for better quality
            guidance_scale=15.0       # Higher guidance for more accurate results
        ).images[0]

    return output

@app.post("/upload/")
async def upload_room_photo(file: UploadFile = File(...), style: str = "modern minimalist"):
    try:
        # Read the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate a mask based on the center of the image
        mask = create_detailed_mask(image_rgb)

        # Generate the new design using the inpainting pipeline
        result = generate_design(image_rgb, mask, style)

        # Convert the result to Base64 for sending back to the client
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()

        return {"message": "Design generated successfully", "design": img_base64}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
