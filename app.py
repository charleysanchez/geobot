"""
GeoBot Interactive Demo
Run with: python app.py
"""
import torch
import gradio as gr
from PIL import Image
import folium
from io import BytesIO
import base64

from src.model import ImageToGeoModel
from src.dataset import get_val_transform


# Global model variable
model = None
device = None
transform = None


def load_model(checkpoint_path="checkpoint.pt"):
    """Load the trained model."""
    global model, device, transform
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone_name = checkpoint.get('backbone_name', 'mobilenetv3_large_100')
    
    model = ImageToGeoModel(backbone_name=backbone_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    transform = get_val_transform()
    
    print(f"Model loaded: {backbone_name} on {device}")
    return True


def create_map_html(lat, lon):
    """Create an embedded map centered on the prediction."""
    m = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker(
        [lat, lon],
        popup=f"Predicted: {lat:.4f}°, {lon:.4f}°",
        icon=folium.Icon(color='red', icon='map-marker', prefix='fa')
    ).add_to(m)
    
    # Add a circle to show uncertainty
    folium.Circle(
        [lat, lon],
        radius=100000,  # 100km radius
        color='red',
        fill=True,
        fill_opacity=0.1
    ).add_to(m)
    
    return m._repr_html_()


def predict(image):
    """Make prediction on uploaded image."""
    if model is None:
        return "Error: Model not loaded!", None
    
    if image is None:
        return "Please upload an image.", None
    
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    image = image.convert("RGB")
    
    # Transform and predict
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
    
    # Denormalize
    lat = output[0, 0].item() * 90
    lon = output[0, 1].item() * 180
    
    # Format result
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    result_text = f"""
## 📍 Predicted Location

**Latitude:** {abs(lat):.4f}° {lat_dir}  
**Longitude:** {abs(lon):.4f}° {lon_dir}

[Open in Google Maps](https://www.google.com/maps?q={lat},{lon})
"""
    
    # Create map
    map_html = create_map_html(lat, lon)
    
    return result_text, map_html


# Create Gradio interface
def create_demo():
    with gr.Blocks(title="GeoBot - Image Geolocation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🌍 GeoBot
### Predict geographic coordinates from Street View images
        
Upload a street-level photograph and GeoBot will predict where in the world it was taken.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=400
                )
                predict_btn = gr.Button("🔍 Predict Location", variant="primary", size="lg")
                
                gr.Markdown("""
### Tips for best results:
- Use street-level outdoor images
- Google Street View screenshots work great
- Avoid indoor photos or aerial views
                """)
            
            with gr.Column(scale=1):
                result_text = gr.Markdown(label="Prediction")
                map_output = gr.HTML(label="Map")
        
        # Example images
        gr.Markdown("### Try these examples:")
        gr.Examples(
            examples=[
                "dataset/images/0.png",
                "dataset/images/100.png",
                "dataset/images/500.png",
            ],
            inputs=image_input,
            label="Sample Images"
        )
        
        # Connect the button
        predict_btn.click(
            fn=predict,
            inputs=[image_input],
            outputs=[result_text, map_output]
        )
        
        # Also predict when image is uploaded
        image_input.change(
            fn=predict,
            inputs=[image_input],
            outputs=[result_text, map_output]
        )
    
    return demo


if __name__ == "__main__":
    # Load model
    print("Loading model...")
    load_model("checkpoint.pt")
    
    # Create and launch demo
    demo = create_demo()
    demo.launch(
        share=False,  # Set to True to get a public link
        server_name="0.0.0.0",
        server_port=7860
    )
