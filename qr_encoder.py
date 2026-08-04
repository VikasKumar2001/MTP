import json
import zlib
import base64
import qrcode #
import os

class LatentQREncoder:
    def __init__(self, error_correction=qrcode.constants.ERROR_CORRECT_L):
        # Using ERROR_CORRECT_L (approx 7% error correction) maximizes data capacity
        self.error_correction = error_correction

    def compress_features(self, feature_data: dict) -> str:
        """Strips, compresses, and encodes the feature payload."""
        
        # 1. Isolate only the data the Generative AI needs to reconstruct the shape
        # We use single-character keys to save JSON string space
        latent_payload = {
            "v": feature_data.get("feature_vector", []),
            "s": feature_data.get("slice_signature", []),
            "t": feature_data.get("chair_type", "unknown")
        }

        # 2. Convert to string with zero whitespace
        json_str = json.dumps(latent_payload, separators=(',', ':'))
        
        # 3. Compress using zlib
        compressed_bytes = zlib.compress(json_str.encode('utf-8'))
        
        # 4. Encode to Base64 (more space-efficient than hex for QR codes)
        encoded_string = base64.b64encode(compressed_bytes).decode('utf-8')
        
        return encoded_string

    def generate_qr(self, encoded_data: str, output_path: str):
        """Generates and saves the QR code image."""
        qr = qrcode.QRCode(
            version=None, # Auto-size based on data length
            error_correction=self.error_correction,
            box_size=10,
            border=4,
        )
        
        qr.add_data(encoded_data)
        qr.make(fit=True)

        # Generate and save the image
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(output_path)
        
        print(f"[QR] Compressed {len(encoded_data)} characters into QR code -> {output_path}")

# --- Example Usage ---
if __name__ == "__main__":
    encoder = LatentQREncoder()
    
    # Simulating loading from your main.py pipeline
    feature_path = "results/sample_chair/features.json"
    
    if os.path.exists(feature_path):
        with open(feature_path, 'r') as f:
            features = json.load(f)
            
        encoded_payload = encoder.compress_features(features)
        encoder.generate_qr(encoded_payload, "results/sample_chair/latent_qr.png")