import json
import zlib
import base64
import qrcode
import os
import struct

class LatentQREncoder:
    def __init__(self, error_correction=qrcode.constants.ERROR_CORRECT_L):
        # Using ERROR_CORRECT_L (approx 7% error correction) maximizes data capacity
        self.error_correction = error_correction

    def compress_features(self, payload: dict) -> str:
        """
        Extracts the latent vector, packs it into binary, and compresses it.
        """
        # Retrieve the 512-D latent vector
        latent_vector = payload.get("latent_vector", [])
        
        if not latent_vector:
            print("[WARNING] No latent vector provided to QR Encoder. Using zeros.")
            latent_vector = [0.0] * 512

        try:
            # 1. Binary Packing: Pack floats into 16-bit half-precision ('e' format)
            # This cuts the size in half compared to standard 32-bit floats
            pack_format = f'{len(latent_vector)}e'
            binary_data = struct.pack(pack_format, *latent_vector)
            
            # 2. Compress using zlib at maximum compression level (9)
            compressed_bytes = zlib.compress(binary_data, level=9)
            
            # 3. Encode to Base64 (QR codes encode alphanumeric/byte data efficiently)
            encoded_string = base64.b64encode(compressed_bytes).decode('utf-8')
            
            return encoded_string
            
        except struct.error as e:
            print(f"[ERROR] Binary packing failed: {e}. Falling back to JSON compression.")
            # Fallback if the packing format fails for any reason
            json_str = json.dumps(latent_vector, separators=(',', ':'))
            compressed_bytes = zlib.compress(json_str.encode('utf-8'), level=9)
            return base64.b64encode(compressed_bytes).decode('utf-8')

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
        
        print(f"[QR] Embedded {len(encoded_data)} characters of latent space into QR code -> {output_path}")