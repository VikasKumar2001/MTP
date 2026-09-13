import cv2
import base64
import zlib
import struct
import json
import os

class LatentQRDecoder:
    """
    Milestone 10: Decodes a QR image back into the 512-D latent representation.
    Reverses the Base64 -> Zlib -> Binary packing process.
    """
    def __init__(self, expected_dim: int = 512):
        self.expected_dim = expected_dim

    def _read_qr_image(self, image_path: str) -> str:
        """Reads the QR code image and extracts the encoded string."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"QR code image not found at: {image_path}")

        # 1. Try OpenCV's built-in QR Code Detector
        img = cv2.imread(image_path)
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(img)
        
        # 2. Fallback to pyzbar if OpenCV fails (pyzbar is often more robust)
        if not data:
            try:
                from pyzbar.pyzbar import decode
                from PIL import Image
                decoded_objects = decode(Image.open(image_path))
                if decoded_objects:
                    data = decoded_objects[0].data.decode('utf-8')
            except ImportError:
                print("[WARNING] pyzbar not installed. OpenCV failed to detect QR. Try: pip install pyzbar")
                
        if not data:
            raise ValueError("No QR code detected or decoded from the image.")
            
        return data

    def _decompress_features(self, encoded_string: str) -> list:
        """Reverses the compression pipeline back to the float array."""
        try:
            # 1. Base64 Decode
            compressed_bytes = base64.b64decode(encoded_string)
            
            # 2. Zlib Decompress
            binary_data = zlib.decompress(compressed_bytes)
            
            # 3. Binary Unpacking: 16-bit half-floats ('e')
            pack_format = f'{self.expected_dim}e'
            expected_bytes = struct.calcsize(pack_format)
            
            if len(binary_data) == expected_bytes:
                # Successfully unpacked binary floats
                latent_vector = struct.unpack(pack_format, binary_data)
                return [round(float(v), 6) for v in latent_vector]
            else:
                # Fallback: If the encoder used the JSON fallback method
                json_str = binary_data.decode('utf-8')
                return json.loads(json_str)

        except Exception as e:
            print(f"[ERROR] Decoding payload failed: {e}")
            return [0.0] * self.expected_dim

    def decode(self, image_path: str) -> list:
        """
        Main pipeline execution:
        Image Path -> Read QR -> Decode Base64 -> Decompress -> Unpack -> Latent Vector
        """
        try:
            print(f"[QR Decoder] Scanning {image_path}...")
            encoded_data = self._read_qr_image(image_path)
            
            latent_vector = self._decompress_features(encoded_data)
            
            print(f"[QR Decoder] Successfully recovered {len(latent_vector)}-D latent vector.")
            return latent_vector
            
        except Exception as e:
            print(f"[ERROR] {e}")
            return [0.0] * self.expected_dim


# --- Example Usage / Standalone Testing ---
if __name__ == "__main__":
    # Test the decoder on a previously generated QR code
    test_qr_path = "results/test_latent_qr.png" 
    
    decoder = LatentQRDecoder(expected_dim=512)
    
    if os.path.exists(test_qr_path):
        recovered_vector = decoder.decode(test_qr_path)
        print(f"Sample of recovered vector: {recovered_vector[:5]} ...")
    else:
        print(f"Generate a QR code first using your main.py pipeline to test the decoder.")