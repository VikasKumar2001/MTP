import os
import shutil

# -----------------------------------------
# Configuration
# -----------------------------------------

SOURCE_DIR = r"C:\Users\kvika\Desktop\MTP\encoder_final_V1\results"
OUTPUT_DIR = "all_qr"

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------
# Find and copy QR codes
# -----------------------------------------

count = 0

for folder_name in os.listdir(SOURCE_DIR):

    folder_path = os.path.join(
        SOURCE_DIR,
        folder_name
    )

    # Only process directories
    if not os.path.isdir(folder_path):
        continue

    qr_path = os.path.join(
        folder_path,
        "latent_qr.png"
    )

    # Check if QR exists
    if os.path.isfile(qr_path):

        # Save using model/folder name
        output_path = os.path.join(
            OUTPUT_DIR,
            f"{folder_name}.png"
        )

        shutil.copy2(
            qr_path,
            output_path
        )

        print(
            f"[COPIED] {qr_path} -> {output_path}"
        )

        count += 1


print()
print("=" * 50)
print(f"Total QR codes copied: {count}")
print(f"Saved in: {OUTPUT_DIR}")
print("=" * 50)