import numpy as np
try:
    import pyvista as pv
    HAS_PV = True
except ImportError:
    HAS_PV = False
from PIL import Image

class CADRenderer:
    """
    Milestone 8: Renders 12 isometric views of a CAD model for 2D visual feature extraction.
    """
    def __init__(self, num_views: int = 12, resolution: tuple = (224, 224)):
        self.num_views = num_views
        self.resolution = resolution

    def render_views(self, filepath: str) -> list:
        if not HAS_PV:
            print("[WARNING] PyVista not found. Skipping rendering. Install with: pip install pyvista")
            return []

        try:
            mesh = pv.read(filepath)
            
            # Setup off-screen plotter
            plotter = pv.Plotter(off_screen=True, window_size=self.resolution)
            plotter.add_mesh(mesh, color='white', smooth_shading=True)
            plotter.set_background('black')

            images = []
            angles = np.linspace(0, 360, self.num_views, endpoint=False)
            
            for angle in angles:
                plotter.camera_position = 'iso'
                plotter.camera.azimuth = angle
                plotter.render()
                
                # Capture the screen as a numpy array and convert to PIL Image
                img_array = plotter.screenshot(transparent_background=False)
                img = Image.fromarray(img_array).convert('RGB')
                images.append(img)
            
            plotter.close()
            return images
            
        except Exception as e:
            print(f"[ERROR] Rendering failed for {filepath}: {e}")
            return []