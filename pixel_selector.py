#!/usr/bin/env python3
"""
Pixel Selector Tool
Opens an image, resizes it to 704x1280, and allows clicking to get pixel coordinates.
Useful for selecting target pixels for pixel focusing trajectory methods.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

class PixelSelector:
    def __init__(self, image_path, target_width=1280, target_height=704):
        self.image_path = image_path
        self.target_width = target_width
        self.target_height = target_height
        self.clicked_pixel = None
        self.fig = None
        self.ax = None
        
    def load_and_resize_image(self):
        """Load image and resize to target dimensions."""
        # Load image
        if not Path(self.image_path).exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
            
        # Read image
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Could not read image: {self.image_path}")
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get original dimensions
        orig_height, orig_width = img_rgb.shape[:2]
        print(f"Original image size: {orig_width} x {orig_height}")
        
        # Resize to target dimensions
        img_resized = cv2.resize(img_rgb, (self.target_width, self.target_height))
        print(f"Resized image to: {self.target_width} x {self.target_height}")
        
        return img_resized
    
    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < self.target_width and 0 <= y < self.target_height:
                self.clicked_pixel = (x, y)
                print(f"\n🎯 Clicked pixel coordinates: ({x}, {y})")
                print(f"   X: {x}, Y: {y}")
                plt.close(self.fig)
            else:
                print(f"⚠️  Click outside image bounds: ({x}, {y})")
    
    def select_pixel(self):
        """Display image and wait for pixel selection."""
        # Load and resize image
        img_resized = self.load_and_resize_image()
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Display image
        self.ax.imshow(img_resized)
        self.ax.set_title(f"Click on a pixel to select coordinates\nImage: {Path(self.image_path).name}\nSize: {self.target_width} x {self.target_height}")
        self.ax.set_xlabel(f"X coordinate (0 to {self.target_width-1})")
        self.ax.set_ylabel(f"Y coordinate (0 to {self.target_height-1})")
        
        # Add grid for easier pixel selection
        self.ax.grid(True, alpha=0.3)
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Show instructions
        print(f"\n📱 Image displayed. Click on any pixel to get coordinates.")
        print(f"   Target dimensions: {self.target_width} x {self.target_height}")
        print(f"   Click anywhere on the image to continue...")
        
        # Display plot
        plt.tight_layout()
        plt.show()
        
        return self.clicked_pixel

def main():
    parser = argparse.ArgumentParser(description="Pixel Selector Tool - Click to get pixel coordinates")
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to the input image"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Target width (default: 1280)"
    )
    parser.add_argument(
        "--height", 
        type=int,
        default=704,
        help="Target height (default: 704)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create pixel selector
        selector = PixelSelector(args.image_path, args.width, args.height)
        
        # Select pixel
        pixel_coords = selector.select_pixel()
        
        if pixel_coords:
            x, y = pixel_coords
            print(f"\n✅ Final selected pixel coordinates:")
            print(f"   X: {x}")
            print(f"   Y: {y}")
            print(f"   Format: ({x}, {y})")
            
            # Save coordinates to file for easy copy-paste
            output_file = "selected_pixel_coordinates.txt"
            with open(output_file, "w") as f:
                f.write(f"Selected pixel coordinates:\n")
                f.write(f"X: {x}\n")
                f.write(f"Y: {y}\n")
                f.write(f"Format: ({x}, {y})\n")
                f.write(f"Image: {args.image_path}\n")
                f.write(f"Target size: {args.width} x {args.height}\n")
            
            print(f"\n💾 Coordinates saved to: {output_file}")
            
        else:
            print("\n❌ No pixel was selected.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 