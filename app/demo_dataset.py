import sys
import os
import argparse
import customtkinter as ctk
from PIL import ImageDraw
import cv2
import numpy as np

# add parent of this file to path to enable importing
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets.data_loader import DataLoader
from datasets.data_structure import Attribute
from face_detection.face_detection_engine import FaceDetectionEngine


class App(ctk.CTk):
    def __init__(self, data_path):
        super().__init__()
        ctk.set_appearance_mode("light")
        self.grid_columnconfigure(1, weight=1)
        
        self.data_path = data_path
        self.data_loader = DataLoader(data_path)
        self.face_detection_engine = None
        self.index = -1


        self.title("Face identification demo")
        self.geometry(f"{720}x{600}")

        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)
        title = ctk.CTkLabel(self.sidebar_frame, text="Face identification", font=ctk.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 10))

        next_button = ctk.CTkButton(self.sidebar_frame, text="Next image", command=self.next_image)
        next_button.grid(row=1, column=0, padx=20, pady=10)
        detect_button = ctk.CTkButton(self.sidebar_frame, text="Detect face", command=self.detect_face)
        detect_button.grid(row=2, column=0, padx=20, pady=10)
        
        
        reload_button = ctk.CTkButton(self.sidebar_frame, text="Reload", command=self.reload_data)
        reload_button.grid(row=3, column=0, padx=20, pady=(20, 10))
        
        filter_label = ctk.CTkLabel(self.sidebar_frame, text="Filter by class (0 = ALL)", font=ctk.CTkFont(size=14))
        filter_label.grid(row=4, column=0, padx=20, pady=(20, 5))
        self.filter_class = ctk.CTkEntry(self.sidebar_frame)
        self.filter_class.grid(row=5, column=0, padx=20, pady=5)
        self.filter_class.insert(0, "0")
        
        filter_attr_label = ctk.CTkLabel(self.sidebar_frame, text="Filter by attribute", font=ctk.CTkFont(size=14))
        filter_attr_label.grid(row=6, column=0, padx=20, pady=0)
        self.scrollable_frame = ctk.CTkScrollableFrame(self.sidebar_frame, width=140, height=250)
        self.scrollable_frame.grid(row=7, column=0, padx=20, pady=0, sticky="nsew")
        
        self.chk_attributes = {}
        for i, attr in enumerate(Attribute):
            var = ctk.BooleanVar()
            chk = ctk.CTkCheckBox(self.scrollable_frame, text=attr.name, variable=var, font=ctk.CTkFont(size=10))
            chk.grid(row=i, column=0, padx=20, pady=(0, 5), sticky="w")
            self.chk_attributes[attr.name] = var


        image_frame = ctk.CTkFrame(self, fg_color="transparent")
        image_frame.grid(row=0, column=1, padx=(20, 20), sticky="nsew")
        
        self.image = ctk.CTkLabel(image_frame, text="")
        self.image.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        
        attributes_frame = ctk.CTkFrame(self, fg_color="gray", corner_radius=0)
        attributes_frame.grid(row=0, column=2, rowspan=4, sticky="nsew")
        attributes_frame.grid_rowconfigure(6, weight=1)
        
        filename_label = ctk.CTkLabel(attributes_frame, text="Filename", font=ctk.CTkFont(weight="bold"), width=200)
        filename_label.grid(row=0, column=0, padx=20, pady=(10, 5))
        self.filename = ctk.CTkLabel(attributes_frame)
        self.filename.grid(row=1, column=0, padx=20, pady=(0, 10))

        class_label = ctk.CTkLabel(attributes_frame, text="Class", font=ctk.CTkFont(weight="bold"))
        class_label.grid(row=2, column=0, padx=20, pady=(10, 5))
        self.classID = ctk.CTkLabel(attributes_frame)
        self.classID.grid(row=3, column=0, padx=20, pady=(0, 10))

        attributes_label = ctk.CTkLabel(attributes_frame, text="Attributes", font=ctk.CTkFont(weight="bold"))
        attributes_label.grid(row=4, column=0, padx=20, pady=(10, 5))
        self.attributes = ctk.CTkLabel(attributes_frame)
        self.attributes.grid(row=5, column=0, padx=20, pady=(0, 10))
        
        self.next_image()
        

    def next_image(self):
        self.index += 1
        if self.index >= len(self.data_loader):
            print("No more images")
            return
        
        data = self.data_loader[self.index]

        self.image.configure(image=ctk.CTkImage(light_image=data.image, size=data.image.size))
        self.filename.configure(text=data.filename)
        self.classID.configure(text=data.id)
        self.attributes.configure(text="\n".join(data.attributes()))
        
    
    def detect_face(self):
        if self.face_detection_engine is None:
            self.face_detection_engine = FaceDetectionEngine()
        
        img = self.data_loader[self.index].image.copy()
        face = self.face_detection_engine(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        draw = ImageDraw.Draw(img)
        draw.rectangle(face[0]["box"], outline="green", width=3)
        
        self.image.configure(image=ctk.CTkImage(light_image=img, size=img.size))
  
  
    def reload_data(self):
        filter_class = int(self.filter_class.get())
        filter_attributes = [Attribute[attr] for attr, var in self.chk_attributes.items() if var.get()]
        
        self.data_loader = DataLoader(self.data_path, filter_class=filter_class, filter_attributes=filter_attributes)
        self.index = -1
        self.next_image()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", required=True, help="Dataset path")
    args = parser.parse_args()
    
    app = App(args.src_path)
    app.mainloop()
    