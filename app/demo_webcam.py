import sys
import os
import argparse
import customtkinter as ctk
from PIL import ImageDraw

# add parent of this file to path to enable importing
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from face_detection.face_detection_engine import FaceDetectionEngine
import cv2
import numpy as np
from PIL import Image


class App(ctk.CTk):
    def __init__(self, data_path, webcam_index):
        super().__init__()
        ctk.set_appearance_mode("light")
        self.grid_columnconfigure(1, weight=1)
        
        self.data_path = data_path
        self.webcam_index = webcam_index
        self.face_detection_engine = FaceDetectionEngine()

        self.title("Face identification demo")
        self.geometry(f"{720}x{600}")

        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)
        title = ctk.CTkLabel(self.sidebar_frame, text="Face identification", font=ctk.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 10))

        file_button = ctk.CTkButton(self.sidebar_frame, text="Load image from file", command=self.load_file, width=200)
        file_button.grid(row=1, column=0, padx=20, pady=10)
        webcam_button = ctk.CTkButton(self.sidebar_frame, text="Capture image from webcam", command=self.load_webcam, width=200)
        webcam_button.grid(row=2, column=0, padx=20, pady=10)
        

        image_frame = ctk.CTkFrame(self, fg_color="transparent")
        image_frame.grid(row=0, column=1, padx=(20, 20), sticky="nsew")
        
        self.image_show = ctk.CTkLabel(image_frame, text="")
        self.image_show.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        
        attributes_frame = ctk.CTkFrame(self, fg_color="gray", corner_radius=0)
        attributes_frame.grid(row=0, column=2, rowspan=4, sticky="nsew")
        attributes_frame.grid_rowconfigure(6, weight=1)
        
        person_label = ctk.CTkLabel(attributes_frame, text="Person", font=ctk.CTkFont(weight="bold"), width=200)
        person_label.grid(row=0, column=0, padx=20, pady=(10, 5))
        self.person = ctk.CTkLabel(attributes_frame)
        self.person.grid(row=1, column=0, padx=20, pady=(0, 10))

        class_label = ctk.CTkLabel(attributes_frame, text="Class", font=ctk.CTkFont(weight="bold"))
        class_label.grid(row=2, column=0, padx=20, pady=(10, 5))
        self.classID = ctk.CTkLabel(attributes_frame)
        self.classID.grid(row=3, column=0, padx=20, pady=(0, 10))
        

    def load_file(self):
        filename = ctk.filedialog.askopenfilename(filetypes=[("Image files", ".png .jpg .jpeg")])
        if filename:
            self.image = Image.open(filename)
            self.show_image()
                
    def load_webcam(self):
        cap = cv2.VideoCapture(self.webcam_index)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            cap.release()
            return
        cap.release()

        self.image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.show_image()           

    def show_image(self):
        # Detect face in image (convert to cv2 first)
        face = self.face_detection_engine(cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR))
        # Draw bbox on image
        draw = ImageDraw.Draw(self.image)
        draw.rectangle(face[0]["box"], outline="green", width=3)
        
        # Render image
        resized_image = self.image.resize((220, int(220 * self.image.size[1] / self.image.size[0])))
        self.image_show.configure(image=ctk.CTkImage(light_image=resized_image, size=resized_image.size))
        
        # TODO - Detect person and class
        person = "TODO"
        class_id = "TODO"
        self.person.configure(text=person)
        self.classID.configure(text=class_id)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", required=True, help="Dataset path")
    parser.add_argument("-w", "--webcam_index", type=int, default=0, help="Webcam index")
    args = parser.parse_args()
    
    app = App(args.src_path, args.webcam_index)
    app.mainloop()
    