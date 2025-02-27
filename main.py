import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from skimage.morphology import skeletonize

# --------------------- Biometric Processing Functions ---------------------
def process_fingerprint(image_path):
    # Read and preprocess image
    img = cv2.imread(image_path, 0)
    
    # Enhance contrast and reduce noise
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    enhanced = cv2.medianBlur(enhanced, 5)
    enhanced = cv2.GaussianBlur(enhanced, (5,5), 0)
    
    # Binarization
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours to create a mask for the fingerprint area
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    
    # Assuming the largest contour is the fingerprint
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Ridge thinning
    skeleton = skeletonize(binary // 255)
    skeleton = (skeleton * 255).astype('uint8')

    # Initialize result image and feature dictionary
    result = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    features = {
        'ridge_endings': {'count': 0, 'color': (0, 0, 255), 'importance': 35},    # Red
        'bifurcations': {'count': 0, 'color': (0, 255, 0), 'importance': 35},     # Green
        'core_points': {'count': 0, 'color': (255, 0, 0), 'importance': 20},      # Blue
        'ridge_patterns': {'count': 0, 'color': (255, 255, 0), 'importance': 10}  # Yellow
    }
    
    # Analyze only the fingerprint area
    for i in range(skeleton.shape[0]):
        for j in range(skeleton.shape[1]):
            if skeleton[i, j] == 255 and mask[i, j] == 255:  # Check if within the mask
                neighbors = skeleton[i-1:i+2, j-1:j+2]
                cn = np.sum(neighbors) // 255
                
                if cn == 2:  # Ridge ending
                    features['ridge_endings']['count'] += 1
                    cv2.circle(result, (j, i), 3, features['ridge_endings']['color'], -1)
                elif cn == 3:  # Bifurcation
                    features['bifurcations']['count'] += 1
                    cv2.circle(result, (j, i), 3, features['bifurcations']['color'], -1)

    # Calculate ridge patterns and draw them
    edges = cv2.Canny(enhanced, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features['ridge_patterns']['count'] = len(contours)

    # Draw ridge patterns on the result image
    for contour in contours:
        cv2.drawContours(result, [contour], -1, features['ridge_patterns']['color'], 1)

    # Draw analysis results with background for better readability
    y_offset = 30
    def put_text_with_background(text, y_pos, color=(255, 255, 255)):
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result, (8, y_pos-text_h-5), (text_w+12, y_pos+5), (0, 0, 0), -1)
        cv2.putText(result, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Feature counts
    put_text_with_background("Feature Analysis:", y_offset)
    for feature, data in features.items():
        y_offset += 25
        put_text_with_background(
            f"{feature.replace('_', ' ').title()}: {data['count']}", 
            y_offset, 
            data['color']
        )

    # Quality metrics
    y_offset += 35
    ridge_density = np.sum(skeleton == 255) / (np.sum(mask == 255) + 1e-5)  # Avoid division by zero
    quality_score = min(100, int((features['ridge_endings']['count'] + 
                                   features['bifurcations']['count']) * ridge_density * 100))

    put_text_with_background(f"Ridge Density: {ridge_density:.3f}", y_offset)
    y_offset += 25
    put_text_with_background(f"Quality Score: {quality_score}%", y_offset)

    return result

def detect_face(image_path):
  
    import dlib
    
  
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    detector = dlib.get_frontal_face_detector()
    
    # Read and preprocess image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using dlib
    faces = detector(gray)
    
    # Define facial features and their importance
    features = {
        'jaw': {'points': range(0, 17), 'color': (255, 255, 0), 'importance': 10},
        'right_eyebrow': {'points': range(17, 22), 'color': (255, 0, 255), 'importance': 15},
        'left_eyebrow': {'points': range(22, 27), 'color': (255, 0, 255), 'importance': 15},
        'nose_bridge': {'points': range(27, 31), 'color': (255, 0, 0), 'importance': 20},
        'nose_tip': {'points': range(31, 36), 'color': (255, 0, 0), 'importance': 15},
        'right_eye': {'points': range(36, 42), 'color': (0, 255, 0), 'importance': 25},
        'left_eye': {'points': range(42, 48), 'color': (0, 255, 0), 'importance': 25},
        'outer_lip': {'points': range(48, 60), 'color': (0, 0, 255), 'importance': 20},
        'inner_lip': {'points': range(60, 68), 'color': (0, 0, 255), 'importance': 15}
    }
    
    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)
        
        # Draw rectangle around face
        x, y = face.left(), face.top()
        w, h = face.width(), face.height()
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
        
        # Draw and analyze each feature
        feature_scores = {}
        y_offset = 30
        
        for feature_name, feature_data in features.items():
            points = []
            for point_idx in feature_data['points']:
                point = landmarks.part(point_idx)
                points.append((point.x, point.y))
                # Draw point
                cv2.circle(img, (point.x, point.y), 2, feature_data['color'], -1)
            
            # Connect points
            points = np.array(points, dtype=np.int32)
            cv2.polylines(img, [points], False, feature_data['color'], 1)
            
            # Calculate feature metrics
            if len(points) > 0:
                roi = gray[np.min(points[:, 1]):np.max(points[:, 1]),
                         np.min(points[:, 0]):np.max(points[:, 0])]
                if roi.size > 0:
                    feature_scores[feature_name] = {
                        'variance': np.var(roi),
                        'importance': feature_data['importance']
                    }
        
        # Draw feature analysis
        cv2.putText(img, "Feature Importance:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Sort features by importance
        sorted_features = sorted(features.items(), 
                               key=lambda x: x[1]['importance'], 
                               reverse=True)
        
        for feature_name, feature_data in sorted_features:
            feature_text = f"{feature_name.replace('_', ' ').title()}: {feature_data['importance']}%"
            cv2.putText(img, feature_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, feature_data['color'], 2)
            y_offset += 20
        
        # Calculate and display overall quality score
        if feature_scores:
            quality_score = sum(score['variance'] * score['importance'] 
                              for score in feature_scores.values()) / 1000
            quality_score = min(100, int(quality_score))
            
            cv2.putText(img, f"Quality Score: {quality_score}%",
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

def detect_iris(image_path):
    # Read and preprocess image
    img = cv2.imread(image_path, 0)
    
    # Enhance image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
    
    # Detect iris circles
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.5,
        minDist=200,
        param1=50, 
        param2=40,
        minRadius=40, 
        maxRadius=100
    )
    
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if circles is not None:
        # Take only the strongest circle detection
        circle = circles[0][0]
        x, y, r = map(int, circle)
        
        # Define iris features
        features = {
            'pupil': {'color': (0, 0, 255), 'importance': 30},         # Red
            'iris_inner': {'color': (0, 255, 0), 'importance': 35},    # Green
            'iris_outer': {'color': (255, 0, 0), 'importance': 25},    # Blue
            'iris_patterns': {'color': (255, 255, 0), 'importance': 10} # Yellow
        }
        
        # Draw iris boundaries
        cv2.circle(result, (x, y), r, features['pupil']['color'], 2)
        iris_r = int(r * 1.6)
        cv2.circle(result, (x, y), iris_r, features['iris_outer']['color'], 2)
        
        # Extract and analyze iris texture
        mask = np.zeros_like(img)
        cv2.circle(mask, (x, y), iris_r, 255, -1)
        cv2.circle(mask, (x, y), r, 0, -1)
        iris_region = cv2.bitwise_and(enhanced, enhanced, mask=mask)
        
        # Detect iris patterns
        edges = cv2.Canny(iris_region, 70, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw analysis results with background
        y_offset = 30
        def put_text_with_background(text, y_pos, color=(255, 255, 255)):
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result, (8, y_pos-text_h-5), (text_w+12, y_pos+5), (0, 0, 0), -1)
            cv2.putText(result, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Feature metrics
        put_text_with_background("Iris Analysis:", y_offset)
        y_offset += 30
        
        metrics = {
            'Pupil Size': f"{int(np.pi * r * r)}px²",
            'Iris Size': f"{int(np.pi * iris_r * iris_r)}px²",
            'Pattern Count': f"{len(contours)}",
            'Texture Density': f"{np.sum(edges) / (np.pi * (iris_r * iris_r - r * r)):.3f}"
        }
        
        for metric, value in metrics.items():
            put_text_with_background(f"{metric}: {value}", y_offset)
            y_offset += 25
        
        # Feature importance
        y_offset += 10
        put_text_with_background("Feature Importance:", y_offset)
        for feature, data in features.items():
            y_offset += 25
            put_text_with_background(
                f"{feature.replace('_', ' ').title()}: {data['importance']}%",
                y_offset,
                data['color']
            )
    
    return result

# --------------------- GUI Application ---------------------
class BioMetricApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Biometric Feature Analysis System")
        
        # Set window size and position
        window_width = 1000
        window_height = 600
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Set theme and style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.style.configure('Main.TFrame', background='#f0f0f0')
        self.style.configure('Header.TLabel', 
                           background='#2c3e50', 
                           foreground='white', 
                           font=('Helvetica', 24, 'bold'),
                           padding=10)
        self.style.configure('SubHeader.TLabel',
                           background='#f0f0f0',
                           font=('Helvetica', 12),
                           padding=5)
        self.style.configure('Action.TButton',
                           font=('Helvetica', 11),
                           padding=10)
        
        # Create main container
        self.main_frame = ttk.Frame(root, style='Main.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(header_frame, 
                 text="Biometric Feature Analysis System",
                 style='Header.TLabel').pack(fill=tk.X)
        
        ttk.Label(header_frame,
                 text="Select a biometric feature to analyze",
                 style='SubHeader.TLabel').pack(fill=tk.X)
        
        # Create feature selection frame
        feature_frame = ttk.Frame(self.main_frame)
        feature_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Configure grid
        feature_frame.columnconfigure(0, weight=1)
        feature_frame.columnconfigure(1, weight=1)
        feature_frame.columnconfigure(2, weight=1)
        
        # Feature buttons with icons (you'll need to add your own icons)
        self.create_feature_button(feature_frame, 
                                 "Fingerprint Analysis",
                                 "Analyze fingerprint patterns and minutiae",
                                 lambda: self.upload_image("fingerprint"),
                                 0)
        
        self.create_feature_button(feature_frame,
                                 "Facial Recognition",
                                 "Detect and analyze facial features",
                                 lambda: self.upload_image("face"),
                                 1)
        
        self.create_feature_button(feature_frame,
                                 "Iris Detection",
                                 "Analyze iris patterns and features",
                                 lambda: self.upload_image("iris"),
                                 2)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.main_frame, 
                             textvariable=self.status_var,
                             relief=tk.SUNKEN,
                             padding=5)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(20, 0))

    def create_feature_button(self, parent, title, description, command, column):
        frame = ttk.Frame(parent, padding=10)
        frame.grid(row=0, column=column, sticky='nsew', padx=10)
        
        # Button style for hover effect
        self.style.configure(f'{title}.TButton',
                           background='#3498db',
                           font=('Helvetica', 12, 'bold'))
        
        # Main button
        btn = ttk.Button(frame, 
                        text=title,
                        command=command,
                        style=f'{title}.TButton',
                        padding=20)
        btn.pack(fill=tk.X, pady=(0, 10))
        
        # Description
        ttk.Label(frame,
                 text=description,
                 wraplength=200,
                 justify=tk.CENTER,
                 style='SubHeader.TLabel').pack()

    def upload_image(self, mode):
        self.status_var.set(f"Selecting image for {mode} analysis...")
        file_path = filedialog.askopenfilename(
            title=f"Select {mode.capitalize()} Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            self.status_var.set("Ready")
            return
        
        try:
            self.status_var.set(f"Processing {mode} image...")
            
            # Process image based on mode
            if mode == "fingerprint":
                result = process_fingerprint(file_path)
            elif mode == "face":
                result = detect_face(file_path)
            elif mode == "iris":
                result = detect_iris(file_path)
            
            # Get original image dimensions
            height, width = result.shape[:2]
            
            # Calculate screen dimensions (accounting for taskbar and window borders)
            screen_width = self.root.winfo_screenwidth() - 100
            screen_height = self.root.winfo_screenheight() - 100
            
            # Calculate scaling only if image is larger than screen
            if width > screen_width or height > screen_height:
                scale_width = screen_width / width
                scale_height = screen_height / height
                scale = min(scale_width, scale_height)
                
                width = int(width * scale)
                height = int(height * scale)
                result = cv2.resize(result, (width, height))
            
            # Create window and set properties
            window_name = f"{mode.capitalize()} Analysis Results"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, width, height)
            
            # Display results
            cv2.imshow(window_name, result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            self.status_var.set("Analysis complete")
            
        except Exception as e:
            self.status_var.set("Error during processing")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BioMetricApp(root)
    root.mainloop()