import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from skimage.morphology import skeletonize

# --------------------- Biometric Processing Functions ---------------------
def process_fingerprint(image_path):
    img = cv2.imread(image_path, 0)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    enhanced = cv2.medianBlur(enhanced, 5)
    enhanced = cv2.GaussianBlur(enhanced, (5,5), 0)
    
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)

    skeleton = skeletonize(binary // 255)
    skeleton = (skeleton * 255).astype('uint8')
    
    result = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    features = {
        'ridge_endings': {'count': 0, 'color': (0, 0, 255), 'importance': 35},    # Red
        'bifurcations': {'count': 0, 'color': (0, 255, 0), 'importance': 35},     # Green
        'core_points': {'count': 0, 'color': (255, 0, 0), 'importance': 20},      # Blue
        'ridge_patterns': {'count': 0, 'color': (255, 255, 0), 'importance': 10}  # Yellow
    }
    
    border = 20
    for i in range(border, skeleton.shape[0]-border):
        for j in range(border, skeleton.shape[1]-border):
            if skeleton[i,j] == 255:
                neighbors = skeleton[i-1:i+2, j-1:j+2]
                cn = np.sum(neighbors) // 255
                
                if cn == 2: 
                    features['ridge_endings']['count'] += 1
                    cv2.circle(result, (j, i), 3, features['ridge_endings']['color'], -1)
                elif cn == 3: 
                    features['bifurcations']['count'] += 1
                    cv2.circle(result, (j, i), 3, features['bifurcations']['color'], -1)
    
    edges = cv2.Canny(enhanced, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features['ridge_patterns']['count'] = len(contours)
    
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
    

    y_offset += 35
    ridge_density = np.sum(skeleton == 255) / (skeleton.shape[0] * skeleton.shape[1])
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
        self.root.title("Biometric Feature Extractor")
        
        # Create mode selection buttons
        tk.Button(
            root, text="Fingerprint Analysis", command=lambda: self.upload_image("fingerprint"),
            width=25, height=3
        ).pack(pady=10)
        
        tk.Button(
            root, text="Face Detection", command=lambda: self.upload_image("face"),
            width=25, height=3
        ).pack(pady=10)
        
        tk.Button(
            root, text="Iris Detection", command=lambda: self.upload_image("iris"),
            width=25, height=3
        ).pack(pady=10)

    def upload_image(self, mode):
        file_path = filedialog.askopenfilename(
            title=f"Select {mode.capitalize()} Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
        
        try:
            # Process image based on mode
            if mode == "fingerprint":
                result = process_fingerprint(file_path)
            elif mode == "face":
                result = detect_face(file_path)
            elif mode == "iris":
                result = detect_iris(file_path)
            
            # Simple display with fixed size
            cv2.namedWindow(f"{mode.capitalize()} Results", cv2.WINDOW_AUTOSIZE)
            cv2.imshow(f"{mode.capitalize()} Results", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BioMetricApp(root)
    root.geometry("300x300")
    root.mainloop()