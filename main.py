import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import requests
from io import BytesIO

def load_yolo():
    # Load YOLO model and configuration
    net = cv2.dnn.readNet('D:/C Files/object-detection/models/yolov3.weights', 'D:/C Files/object-detection/models/yolov3.cfg')

    # Load COCO class labels
    with open('D:/C Files/object-detection/models/coco.names', 'r') as f:
        classes = f.read().strip().split('\n')

    # Additional object classes
    new_classes = [
        'lamp', 'mirror', 'pillow', 'picture frame', 'rug', 'curtain', 'shoe', 'backpack', 'camera', 'keyboard',
        'printer', 'monitor', 'mouse pad', 'wallet', 'headphones', 'sunglasses', 'umbrella', 'watch', 'bracelet',
        'ring', 'necklace', 'wine bottle', 'coffee cup', 'tea cup', 'fork', 'knife', 'spoon', 'plate', 'napkin',
        'tablecloth', 'candle', 'flower', 'plant', 'tree', 'grass', 'cloud', 'moon', 'sun', 'star', 'mountain',
        'ocean', 'river', 'lake', 'beach', 'desert', 'bridge', 'building', 'skyscraper', 'house', 'church', 'castle'
    ]

    # Combine existing and new classes
    classes += new_classes

    return net, classes

def detect_objects(image_path):
    # Read the image
    if image_path.startswith('http'):  # If URL, download image
        response = requests.get(image_path)
        image_data = response.content
        image = Image.open(BytesIO(image_data))
    else:  # If local file
        image = Image.open(image_path)

    # Convert PIL image to OpenCV format
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the model
    net.setInput(blob)

    # Get the output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Perform forward pass and get predictions
    detections = net.forward(output_layer_names)

    # Initialize lists to store bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Process each detection
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                # Scale the bounding box coordinates to the original image size
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Add bounding box coordinates, confidence, and class ID to the lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Clear previous detected objects in the listbox
    listbox.delete(0, tk.END)

    # Check if any detections are left after NMS
    if len(indices) > 0:
        # Loop over the remaining indices
        for i in indices.flatten():
            # Get bounding box coordinates
            x, y, w, h = boxes[i]

            # Get class label and confidence
            class_id = class_ids[i]
            class_name = classes[class_id]
            confidence = confidences[i]

            # Add detected object to the listbox
            listbox.insert(tk.END, f"{class_name}: {confidence:.2f}")

            # Draw the bounding box and label on the frame
            color = (0, 255, 0)  # Green color
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    result_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_image = Image.fromarray(result_image)
    result_image.thumbnail((800, 800))
    result_image = ImageTk.PhotoImage(result_image)
    panel_result.config(image=result_image)
    panel_result.image = result_image

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        detect_objects(file_path)

def detect_from_url():
    url = url_entry.get()
    if url:
        detect_objects(url)

# Set confidence threshold and non-maximum suppression threshold
confidence_threshold = 0.5
nms_threshold = 0.4  # You can adjust this threshold value as needed

# Create the main window
root = tk.Tk()
root.title("Object Detection App")

# Load YOLO model
net, classes = load_yolo()

# Customize GUI appearance
root.configure(bg='#f0f0f0')  # Set background color

# Create GUI components
upload_label = tk.Label(root, text="Upload Image:", font=('Arial', 14), bg='#f0f0f0')
upload_label.pack(pady=10)

upload_button = tk.Button(root, text="Browse", command=upload_image, padx=10, pady=5, bg='#4CAF50', fg='white', font=('Arial', 12))
upload_button.pack()

url_label = tk.Label(root, text="Image URL:", font=('Arial', 14), bg='#f0f0f0')
url_label.pack(pady=10)

url_entry = tk.Entry(root, font=('Arial', 12), width=50)
url_entry.pack()

url_button = tk.Button(root, text="Detect from URL", command=detect_from_url, padx=10, pady=5, bg='#4CAF50', fg='white', font=('Arial', 12))
url_button.pack(pady=10)

panel_result = tk.Label(root, bg='#f0f0f0')
panel_result.pack()

# Create a scrolled text widget for displaying detected objects
listbox = tk.Listbox(root, selectbackground='#4CAF50', font=('Arial', 12), width=40, height=10)
listbox.pack(pady=20)

# Start the GUI main loop
root.mainloop()
