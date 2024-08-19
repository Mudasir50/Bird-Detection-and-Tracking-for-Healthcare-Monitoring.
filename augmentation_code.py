import os
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# Set the paths to your image and coordinates folders
images_folder = r"C:\Users\hp\Desktop\Project_1\augmentated data_set\Bird_Det.v8i.yolov8\train\images"
coordinates_folder = r"C:/Users/hp/Desktop/Project_1/augmentated data_set/Bird_Det.v8i.yolov8/train/labels"
# Create output folders for augmented data
output_images_folder = r"C:\Users\hp\Desktop\Project_1\augmentated data_set\Bird_Det.v8i.yolov8\outputs\images"
output_coordinates_folder =r"C:/Users/hp/Desktop/Project_1/augmentated data_set/Bird_Det.v8i.yolov8\outputs\labels"
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_coordinates_folder, exist_ok=True)

# Initialize the imgaug augmentation sequence with additional augmentations
seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25), scale=(0.5, 1.5)),  # Rotation and scaling
    iaa.Fliplr(0.5),  # Horizontal flip
    iaa.Multiply((0.7, 1.3)),  # Brightness adjustment
    iaa.Crop(percent=(0, 0.1)),  # Crop a random portion of the image
], random_order=True)

# Helper function to load coordinates from a text file
def load_coordinates(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    normalized_coordinates = []
    for line in lines:
        values = list(map(float, line.strip().split()))
        # Extract the first four values as bounding box coordinates (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = values[1:]
        normalized_coordinates.append([x_min, y_min, x_max, y_max])
    return normalized_coordinates

# Define the number of augmented samples you want to generate for each image
num_augmented_samples = 1000

# Iterate through image files and generate a fixed number of augmented samples for each image
for image_file in os.listdir(images_folder):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(images_folder, image_file)

        # Load the image
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        # Load corresponding coordinates (assuming one text file per image)
        coordinates_file = os.path.splitext(image_file)[0] + ".txt"
        coordinates_path = os.path.join(coordinates_folder, coordinates_file)

        if os.path.exists(coordinates_path):
            coordinates_data = load_coordinates(coordinates_path)

            # Convert bounding box coordinates to imgaug format
            bbs = [BoundingBox(x1=x_min * image_width, y1=y_min * image_height, x2=x_max * image_width, y2=y_max * image_height) for x_min, y_min, x_max, y_max in coordinates_data]
            bbs = BoundingBoxesOnImage(bbs, shape=image.shape)

            # Generate a fixed number of augmented samples for each image
            for i in range(num_augmented_samples):
                # Apply augmentation
                augmented_image, augmented_bbs = seq(image=image, bounding_boxes=bbs)

                # Save augmented image
                augmented_image_filename = f"{os.path.splitext(image_file)[0]}_{i}.jpg"
                augmented_image_path = os.path.join(output_images_folder, augmented_image_filename)
                cv2.imwrite(augmented_image_path, augmented_image)

                # Save augmented coordinates to a new text file
                augmented_coordinates_filename = f"{os.path.splitext(coordinates_file)[0]}_{i}.txt"
                augmented_coordinates_path = os.path.join(output_coordinates_folder, augmented_coordinates_filename)
                with open(augmented_coordinates_path, 'w') as file:
                    for bbox in augmented_bbs.bounding_boxes:
                        x_min = bbox.x1 / image_width
                        y_min = bbox.y1 / image_height
                        x_max = bbox.x2 / image_width
                        y_max = bbox.y2 / image_height
                        file.write(f"{x_min} {y_min} {x_max} {y_max}\n")

            # Break the loop after generating the desired number of augmented samples
            if i + 1 >= num_augmented_samples:
                break
        else:
            print(f"Coordinates file not found for {image_file}")

print(f"Data augmentation completed. Generated {num_augmented_samples} augmented samples for each image.")