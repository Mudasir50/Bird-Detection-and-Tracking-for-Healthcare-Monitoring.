import os

# Specify the path to the directory containing the YOLO annotation files
annotation_dir = "C:/Users/hp/Desktop/Project_1/augmentated data_set/Bird_Det.v8i.yolov8/train/labels"

# Define a mapping of old labels to new labels
label_mapping = {
    "0": "14days",
    "1": "WM",
   # Add more label mappings as needed
}

# Iterate through the annotation files and update labels
for filename in os.listdir(annotation_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(annotation_dir, filename), "r") as file:
            lines = file.readlines()

        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                old_label = parts[0]
                if old_label in label_mapping:
                    new_label = label_mapping[old_label]
                    parts[0] = new_label
                updated_lines.append(" ".join(parts))

        with open(os.path.join(annotation_dir, filename), "w") as file:
            file.write("\n".join(updated_lines))

print("Labels have been updated.")