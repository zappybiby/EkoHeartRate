from PIL import Image, ImageDraw, ImageFont
import numpy as np

def group_outlier_columns(outlier_columns, gap_threshold):
    """Groups outlier columns to identify individual heartbeats in phonocardiograph data.

    Since a single heartbeat may span multiple pixels due to its duration and signal variations,
    closely spaced outliers are grouped together using a gap threshold."""
    grouped_outliers = []
    current_group = [outlier_columns[0]]

    for column in outlier_columns[1:]:
        if column - current_group[-1] <= gap_threshold:
            current_group.append(column)
        else:
            grouped_outliers.append(current_group)
            current_group = [column]
    grouped_outliers.append(current_group)
    return grouped_outliers


def process_image_and_calculate_lines(input_image_path):
    image = Image.open(input_image_path).convert("L")
    binary_image = np.array(image) < 128

    vertical_projection = np.sum(binary_image, axis=0)
    threshold_for_outliers = np.mean(vertical_projection) + 2 * np.std(vertical_projection)
    outlier_columns = [i for i, count in enumerate(vertical_projection) if count > threshold_for_outliers]

    grouped_outliers = group_outlier_columns(outlier_columns, 5)

    # Convert the image back to RGB to draw in color
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    # Attempt to use a default font, may need adjustment based on your system/environment
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for index, group in enumerate(grouped_outliers):
        if group:
            start_x = group[0]
            end_x = group[-1]
            # Draw red line for the detected group
            draw.line((start_x, 0, end_x, image.height), fill="red", width=3)
            # Number the group with a blue label to the right
            label_position = (end_x + 5, 10)  # Adjust positioning as needed
            draw.text(label_position, str(index + 1), fill="blue", font=font)

    output_image_path = input_image_path.replace('.jpg', '_modified.jpg')
    image.save(output_image_path)

    return output_image_path, len(grouped_outliers), len(grouped_outliers) * 20

# Example usage
if __name__ == "__main__":
    input_image_path = 'example_image.jpg'
    output_image_path, number_of_groups, estimated_heart_rate = process_image_and_calculate_lines(input_image_path)
    print(f"Modified image saved to: {output_image_path}")
    print(f"Number of outlier groups: {number_of_groups}")
    print(f"Estimated heart rate: {estimated_heart_rate} beats per minute")
