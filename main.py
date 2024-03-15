from PIL import Image, ImageDraw
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
    """Process the image to find and visualize groups of outlier columns and estimate heart rate."""
    # Load and process the image to grayscale and then to a binary array
    image = Image.open(input_image_path).convert("L")
    binary_image = np.array(image) < 128

    # Identify outlier columns based on vertical projection
    vertical_projection = np.sum(binary_image, axis=0)
    threshold_for_outliers = np.mean(vertical_projection) + 2 * np.std(vertical_projection)
    outlier_columns = [i for i, count in enumerate(vertical_projection) if count > threshold_for_outliers]

    # Group closely located outliers and draw red lines on the image
    grouped_outliers = group_outlier_columns(outlier_columns, 5)
    draw = ImageDraw.Draw(image)
    for group in grouped_outliers:
        draw.line((group[0], 0, group[-1], image.height), fill="red", width=3)

    # Save the modified image with highlighted outlier groups
    output_image_path = input_image_path.replace('.jpg', '_modified.jpg')
    image.save(output_image_path)

    # Estimate heart rate based on the assumption of a 3-second capture window
    beats_to_bpm_factor = 60 / 3  # Convert capture window rate to beats per minute
    estimated_heart_rate = len(grouped_outliers) * beats_to_bpm_factor

    return output_image_path, len(grouped_outliers), estimated_heart_rate

# Example usage
if __name__ == "__main__":
    input_image_path = 'example_image.jpg'
    output_image_path, number_of_beats, estimated_heart_rate = process_image_and_calculate_lines(input_image_path)
    print(f"Modified image saved to: {output_image_path}")
    print(f"Number of beats: {number_of_beats}")
    print(f"Estimated heart rate: {estimated_heart_rate} beats per minute")
