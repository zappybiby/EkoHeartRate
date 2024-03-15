from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt


def calculate_threshold_for_outliers(vertical_projection):
    """Calculates the threshold for identifying outliers in the vertical projection."""
    return np.mean(vertical_projection) + 2 * np.std(vertical_projection)


def group_outlier_columns(outlier_columns, gap_threshold):
    """Groups outlier columns based on a specified gap threshold."""
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


def calculate_estimated_heart_rate(grouped_outliers, capture_window_seconds=3):
    """Calculates the estimated heart rate from the number of detected beats."""
    beats_to_bpm_factor = 60 / capture_window_seconds
    estimated_heart_rate = len(grouped_outliers) * beats_to_bpm_factor
    return estimated_heart_rate


def process_image(input_image_path, gap_threshold=10, capture_window_seconds=3, debug=False):
    image = Image.open(input_image_path).convert("L")
    binary_image = np.array(image) < 128
    vertical_projection = np.sum(binary_image, axis=0)
    threshold_for_outliers = calculate_threshold_for_outliers(vertical_projection)
    outlier_columns = [i for i, count in enumerate(vertical_projection) if count > threshold_for_outliers]
    grouped_outliers = group_outlier_columns(outlier_columns, gap_threshold)
    estimated_heart_rate = calculate_estimated_heart_rate(grouped_outliers, capture_window_seconds)

    if debug:
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(binary_image, cmap='gray')
        plt.title("Thresholded Image")

        plt.subplot(1, 3, 2)
        threshold_value = np.mean(vertical_projection) + 2 * np.std(vertical_projection)
        column_indices = range(len(vertical_projection))
        above_threshold = vertical_projection >= threshold_value

        plt.bar(column_indices, vertical_projection, color='black', label='Within 2 Std Dev')
        plt.bar(column_indices, vertical_projection * above_threshold, color='blue', label='Outliers (> 2 Std Dev)')
        plt.fill_between(column_indices, 0, threshold_value, color='yellow', alpha=0.3, label='2 Std Dev Area')
        plt.title("Vertical Projection & Threshold")
        plt.xlabel("Column Index")
        plt.ylabel("Sum of Black Pixels per Column")
        plt.legend()
        plt.tight_layout()

        plt.subplot(1, 3, 3)
        img = np.array(Image.open(input_image_path).convert("L"), dtype=np.float32) / 255
        ax = plt.gca()
        ax.imshow(img, cmap='gray', extent=[0, img.shape[1], 0, img.shape[0]])
        for i, group in enumerate(grouped_outliers, start=1):
            peak_column = max(group, key=lambda col: np.sum(binary_image[:, col]))
            ax.axvline(x=peak_column, color='red', linewidth=2)
            ax.text(peak_column, 5, str(i), color='blue', fontsize=12, verticalalignment='bottom')
        ax.set_title("Detected Beats on Image")
        plt.show()

    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    for index, group in enumerate(grouped_outliers):
        if group:
            start_x = group[0]
            end_x = group[-1]
            draw.line((start_x, 0, end_x, image.height), fill="red", width=3)
            label_position = (end_x + 5, 10)
            draw.text(label_position, str(index + 1), fill="blue", font=font)
    output_image_path = "processed_" + input_image_path
    image.save(output_image_path)
    return output_image_path, len(grouped_outliers), estimated_heart_rate


if __name__ == "__main__":
    input_image_path = 'example_image.jpg'
    output_image_path, number_of_beats, estimated_heart_rate = process_image(input_image_path, debug=False)
    print(f"Modified image saved to: {output_image_path}")
    print(f"Number of beats detected: {number_of_beats}")
    print(f"Estimated heart rate: {estimated_heart_rate} beats per minute")
