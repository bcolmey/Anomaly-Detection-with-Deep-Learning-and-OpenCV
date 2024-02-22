
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import heapq 
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np

def count_specific_pixels(image, color_to_count):
    """
    Counts the number of pixels in an image that match a specific color.

    Args:
    - image (PIL.Image): Image to process.
    - color_to_count (tuple): Color to count (in RGB).

    Returns:
    - int: Number of pixels matching the specified color.
    """
    data = np.array(image)
    return np.sum(np.all(data == np.array(color_to_count, dtype=np.uint8), axis=-1))
  


def build_anomaly_detection_df(base_dir, autoencoder, device, threshold, grid_size):
    """
    Evaluates anomaly detection on images in the specified directory using the given autoencoder model.

    Parameters:
    - base_dir: Path to the directory containing the images.
    - autoencoder: Autoencoder model for anomaly detection.
    - device: Device on which the autoencoder model is loaded.
    - threshold: Threshold for classifying an image as an anomaly based on MSE.
    - grid_size: The grid size to be used for processing image patches.

    Returns:
    - DataFrame containing the evaluation results.
    """
    transform_to_tensor = transforms.ToTensor()

    def get_anomaly_type(image_filename):
        """
        Extracts the anomaly type from the image filename.
        """
        return image_filename.split('_')[0]

    anomaly_types = set("_".join(file.rsplit('_', 2)[:-1]) for file in os.listdir(base_dir) if file.lower().endswith('.png'))
    print(anomaly_types)
    data = []

    for anomaly in anomaly_types:
        max_index = find_max_index_for_anomaly_type(base_dir, anomaly, grid_size)
        for set_index in range(1, max_index + 1):
            scores = []
            for i in range(grid_size):
                for j in range(grid_size):
                    image_path = os.path.join(base_dir, f"{anomaly}_{set_index:03}x{i}x{j}.png")
                    try:
                        image = Image.open(image_path).convert('RGB')
                        image_tensor = transform_to_tensor(image).unsqueeze(0).to(device)
                        autoencoder.eval()
                        with torch.no_grad():
                            output = autoencoder(image_tensor)
                            mse = torch.mean((output - image_tensor) ** 2).item()
                            #color_to_count = (0, 255, 0)
                            #pixel_count = count_specific_pixels(image, color_to_count)
                            #adjusted_mse = adjust_score(pixel_count, mse)
                            scores.append(mse)
                        
                    except FileNotFoundError:
                        continue
            #print(set_index)
            top_three_scores = heapq.nlargest(3, scores)
            above_threshold_count = sum(score > threshold for score in scores)
            classification = "ANOMALY" if above_threshold_count >= 1 else "NOT an anomaly"
            is_correct = (anomaly == 'good' and classification == "NOT an anomaly") or \
                         (anomaly != 'good' and classification == "ANOMALY")

            correctly_classified = "Yes" if is_correct else "No"

            data.append([
                f"{anomaly}_{set_index:03}",
                *top_three_scores,
                above_threshold_count,
                classification,
                correctly_classified
            ])

    # Create DataFrame
    df = pd.DataFrame(data, columns=[
        "Image Set",
        "1st Highest Score",
        "2nd Highest Score",
        "3rd Highest Score",
        "Scores Above Threshold",
        "Classification",
        "Correctly Classified"
    ])

    correct_counts = df["Correctly Classified"].value_counts().get("Yes", 0)
    total = len(df)
    correct_percentage = (correct_counts / total) * 100 if total > 0 else 0

    print(f"Percentage of Correctly Classified: {correct_percentage:.2f}%")

    # After creating the DataFrame
    good_scores = df[df['Classification'] == 'NOT an anomaly']['1st Highest Score']
    median_good_score = good_scores.median()

    print(f"Median of the highest scores for 'good' objects: {median_good_score:.4f}")
    return df



def find_max_index_for_anomaly_type(base_path, anomaly_type, initial_grid_size):
    """
    Find the maximum index for which at least one image file exists for the given anomaly type.
    """
    max_index = 0  # Start with 0 to handle the case where no files are found
    set_index = 1  # Initial set index
    while True:
        # Construct the image name using only the set_index and the initial indices (0, 0)
        image_name = f'{anomaly_type}_{set_index:03}x0x0.png'
        image_path = os.path.join(base_path, image_name)
        # Check if the image exists
        if os.path.exists(image_path):
            max_index = set_index  # Update max_index to the last found set_index
            set_index += 1  # Increment set_index for the next iteration
        else:
            break  # If no file is found for the current set_index, exit the loop
    return max_index

def plot_worst_reconstructions_for_single_image(autoencoder, base_image_path, device, grid_size, numb_of_reconstructions):
    """
    Plots worst reconstructions of an image grid.

    Divides an image into a grid, reconstructs each segment, and plots segments with the highest reconstruction error.

    Args:
    - autoencoder (nn.Module): Autoencoder model.
    - base_image_path (str): Path of the image to process.
    - device (torch.device): Computation device.
    - numb_of_reconstructions (int): Number of reconstructions to plot.
    - grid_size (int): Size of the grid (e.g., 8 for 8x8 grid).
    """
    transform_to_tensor = transforms.ToTensor()
    anomalies = []

    for i in range(grid_size):
        for j in range(grid_size):
            new_suffix = f"x{i}x{j}.png"
            image_path = f"{base_image_path[:-4]}{new_suffix}"
            image_prefix = os.path.basename(base_image_path).split('.')[0]
            image_name =  f'{image_prefix}x{i}x{j}.png'
            try:
                print(image_path)
                image = Image.open(image_path).convert('RGB')
                print(image_path)
            except FileNotFoundError:
                print("no file")
                continue

            image_tensor = transform_to_tensor(image).unsqueeze(0).to(device)
            autoencoder.eval()
            with torch.no_grad():
                output = autoencoder(image_tensor)
                mse = torch.nn.functional.mse_loss(output, image_tensor, reduction='none')
                mse = mse.view(mse.size(0), -1).mean(1)
                anomaly_score = mse.cpu().numpy()[0]

            anomalies.append((anomaly_score, image_name, image_tensor))

    anomalies.sort(key=lambda x: x[0], reverse=True)
    

    # Dynamically adjust subplot grid
    fig, axs = plt.subplots(numb_of_reconstructions, 2, figsize=(10, 7))
    
    for idx, (anomaly_score, image_name, image_tensor) in enumerate(anomalies[:numb_of_reconstructions]):
        output_image = transforms.ToPILImage()(autoencoder(image_tensor.to(device)).squeeze(0).cpu())
        original_image = transforms.ToPILImage()(image_tensor.squeeze(0).cpu())

        ax1 = axs[idx][0] 
        ax2 = axs[idx][1] 

        ax1.imshow(original_image)
        ax1.set_title(f'Original Image {image_name}')
        ax1.axis('off')

        ax2.imshow(output_image)
        ax2.set_title(f'Reconstructed Image\nAnomaly Score: {anomaly_score:.4f}')
        ax2.axis('off')
    plt.show()

def print_classification_report(TP, FP, TN, FN, precision, recall, f1_score, accuracy):
    # Header for the table
    header = f"{'':<17} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}"
    divider = "-" * len(header)
    
    # Rows for the table
    anomaly_row = f"{'ANOMALY':<17} {precision:<10.4f} {recall:<10.4f} {f1_score:<10.4f} {TP + FN:<10}"
    not_anomaly_row = f"{'NOT ANOMALY':<17} {(TN / (TN + FN)):<10.4f} {(TN / (TN + FP)):<10.4f} {(2 * ((TN / (TN + FN)) * (TN / (TN + FP))) / ((TN / (TN + FN)) + (TN / (TN + FP)))):<10.4f} {TN + FP:<10}"
    accuracy_row = f"{'Accuracy':<17} {'':<10} {'':<10} {accuracy:<10.4f} {TP + FP + TN + FN:<10}"
    macro_avg_row = f"{'Macro Avg':<17} {((precision + (TN / (TN + FN)))/2):<10.4f} {((recall + (TN / (TN + FP)))/2):<10.4f} {((f1_score + (2 * ((TN / (TN + FN)) * (TN / (TN + FP))) / ((TN / (TN + FN)) + (TN / (TN + FP)))))/2):<10.4f} {TP + FP + TN + FN:<10}"
    weighted_avg_row = f"{'Weighted Avg':<17} {((TP * precision) + (TN * (TN / (TN + FN))))/(TP + TN):<10.4f} {((TP * recall) + (TN * (TN / (TN + FP))))/(TP + TN):<10.4f} {((TP * f1_score) + (TN * (2 * ((TN / (TN + FN)) * (TN / (TN + FP))) / ((TN / (TN + FN)) + (TN / (TN + FP))))))/(TP + TN):<10.4f} {TP + FP + TN + FN:<10}"

    # Combine the rows into a single string
    report = f"\n{header}\n{divider}\n{anomaly_row}\n{not_anomaly_row}\n{divider}\n{accuracy_row}\n{macro_avg_row}\n{weighted_avg_row}\n{divider}\n"

    # Print the report
    print(report)


def generate_classification_report(df):
    """
    Generates a classification report based on the provided DataFrame.

    Parameters:
    - df: DataFrame containing the anomaly detection evaluation results.

    Returns:
    - A string containing the classification report.
    """
    

    # Convert 'Classification' into a boolean where 'ANOMALY' is True and 'NOT an anomaly' is False
    df['Is Anomaly'] = df['Classification'] == 'ANOMALY'

    # Convert 'Correctly Classified' into a boolean where 'Yes' is True (correctly classified)
    df['Is Correct'] = df['Correctly Classified'] == 'Yes'

    # True Positives (TP)
    TP = df[(df['Is Anomaly'] == True) & (df['Is Correct'] == True)].shape[0]

    # False Positives (FP)
    FP = df[(df['Is Anomaly'] == True) & (df['Is Correct'] == False)].shape[0]

    # True Negatives (TN)
    TN = df[(df['Is Anomaly'] == False) & (df['Is Correct'] == True)].shape[0]

    # False Negatives (FN)
    FN = df[(df['Is Anomaly'] == False) & (df['Is Correct'] == False)].shape[0]

    # Calculate Precision, Recall, and F1 Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
    print_classification_report(TP, FP, TN, FN, precision, recall, f1_score, accuracy)

