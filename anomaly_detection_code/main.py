from preprocess import divide_image_into_grids,flatten_and_prefix_files,highlight_colors_in_folder,highlight_differences,load_reference_colors
from collections import defaultdict
from models import train_autoencoder, AutoencoderWithDropout
from dataset import GridDataset
from torch.utils.data import DataLoader, Dataset
import torch
from model_eval import plot_worst_reconstructions_for_single_image,build_anomaly_detection_df,generate_classification_report
import pandas as pd
import os
import argparse

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Run anomaly detection on MVTec dataset')
    parser.add_argument('--object_choice', type=str, required=True, help='Object choice for anomaly detection')
    parser.add_argument('--preprocess', action='store_true', help='Whether to preprocess the data')
    parser.add_argument('--plot_reconstructions', action='store_true', help='Whether to plot the worst 3 reconstructions')
    parser.add_argument('--threshold', type=float, required=True, help='Threshold for anomaly detection')
    parser.add_argument('--grid_size', type=int, default=8, help='Grid size for image patches (default: 8)')
    parser.add_argument('--load_model', action='store_true', help='Whether to load a saved model')
    parser.add_argument('--train_model', action='store_true', help='Whether to train the model')
    parser.add_argument('--model_dir', type=str, default='./models/', help='Directory where the models are saved')
    args = parser.parse_args()

    # Use args in your script
    object_choice = args.object_choice
    base_dir = f"/Users/benjamincolmey/Desktop/mvtec_anomaly_detection/{object_choice}"
    original_train_dir = os.path.join(base_dir, "train/good/")
    original_test_dir = os.path.join(base_dir, "test/")
    grid_size = args.grid_size

    if args.preprocess:
        flatten_and_prefix_files(original_test_dir)
        reference_image_path = original_test_dir
        target_folder_path = original_test_dir
        output_folder_path = original_test_dir
        highlight_colors_in_folder(reference_image_path, target_folder_path, output_folder_path)
        divide_image_into_grids(original_test_dir, output_dir_name='grid', initial_grid_size=grid_size)
        divide_image_into_grids(original_train_dir, output_dir_name='grid', initial_grid_size=grid_size)


    grid_train_dir = os.path.join(original_train_dir, 'grid')
    grid_test_dir = os.path.join(original_test_dir, 'grid')

    # Setup datasets and DataLoaders
    train_dataset = GridDataset(grid_train_dir, is_train=True)
    test_dataset = GridDataset(grid_test_dir, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = AutoencoderWithDropout().to(device)


    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    save_model_path = os.path.join(model_dir, f"{object_choice}_model{grid_size}x{grid_size}.pth")

    if args.load_model:
        autoencoder.load_state_dict(torch.load(save_model_path))

    if args.load_model:
        autoencoder.load_state_dict(torch.load(save_model_path))

    if args.train_model:
        train_autoencoder(autoencoder, save_model_path, train_loader, epochs=1, learning_rate=0.001, data_usage_percent=100)

        autoencoder.eval()

    if args.plot_reconstructions:
        base_image_path = os.path.join(grid_test_dir, "glue_008.png")
        plot_worst_reconstructions_for_single_image(autoencoder, base_image_path, device, grid_size, numb_of_reconstructions=3)

    threshold = args.threshold
    df = build_anomaly_detection_df(grid_test_dir, autoencoder, device, threshold, grid_size)
    pd.set_option('display.max_rows', len(df))
    print(df.head(len(df)))
    report = generate_classification_report(df)
    print(report)


if __name__ == "__main__":
    main()
