# Anomaly-Detection-with-Deep-Learning-and-OpenCV
In modern manufacturing quality control, computer vision systems are crucial for ensuring product integrity, especially when production lines switch between diverse product batches. This project harnesses Python, PyTorch, and OpenCV to develop an unsupervised anomaly detection system, crucial for spotting defects in a flexible manufacturing environment. 

The dataset can be downloaded [here](https://www.mvtec.com/company/research/datasets/mvtec-ad). Note: the file will come with read-only permissions so you will need to allow permissions and can be done with the following command: chmod -R 777 /Users/user/file_path

Packages you will need: scikit-learn, torch, torchvision, matplotlib, pandas, opencv

## Running the script:
The anomaly detection script is designed to work with the MVTec dataset for identifying anomalies in manufacturing products. It supports several command line arguments to customize its execution, including selecting the object for anomaly detection, preprocessing the data, plotting reconstructions, loading a saved model, training the model, and setting the anomaly detection threshold and grid size.

Packages you will need: scikit-learn, torch, torchvision, matplotlib, pandas, opencv
Ensure these are installed in your environment before running the script.

## Basic Usage:
To run the script, navigate to the directory containing the script and use the following command:
Approaches used:
python main.py --object_choice <object> --threshold <value> [options]

Replace <object> with the object type from the MVTec dataset you wish to analyze (e.g., bottle, cable, tile) and <value> with the MSE threshold value for anomaly detection.

Options
--preprocess: Enable this flag to preprocess the data before analysis.
--plot_reconstructions: Enable this flag to plot the worst 3 reconstructions after analysis.
--load_model: Enable this flag to load a saved model instead of training a new one.
--train_model: Enable this flag to train a new model. This will save the model to disk.
--grid_size: Specify the grid size for breaking down the images into smaller patches. Default is 8.
--model_dir: Specify the directory where models are saved/loaded (default: './models')

Example Command:
python main.py --object_choice tile --threshold 0.004 --preprocess --train_model --plot_reconstructions --grid_size 8 --model_dir ./models


To obtain the following result for tile run the following commands: python main.py --object_choice tile  --threshold 0.0246961 --model_dir 'path/anomaly_detection_code/saved_models/'

## Model result on tile dataset:
<img src="https://github.com/bcolmey/Anomaly-Detection-with-Deep-Learning-and-OpenCV/blob/main/images/Tile_report.jpg" width="480" height="250">

## Model result on leather dataset:
<img src="https://github.com/bcolmey/Anomaly-Detection-with-Deep-Learning-and-OpenCV/blob/main/images/Leather_report.jpg" width="480" height="250">


Given these results we see the leather model displays exceptional precision (0.9610) and recall (0.9367) for anomalies, suggesting it is highly accurate in detecting true anomalies and avoiding misclassification. The F1 score of 0.9487 for anomalies reflects excellent model accuracy and balance. The recall for non-anomalies is also high (0.9062), indicating the model reliably recognizes normal instances. The overall accuracy of 0.9279 points to the model's strong general performance. Meanwhile
tile model demonstrates high precision, indicating a strong capability to correctly label true anomalies, with a precision of 0.9054 meaning it has a low rate of false positives. Its recall rate of 0.7701 suggests it successfully identifies 77.01% of all actual anomalies. The F1 score of 0.8323 shows a balanced precision and recall, crucial for minimizing the cost of false detections in quality control.

These metrics indicate that both models are effective but might excel under different conditions due to the nature of the anomalies in their respective datasets.


#### Initial assesment of different object types:
<img src="https://github.com/bcolmey/Anomaly-Detection-with-Deep-Learning-and-OpenCV/blob/main/images/Objects.jpg" width="900" height="500">

In the domain of anomaly detection, certain properties significantly enhance the model's ability to distinguish between normal variations and defects. Good properties include rotational invariance, where the object's orientation doesn't affect its appearance, and color variance, which helps in differentiating anomalies based on color. The absence of a need for a region of interest (ROI) simplifies the model by avoiding the additional step of ROI extraction. Large anomalies and those colored differently from the object are easier to detect.

Considering these properties, grid, tile, carpet, and leather were chosen as the initial focus for classification. Tile and carpet, although not rotationally invariant, have large anomalies and do not require centering or a consistent ROI, allowing for a simpler model structure. Leather, which scores the highest among all objects, is both rotationally and color invariant, with large, distinctly colored anomalies. These characteristics make it an ideal candidate for developing robust anomaly detection models, as they mitigate the complexities introduced by orientation, color similarities, and variable positioning within images.

<img src="https://github.com/bcolmey/Anomaly-Detection-with-Deep-Learning-and-OpenCV/blob/main/images/gradients.jpg" width="700" height="500">


Beginning with the grid dataset, the initial approach aimed to take advantage of the consistent pattern, using gradients and edge detection, however this approach fell short. On a full image scale, small anomalies got lost in the grid's expanse, as seen in the figure above, which showed little difference between a normal and abnormal image in terms of gradients present. Breaking down the image into smaller pieces did not work either, as this erased the long-range patterns, essential for spotting irregularities, especially with the rotations in the training set. Because of these reasons I then moved on to a different dataset, trying the leather as it had the highest score, from my initial ranking. 

#### New color enhancement:

<img src="https://github.com/bcolmey/Anomaly-Detection-with-Deep-Learning-and-OpenCV/blob/main/images/Color_enhancement.jpg" width="500" height="300">

For the tile dataset, a different approach was adopted to preprocess images by identifying and highlighting color differences between a reference set and target images, transforming new colors to neon green for visual emphasis, as shown above. This technique effectively isolates anomalies by color deviation, setting the stage for a deep learning model. Following this, an autoencoder-decoder architecture was implemented, leveraging its ability to learn a compressed representation of the normal tiles and thus flag deviations or anomalies in the reconstruction phase, enabling precise anomaly detection in manufacturing quality control.

#### Reconstructions flaws:

<img src="https://github.com/bcolmey/Anomaly-Detection-with-Deep-Learning-and-OpenCV/blob/main/images/Reconstruction_example.jpg" width="500" height="600">

Next, to tackle the challenge of detecting small-scale anomalies in large images, a strategic approach of segmenting each image into smaller grid patches was employed. This method involves dissecting images into smaller, manageable sections, allowing for a more granular inspection of each segment. By examining each patch individually for anomalies, the likelihood of identifying subtle deviations increases significantly. If an anomaly is detected in any of these patches, indicated by a poor reconstruction score as shown above, the entire image is flagged as anomalous. This granular analysis enhances the sensitivity of the anomaly detection process, ensuring that even minor irregularities are not overlooked, which is crucial for maintaining high standards in quality control.

