# Documentation for `compare_final_aligned.py`

This document provides a detailed explanation of the `compare_final_aligned.py` script, which is designed to compare multiple deep learning models for Image Retrieval tasks. It explains the code structure, how it works, and how to run it on different environments.

## 1. Overview
The script performs the following tasks:
1.  **Loads pre-trained models** (EfficientNet, ResNet, VGG, etc.) without their classification heads (using Global Average Pooling).
2.  **Extracts features** from a database of images (`Updated_categories`).
3.  **Indexes features** using K-Nearest Neighbors (KNN).
4.  **Processes test images** from a folder (`Testing samples`):
    *   Finds the top 5 similar images from the database for each model.
    *   Calculates **Cosine Similarity** and **Diversity Index**.
5.  **Generates Visualizations**: Creates a side-by-side comparison image table for each test input, showing results from all models aligned perfectly.
6.  **Saves Statistics**: Exports a CSV file summarizing model size, parameter counts, and retrieval performance metrics.

## 2. Prerequisites
You need **Python 3.8+** and the following libraries:
*   `tensorflow` (Deep Learning framework)
*   `numpy` (Numerical operations)
*   `pandas` (Data manipulation)
*   `scikit-learn` (Nearest Neighbors)
*   `scipy` (Distance metrics)
*   `pillow` (Image processing for visualization)

To install the dependencies, run:
```bash
pip install tensorflow numpy pandas scikit-learn scipy pillow
```

## 3. Code Explanation

### 3.1. Imports & Configuration
The script starts by importing necessary libraries.
*   **Configuration Section**: Defines constants like `IMG_SIZE` (224), `BATCH_SIZE` (32), and `TOP_K` (5).
*   **Paths**: Sets up dynamic paths based on the script's location.
    *   `DATA_DIR`: Database images.
    *   `TEST_IMAGES_DIR`: Query images.
    *   `RESULTS_DIR`: Where output CSV and images are saved.

### 3.2. Models Dictionary
The `MODELS` dictionary defines which architectures to test. Each entry contains:
*   `fn`: The Keras application constructor (e.g., `EfficientNetV2B3`).
*   `preprocess`: The specific preprocessing function required for that model.

### 3.3. Utility Functions
*   **`get_real_param_counts(model)`**: Accurately counts trainable and non-trainable parameters.
*   **`calculate_model_size_mb(model)`**: Estimates the model's memory footprint in MB.
*   **`calculate_similarity_score(...)`**: Computes Cosine Similarity between the query feature vector and result feature vectors.
*   **`calculate_diversity_index(...)`**: Computes the Shannon Diversity Index to measure how "diverse" the retrieved results are (based on visual distance).
*   **`create_comparison_table_png(...)`**: A massive function using `PIL` to draw a generated image. It creates a grid where:
    *   Rows = Models
    *   Columns = Top 1 to Top 5 matches
    *   It aligns text, images, and scores precisely to create a professional-looking comparison chart.

### 3.4. Main Execution
The `main()` function orchestrates the flow:
1.  Checks if test images exist.
2.  Iterates through `MODELS`, building a feature database for each model (using `ImageDataGenerator` and `model.predict`).
3.  Fits `NearestNeighbors` on the database features.
4.  Loops through every **test image**:
    *   Extracts features for the test image using the current model.
    *   Queries KNN to get top matches.
    *   Collects results (image paths, scores).
    *   Calls `create_comparison_table_png` to save the visual result.
5.  Aggregates metrics into a list and saves `comparison_table.csv`.

---

## 4. How to Run

### Option A: Running Locally (Windows/Linux/Mac)

1.  **Folder Structure**: Ensure your folders are organized as follows:
    ```
    ProjectFolder/
    ├── compare_final_aligned.py
    ├── Updated_categories/      <-- Database images (folders containing images)
    │   ├── Category1/
    │   └── Category2/
    └── Testing samples/         <-- Input query images (.jpg/.png)
    ```

2.  **Install Dependencies**:
    Open your terminal/command prompt in the `ProjectFolder` and run:
    ```bash
    pip install tensorflow numpy pandas scikit-learn scipy pillow
    ```

3.  **Run the Script**:
    ```bash
    python compare_final_aligned.py
    ```
    *   Output will be saved in `results3_aligned/model_comparison/`.

### Option B: Running in Google Colab

If you want to run this on Google Colab (to use free GPUs), follow these steps:

1.  **Upload Files**:
    *   Upload `compare_final_aligned.py`.
    *   Upload (or mount) your `Updated_categories` and `Testing samples` folders.
    *   *Tip: It is faster to upload these folders as ZIP files and unzip them in Colab.*

2.  **Create a New Notebook**:
    Create a new `.ipynb` file in Colab.

3.  **Setup Command**:
    Paste the following code into a code cell to install dependencies and unzip data (if you uploaded zips).

    ```python
    # 1. Install dependencies (Colab has most, but ensure versions)
    !pip install pillow pandas scikit-learn

    # 2. Mount Google Drive (Optional - if your data is on Drive)
    from google.colab import drive
    drive.mount('/content/drive')

    # 3. Unzip data (If you uploaded zip files to content)
    # !unzip -q "Updated_categories.zip" -d "."
    # !unzip -q "Testing samples.zip" -d "."
    ```

4.  **Run the Script**:
    You can run the python script directly from the terminal :
    ***Create a Virtual Environment (recommended)***
    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    venv\Scripts\activate      # Windows

    ***Install Dependencies***
    pip install -r requirements.txt

    ***Run the Script***
    python compare_final_aligned.py
    ```

    *Note: If your data folders are in a different location (e.g., inside Drive), you might need to edit lines 23-25 in `compare_final_aligned.py` to point to the correct paths, for example:*
    ```python
    # EDIT THESE PATHS IN THE SCRIPT IF USING DRIVE
    # BASE_DIR = '/content/drive/MyDrive/MyProject'
    ```

### Option C: Updating Paths manually
If your folder structure is different, open `compare_final_aligned.py` in a text editor and modify the **Configuration** section (lines 23-26):

```python
# =========================
# Configuration
# =========================
# ...
## download the dataset and test images and put them in the same folder as the script

BASE_DIR = r"C:\Your\Absolute\Path\To\Project"  # Change this to your project root path
DATA_DIR = os.path.join(BASE_DIR, 'My_Database_Images')  # Change folder name to your dataset images folder name
TEST_IMAGES_DIR = os.path.join(BASE_DIR, 'My_Queries')   # Change folder name to your test images folder name
# ...
```
