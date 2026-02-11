import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    EfficientNetV2B3, ResNet50, VGG16, VGG19,
    EfficientNetB0, EfficientNetB1, EfficientNetB7
)
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from PIL import Image, ImageDraw, ImageFont

# =========================
# Configuration
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
TOP_K = 5  # Number of top results to show per model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Updated_categories')
TEST_IMAGES_DIR = os.path.join(BASE_DIR, 'Testing samples')
RESULTS_DIR = os.path.join(BASE_DIR, 'results3_aligned', 'model_comparison')
COMPARISON_TABLES_DIR = os.path.join(RESULTS_DIR, 'comparison_tables')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(COMPARISON_TABLES_DIR, exist_ok=True)

# =========================
# Models (in order)
# =========================
MODELS = {
    'EfficientNetV2-B3': {
        'fn': EfficientNetV2B3,
        'preprocess': tf.keras.applications.efficientnet_v2.preprocess_input
    },
    'ResNet50': {
        'fn': ResNet50,
        'preprocess': tf.keras.applications.resnet50.preprocess_input
    },
    'VGG16': {
        'fn': VGG16,
        'preprocess': tf.keras.applications.vgg16.preprocess_input
    },
    'VGG19': {
        'fn': VGG19,
        'preprocess': tf.keras.applications.vgg19.preprocess_input
    },
    'EfficientNetB0': {
        'fn': EfficientNetB0,
        'preprocess': tf.keras.applications.efficientnet.preprocess_input
    },
    'EfficientNetB1': {
        'fn': EfficientNetB1,
        'preprocess': tf.keras.applications.efficientnet.preprocess_input
    },
    'EfficientNetB7': {
        'fn': EfficientNetB7,
        'preprocess': tf.keras.applications.efficientnet.preprocess_input
    },
}

# =========================
# Utility Functions
# =========================
def get_real_param_counts(model):
    """Extract REAL parameter counts from TensorFlow."""
    trainable = np.sum([
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights
    ])
    non_trainable = np.sum([
        tf.keras.backend.count_params(w)
        for w in model.non_trainable_weights
    ])
    return trainable + non_trainable, trainable, non_trainable


def calculate_model_size_mb(model):
    return model.count_params() * 4 / (1024 * 1024)


def calculate_similarity_score(query_feature, result_feature):
    """Calculate cosine similarity between query and single result."""
    q_norm = norm(query_feature)
    r_norm = norm(result_feature)
    
    if q_norm == 0 or r_norm == 0:
        return 0.0
    
    return np.dot(query_feature, result_feature) / (q_norm * r_norm)


def calculate_diversity_index(features):
    """Calculate Shannon Diversity Index based on pairwise distances."""
    if len(features) < 2:
        return 0.0
    
    distances = pdist(features, metric='euclidean')
    distance_quantized = np.round(distances, decimals=1)
    unique_dists, counts = np.unique(distance_quantized, return_counts=True)
    proportions = counts / counts.sum()
    shannon_index = -np.sum(proportions * np.log(proportions + 1e-10))
    
    return shannon_index


def get_test_images():
    if not os.path.exists(TEST_IMAGES_DIR):
        return []
    return sorted([
        os.path.join(TEST_IMAGES_DIR, f)
        for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])


def create_comparison_table_png(input_image_path, all_model_results, save_path):
    """
    Create a perfectly aligned comparison table with clear grid structure.
    """
    
    # Grid dimensions
    cell_img_size = 130  # Size of each result image
    cell_width = cell_img_size + 10  # Width of each cell (image + padding)
    cell_height = cell_img_size + 35  # Height of each cell (image + score)
    
    model_col_width = 180  # Width for model name column
    header_row_height = 180  # Space for title and input image
    rank_header_height = 30  # Height for "Rank 1, Rank 2" labels
    
    num_models = len(all_model_results)
    num_ranks = TOP_K
    
    # Calculate total dimensions
    table_width = model_col_width + (num_ranks * cell_width)
    table_height = rank_header_height + (num_models * cell_height)
    total_width = table_width + 40  # Add margins
    total_height = header_row_height + table_height + 40
    
    # Create canvas
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    
    # Load fonts
    try:
        font_title = ImageFont.truetype("arial.ttf", 28)
        font_header = ImageFont.truetype("arialbd.ttf", 14)
        font_model = ImageFont.truetype("arialbd.ttf", 13)
        font_score = ImageFont.truetype("arial.ttf", 13)
    except:
        font_title = ImageFont.load_default()
        font_header = font_title
        font_model = font_title
        font_score = font_title
    
    # Starting positions
    margin_x = 20
    margin_y = 20
    current_y = margin_y
    
    # ===== HEADER SECTION =====
    
    # Title
    input_name = os.path.basename(input_image_path)
    input_num = os.path.splitext(input_name)[0]
    title = f"Input #{input_num}"
    
    bbox = draw.textbbox((0, 0), title, font=font_title)
    title_width = bbox[2] - bbox[0]
    title_x = (total_width - title_width) // 2
    draw.text((title_x, current_y), title, fill="black", font=font_title)
    current_y += 40
    
    # Input image
    try:
        input_img = Image.open(input_image_path).convert("RGB")
        input_img.thumbnail((120, 120), Image.Resampling.LANCZOS)
        img_x = (total_width - input_img.width) // 2
        canvas.paste(input_img, (img_x, current_y))
        current_y += input_img.height + 5
        
        label = "[input image]"
        bbox = draw.textbbox((0, 0), label, font=font_score)
        label_width = bbox[2] - bbox[0]
        draw.text(((total_width - label_width) // 2, current_y), label, 
                 fill="gray", font=font_score)
        current_y += 30
    except:
        current_y += 100
    
    # ===== TABLE SECTION =====
    
    table_start_y = current_y
    table_start_x = margin_x
    
    # Column headers
    # "Model" header
    draw.text((table_start_x + 5, table_start_y + 5), "Model", 
             fill="black", font=font_header)
    
    # "Rank 1", "Rank 2", etc. headers
    for rank_idx in range(num_ranks):
        col_x = table_start_x + model_col_width + (rank_idx * cell_width)
        rank_text = f"Rank {rank_idx + 1}"
        bbox = draw.textbbox((0, 0), rank_text, font=font_header)
        text_width = bbox[2] - bbox[0]
        text_x = col_x + (cell_width - text_width) // 2
        draw.text((text_x, table_start_y + 5), rank_text, fill="black", font=font_header)
    
    # Horizontal line under headers
    line_y = table_start_y + rank_header_height
    draw.line([(table_start_x, line_y), 
               (table_start_x + table_width, line_y)], 
              fill="black", width=2)
    
    # ===== MODEL ROWS =====
    
    row_y = line_y + 5
    
    for model_idx, (model_name, results) in enumerate(all_model_results.items()):
        # Model name (left column)
        text_y = row_y + (cell_height // 2) - 10
        draw.text((table_start_x + 5, text_y), model_name, 
                 fill="black", font=font_model)
        
        # Draw each result cell
        for rank_idx in range(num_ranks):
            if rank_idx < len(results):
                result = results[rank_idx]
                
                # Calculate cell position
                cell_x = table_start_x + model_col_width + (rank_idx * cell_width)
                cell_y = row_y
                
                # Draw cell border (light gray)
                draw.rectangle(
                    [(cell_x, cell_y), (cell_x + cell_width, cell_y + cell_height)],
                    outline="#E0E0E0",
                    width=1
                )
                
                # Load and paste image
                img_x = cell_x + 5
                img_y = cell_y + 5
                
                try:
                    result_img = Image.open(result['image_path']).convert("RGB")
                    result_img = result_img.resize(
                        (cell_img_size, cell_img_size), 
                        Image.Resampling.LANCZOS
                    )
                    canvas.paste(result_img, (img_x, img_y))
                except Exception as e:
                    # Draw placeholder
                    draw.rectangle(
                        [(img_x, img_y), 
                         (img_x + cell_img_size, img_y + cell_img_size)],
                        outline="gray", width=1
                    )
                    draw.text((img_x + 45, img_y + 60), "[img]", 
                             fill="gray", font=font_score)
                
                # Draw score below image (centered in cell)
                score_text = f"{result['similarity']:.2f}"
                bbox = draw.textbbox((0, 0), score_text, font=font_score)
                text_width = bbox[2] - bbox[0]
                score_x = cell_x + (cell_width - text_width) // 2
                score_y = img_y + cell_img_size + 5
                draw.text((score_x, score_y), score_text, 
                         fill="black", font=font_score)
        
        # Move to next row
        row_y += cell_height
    
    # Save
    canvas.save(save_path, quality=95)
    print(f"  ✓ Saved: {os.path.basename(save_path)}")


# =========================
# Main
# =========================
def main():
    """Main execution function."""
    
    print("="*70)
    print("MODEL COMPARISON - PERFECT ALIGNMENT")
    print("="*70)
    print()
    
    # Get test images
    test_images = get_test_images()
    
    if len(test_images) == 0:
        print(f"❌ ERROR: No test images found in {TEST_IMAGES_DIR}")
        return
    
    print(f"✓ Found {len(test_images)} test images")
    print()
    
    # Load all models and extract database features
    print("="*70)
    print("LOADING MODELS")
    print("="*70)
    
    models_data = {}
    overall_results = []
    
    for model_name, cfg in MODELS.items():
        print(f"\n[{model_name}]")
        
        try:
            model = cfg['fn'](
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(IMG_SIZE, IMG_SIZE, 3)
            )
            
            total_p, trainable_p, non_trainable_p = get_real_param_counts(model)
            model_size_mb = calculate_model_size_mb(model)
            
            print(f"  ✓ Params: {total_p:,} ({model_size_mb:.1f} MB)")
            
            # Extract database features
            datagen = ImageDataGenerator(preprocessing_function=cfg['preprocess'])
            generator = datagen.flow_from_directory(
                DATA_DIR,
                target_size=(IMG_SIZE, IMG_SIZE),
                batch_size=BATCH_SIZE,
                class_mode=None,
                shuffle=False
            )
            
            if generator.samples == 0:
                print(f"  ✗ No images in database")
                continue
            
            db_features = model.predict(generator, verbose=0)
            db_filenames = generator.filenames
            
            print(f"  ✓ Features: {len(db_features)}")
            
            # Build KNN index
            nn = NearestNeighbors(n_neighbors=TOP_K, metric='cosine', algorithm='brute')
            nn.fit(db_features)
            
            models_data[model_name] = {
                'model': model,
                'preprocess': cfg['preprocess'],
                'db_features': db_features,
                'db_filenames': db_filenames,
                'nn': nn,
                'total_params': total_p,
                'trainable_params': trainable_p,
                'non_trainable_params': non_trainable_p,
                'model_size_mb': model_size_mb,
                'similarities': [],
                'diversities': []
            }
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    if not models_data:
        print("\n❌ No models loaded")
        return
    
    # Process each test image
    print(f"\n{'='*70}")
    print("PROCESSING TEST IMAGES")
    print(f"{'='*70}")
    
    for test_idx, test_img_path in enumerate(test_images, 1):
        test_name = os.path.basename(test_img_path)
        print(f"\n[{test_idx}/{len(test_images)}] {test_name}")
        
        # Maintain order with OrderedDict
        from collections import OrderedDict
        input_results = OrderedDict()
        
        for model_name, data in models_data.items():
            # Preprocess
            img = image.load_img(test_img_path, target_size=(IMG_SIZE, IMG_SIZE))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = data['preprocess'](x)
            
            # Extract features
            query_feature = data['model'].predict(x, verbose=0)[0]
            
            # Find top-K
            distances, indices = data['nn'].kneighbors([query_feature])
            top_k_idx = indices[0]
            
            top_k_features = data['db_features'][top_k_idx]
            top_k_paths = [
                os.path.join(DATA_DIR, data['db_filenames'][i]) 
                for i in top_k_idx
            ]
            
            # Calculate scores
            results_for_model = []
            for feat, path in zip(top_k_features, top_k_paths):
                sim_score = calculate_similarity_score(query_feature, feat)
                results_for_model.append({
                    'image_path': path,
                    'similarity': sim_score
                })
            
            input_results[model_name] = results_for_model
            
            # Store overall stats
            avg_sim = np.mean([r['similarity'] for r in results_for_model])
            diversity = calculate_diversity_index(top_k_features)
            data['similarities'].append(avg_sim)
            data['diversities'].append(diversity)
        
        # Create comparison table
        output_filename = f"{os.path.splitext(test_name)[0]}_comparison.png"
        output_path = os.path.join(COMPARISON_TABLES_DIR, output_filename)
        create_comparison_table_png(test_img_path, input_results, output_path)
    
    # Save overall results
    print(f"\n{'='*70}")
    print("SAVING STATISTICS")
    print(f"{'='*70}")
    
    for model_name, data in models_data.items():
        overall_results.append({
            'Model': model_name,
            'Total Params': data['total_params'],
            'Trainable Params': data['trainable_params'],
            'Non-trainable Params': data['non_trainable_params'],
            'Model Size (MB)': data['model_size_mb'],
            'Cosine Similarity Score': np.mean(data['similarities']) if data['similarities'] else 0.0,
            'Diversity Index': np.mean(data['diversities']) if data['diversities'] else 0.0
        })
        tf.keras.backend.clear_session()
    
    df = pd.DataFrame(overall_results)
    df = df.sort_values('Cosine Similarity Score', ascending=False)
    
    csv_path = os.path.join(RESULTS_DIR, 'comparison_table.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\n✓ CSV saved: {csv_path}")
    print()
    print(df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("✅ COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput:")
    print(f"  📄 {csv_path}")
    print(f"  🖼️  {COMPARISON_TABLES_DIR}/ ({len(test_images)} tables)")
    print("="*70)


if __name__ == "__main__":
    main()