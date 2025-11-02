import os
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import torch
import warnings
from datetime import datetime
from tqdm import tqdm
import psutil
import time
import numpy as np
warnings.filterwarnings("ignore")

# Download NLTK data (run once)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ============================================================================
# CONFIGURATION SECTION - OPTIMIZED FOR LARGE DATASETS
# ============================================================================
# Get the base path
USAWC = os.getenv('USAWC')
INPUT_FOLDER = f"{USAWC}\\Desktop\\Data\\USAWC_EDU\\Research\\SQRL\\SRR\\Extracted_Texts\\Analysis_Ready_TXTs"
OUTPUT_FOLDER = f"{USAWC}\\Desktop\\Data\\USAWC_EDU\\Research\\SQRL\\SRR\\BERTopic_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# TOPIC MODELING PARAMETERS - Optimized for 3500 documents with LOW OUTLIERS
NUM_TOPICS = None  # Set to None for automatic detection, or specify (e.g., 28)
MIN_TOPIC_SIZE = 40  # OPTIMIZED: ~1.1% of corpus - balance between topics and outliers
MAX_TOPICS = 35  # OPTIMIZED: Cap at reasonable number
UMAP_N_NEIGHBORS = 35  # INCREASED: Smoother manifold = fewer outliers
UMAP_N_COMPONENTS = 8  # Balanced complexity
UMAP_MIN_DIST = 0.1  # INCREASED: More spread = easier to fit documents
HDBSCAN_MIN_CLUSTER_SIZE = 40  # Match min_topic_size
HDBSCAN_MIN_SAMPLES = 3  # DECREASED: Less strict = fewer outliers (was 10)
HDBSCAN_METRIC = 'euclidean'  # Default metric
HDBSCAN_CLUSTER_SELECTION_METHOD = 'leaf'  # CHANGED: 'leaf' captures more documents than 'eom'
HDBSCAN_PREDICTION_DATA = True  # Enable soft clustering for outliers

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Output folder created: {OUTPUT_FOLDER}")

# ============================================================================
# LOGGING AND MONITORING UTILITIES
# ============================================================================
class GPUMonitor:
    """Monitor GPU usage throughout the process"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def log_status(self, stage):
        """Log current GPU/CPU status"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if self.device == "cuda":
            gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1e9
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1e9
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            status = (f"[{timestamp}] {stage}\n"
                     f"  GPU Memory: {gpu_mem_allocated:.2f}GB allocated / "
                     f"{gpu_mem_reserved:.2f}GB reserved / {gpu_mem_total:.2f}GB total\n"
                     f"  GPU Utilization: {(gpu_mem_allocated/gpu_mem_total)*100:.1f}%\n")
        else:
            cpu_percent = psutil.cpu_percent(interval=1)
            ram_used = psutil.virtual_memory().used / 1e9
            ram_total = psutil.virtual_memory().total / 1e9
            
            status = (f"[{timestamp}] {stage}\n"
                     f"  CPU: {cpu_percent:.1f}%\n"
                     f"  RAM: {ram_used:.2f}GB / {ram_total:.2f}GB\n")
        
        print(status)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(status + "\n")
        
        return status

def write_log(log_file, message):
    """Write timestamped message to log file and print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_msg + "\n")

# Initialize logging
log_file = os.path.join(OUTPUT_FOLDER, "processing_log.txt")
gpu_monitor = GPUMonitor(log_file)

# GPU Check and Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
write_log(log_file, f"Using device: {device}")
if device == "cuda":
    write_log(log_file, f"GPU: {torch.cuda.get_device_name(0)}")
    write_log(log_file, f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Log configuration
write_log(log_file, f"CONFIGURATION:")
write_log(log_file, f"  Target topics: {NUM_TOPICS if NUM_TOPICS else 'Auto-detect'}")
write_log(log_file, f"  Min topic size: {MIN_TOPIC_SIZE}")
write_log(log_file, f"  Max topics: {MAX_TOPICS}")
write_log(log_file, f"  UMAP neighbors: {UMAP_N_NEIGHBORS}")
write_log(log_file, f"  UMAP components: {UMAP_N_COMPONENTS}")

gpu_monitor.log_status("Initial Setup")

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text, use_lemmatization=True, use_stemming=False):
    """Preprocess text with lemmatization or stemming"""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english')) 
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    
    if use_lemmatization:
        tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
        tokens = [lemmatizer.lemmatize(token, pos='n') for token in tokens]
    elif use_stemming:
        tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)

# ============================================================================
# DOCUMENT LOADING
# ============================================================================
write_log(log_file, f"Loading documents from: {INPUT_FOLDER}")
documents = []
file_names = []
failed_files = []

if not os.path.exists(INPUT_FOLDER):
    raise ValueError(f"Folder path '{INPUT_FOLDER}' does not exist.")

total_files = len([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")])
write_log(log_file, f"Found {total_files} text files to process")

encodings = ['utf-8', 'latin-1', 'windows-1252']
for filename in tqdm(os.listdir(INPUT_FOLDER), desc="Loading documents", unit="file"):
    if filename.endswith(".txt"):
        file_path = os.path.join(INPUT_FOLDER, filename)
        content = None
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read().strip()
                    if content:
                        cleaned_content = preprocess_text(content, use_lemmatization=True, use_stemming=False)
                        if cleaned_content:
                            documents.append(cleaned_content)
                            file_names.append(filename)
                        else:
                            failed_files.append((filename, "Empty after preprocessing"))
                    else:
                        failed_files.append((filename, "Empty file"))
                    break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                failed_files.append((filename, str(e)))
                break

if not documents:
    raise ValueError("No valid text files found in the folder.")

write_log(log_file, f"Successfully loaded {len(documents)} documents")
write_log(log_file, f"Failed to load {len(failed_files)} documents")

if failed_files:
    failed_df = pd.DataFrame(failed_files, columns=['Filename', 'Error'])
    failed_df.to_csv(os.path.join(OUTPUT_FOLDER, "failed_files.csv"), index=False)

gpu_monitor.log_status("Documents Loaded")

# ============================================================================
# EMBEDDING GENERATION
# ============================================================================
write_log(log_file, "Initializing embedding model...")
embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)
write_log(log_file, f"Embedding model loaded on {device}")

gpu_monitor.log_status("Embedding Model Loaded")

write_log(log_file, f"Generating embeddings for {len(documents)} documents...")
start_time = time.time()
embeddings = embedding_model.encode(documents, show_progress_bar=True, batch_size=32)
embedding_time = time.time() - start_time
write_log(log_file, f"Embeddings generated in {embedding_time:.2f} seconds ({len(documents)/embedding_time:.2f} docs/sec)")

gpu_monitor.log_status("Embeddings Generated")

write_log(log_file, "Calculating document similarities...")
similarities = cosine_similarity(embeddings)
avg_similarity = similarities.mean()
write_log(log_file, f"Average document similarity (cosine): {avg_similarity:.3f}")
if avg_similarity > 0.9:
    write_log(log_file, "WARNING: High document similarity may limit distinct topics")

embeddings_file = os.path.join(OUTPUT_FOLDER, "document_embeddings.npy")
np.save(embeddings_file, embeddings)
write_log(log_file, f"Embeddings saved to: {embeddings_file}")

# ============================================================================
# TOPIC MODELING - OPTIMIZED FOR LARGE DATASETS
# ============================================================================
write_log(log_file, "Initializing UMAP with optimized parameters...")
try:
    from cuml import UMAP as cumlUMAP
    umap_model = cumlUMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric='cosine',
        random_state=42
    )
    write_log(log_file, "Using GPU-accelerated UMAP (cuML)")
except ImportError:
    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric='cosine',
        random_state=42
    )
    write_log(log_file, "Using CPU UMAP (cuML not available)")

# CHANGED: Use HDBSCAN instead of KMeans for better automatic topic detection
# With parameters optimized to reduce outliers
write_log(log_file, "Using HDBSCAN with outlier-reduction settings...")
hdbscan_model = HDBSCAN(
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,  # Lower = less strict = fewer outliers
    metric=HDBSCAN_METRIC,
    cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,  # 'leaf' is less strict than 'eom'
    prediction_data=HDBSCAN_PREDICTION_DATA,
    cluster_selection_epsilon=0.0  # Can increase to 0.5 if still high outliers
)

# CHANGED: Use MaximalMarginalRelevance for more diverse topic words
representation_model = MaximalMarginalRelevance(diversity=0.3)

gpu_monitor.log_status("Before Topic Modeling")

write_log(log_file, "Initializing BERTopic model with optimized parameters...")
topic_model = BERTopic(
    language="english",
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    representation_model=representation_model,
    verbose=True,
    min_topic_size=MIN_TOPIC_SIZE,  # Minimum documents per topic
    nr_topics=NUM_TOPICS,  # Let HDBSCAN determine optimal number
    calculate_probabilities=False,  # Faster without probabilities
    top_n_words=10  # CHANGED: Increased from default 10 for better topic description
)

write_log(log_file, "Fitting BERTopic model...")
start_time = time.time()
topics, probabilities = topic_model.fit_transform(documents)
modeling_time = time.time() - start_time
write_log(log_file, f"Topic modeling completed in {modeling_time:.2f} seconds")

# Log discovered topics
num_topics = len(set(topics)) - (1 if -1 in topics else 0)
num_outliers = list(topics).count(-1)
outlier_percentage = (num_outliers / len(documents)) * 100

write_log(log_file, f"Discovered {num_topics} topics from {len(documents)} documents")
write_log(log_file, f"Outlier documents (topic -1): {num_outliers} ({outlier_percentage:.1f}%)")

# If outliers are still high, try approximate prediction for outliers
if outlier_percentage > 20:
    write_log(log_file, f"WARNING: High outlier rate ({outlier_percentage:.1f}%). Attempting outlier assignment...")
    
    # Method 1: Reduce outliers by assigning them to nearest topic
    try:
        from sklearn.metrics.pairwise import cosine_distances
        
        # Get outlier indices
        outlier_indices = [i for i, t in enumerate(topics) if t == -1]
        
        if outlier_indices:
            write_log(log_file, f"Reassigning {len(outlier_indices)} outliers to nearest topics...")
            
            # Calculate topic centroids
            topic_centroids = {}
            for topic_id in set(topics) - {-1}:
                topic_doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
                topic_centroids[topic_id] = embeddings[topic_doc_indices].mean(axis=0)
            
            # Assign outliers to nearest topic
            reassigned = 0
            for idx in outlier_indices:
                doc_embedding = embeddings[idx]
                
                # Find nearest topic (with confidence threshold)
                min_distance = float('inf')
                nearest_topic = -1
                
                for topic_id, centroid in topic_centroids.items():
                    distance = cosine_distances([doc_embedding], [centroid])[0][0]
                    if distance < min_distance:
                        min_distance = distance
                        nearest_topic = topic_id
                
                # Only reassign if reasonably close (distance < 0.5 is a reasonable threshold)
                if min_distance < 0.6:  # Adjust threshold as needed
                    topics[idx] = nearest_topic
                    reassigned += 1
            
            write_log(log_file, f"Successfully reassigned {reassigned} outliers")
            write_log(log_file, f"Remaining outliers: {list(topics).count(-1)} ({(list(topics).count(-1)/len(documents))*100:.1f}%)")
            
            # Update topic model's topics
            topic_model.topics_ = topics
            
    except Exception as e:
        write_log(log_file, f"Could not reassign outliers: {e}")

gpu_monitor.log_status("Topic Modeling Completed")

# ============================================================================
# TOPIC REDUCTION (if too many topics discovered)
# ============================================================================
if num_topics > MAX_TOPICS:
    write_log(log_file, f"Reducing {num_topics} topics to {MAX_TOPICS} using hierarchical merging...")
    
    # Save pre-reduction statistics
    pre_reduction_topics = topics.copy()
    pre_reduction_count = num_topics
    
    # Reduce topics
    topic_model.reduce_topics(documents, nr_topics=MAX_TOPICS)
    topics = topic_model.topics_
    num_topics = len(set(topics)) - (1 if -1 in topics else 0)
    
    write_log(log_file, f"Reduced from {pre_reduction_count} to {num_topics} topics")
    
    # Save reduction mapping
    reduction_mapping = pd.DataFrame({
        'Original_Topic': pre_reduction_topics,
        'Reduced_Topic': topics
    })
    reduction_file = os.path.join(OUTPUT_FOLDER, "topic_reduction_mapping.csv")
    reduction_mapping.to_csv(reduction_file, index=False)
    write_log(log_file, f"Topic reduction mapping saved to: {reduction_file}")
else:
    write_log(log_file, f"No reduction needed - {num_topics} topics within acceptable range")

# ============================================================================
# HIERARCHICAL TOPIC MODELING (NEW)
# ============================================================================
write_log(log_file, "Building hierarchical topic structure...")
hierarchical_topics = topic_model.hierarchical_topics(documents)
hierarchical_file = os.path.join(OUTPUT_FOLDER, "hierarchical_topics.csv")
hierarchical_topics.to_csv(hierarchical_file, index=False)
write_log(log_file, f"Hierarchical topics saved to: {hierarchical_file}")

# ============================================================================
# TOPIC VISUALIZATION (NEW)
# ============================================================================
write_log(log_file, "Generating topic visualizations...")

# 1. Intertopic Distance Map
try:
    fig1 = topic_model.visualize_topics()
    fig1.write_html(os.path.join(OUTPUT_FOLDER, "topic_distance_map.html"))
    write_log(log_file, "Topic distance map saved")
except Exception as e:
    write_log(log_file, f"Could not create distance map: {e}")

# 2. Topic Hierarchy
try:
    fig2 = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    fig2.write_html(os.path.join(OUTPUT_FOLDER, "topic_hierarchy.html"))
    write_log(log_file, "Topic hierarchy saved")
except Exception as e:
    write_log(log_file, f"Could not create hierarchy: {e}")

# 3. Topic Barchart - Show ALL topics, sorted by size
try:
    # Show all topics instead of limiting to 20
    fig3 = topic_model.visualize_barchart(top_n_topics=num_topics, n_words=8)
    fig3.write_html(os.path.join(OUTPUT_FOLDER, "topic_barchart.html"))
    write_log(log_file, f"Topic barchart saved (showing all {num_topics} topics)")
except Exception as e:
    write_log(log_file, f"Could not create barchart: {e}")

# 4. Topic Heatmap
try:
    fig4 = topic_model.visualize_heatmap()
    fig4.write_html(os.path.join(OUTPUT_FOLDER, "topic_heatmap.html"))
    write_log(log_file, "Topic heatmap saved")
except Exception as e:
    write_log(log_file, f"Could not create heatmap: {e}")

# ============================================================================
# SAVE TOPIC INFORMATION
# ============================================================================
write_log(log_file, "Generating topic information...")
topic_info = topic_model.get_topic_info()
print("\nTopic Information:")
print(topic_info)

topic_info_file = os.path.join(OUTPUT_FOLDER, "topic_info.csv")
topic_info.to_csv(topic_info_file, index=False)
write_log(log_file, f"Topic information saved to: {topic_info_file}")

# Calculate and save topic sizes with percentages
topic_counts = pd.Series(topics).value_counts().sort_index()
topic_sizes = []
for topic_id in sorted(set(topics)):
    count = topic_counts.get(topic_id, 0)
    percentage = (count / len(documents)) * 100
    topic_sizes.append({
        'Topic': topic_id,
        'Count': count,
        'Percentage': percentage
    })

topic_sizes_df = pd.DataFrame(topic_sizes)
topic_sizes_file = os.path.join(OUTPUT_FOLDER, "topic_sizes.csv")
topic_sizes_df.to_csv(topic_sizes_file, index=False)
write_log(log_file, f"Topic sizes saved to: {topic_sizes_file}")

# Save document-topic assignments
doc_topics_df = pd.DataFrame({
    'Filename': file_names,
    'Document': documents,
    'Topic': topics,
    'Document_Length': [len(doc.split()) for doc in documents]
})
doc_topics_file = os.path.join(OUTPUT_FOLDER, "document_topic_assignments.csv")
doc_topics_df.to_csv(doc_topics_file, index=False)
write_log(log_file, f"Document-topic assignments saved to: {doc_topics_file}")

# Save topic words with scores
topic_words_data = []
for topic_id in sorted(set(topics) - {-1}):
    topic_words = topic_model.get_topic(topic_id)
    for rank, (word, score) in enumerate(topic_words, 1):
        topic_words_data.append({
            'Topic': topic_id,
            'Rank': rank,
            'Word': word,
            'Score': score
        })

topic_words_df = pd.DataFrame(topic_words_data)
topic_words_file = os.path.join(OUTPUT_FOLDER, "topic_words.csv")
topic_words_df.to_csv(topic_words_file, index=False)
write_log(log_file, f"Topic words saved to: {topic_words_file}")

# ============================================================================
# TOPIC LABELING (NEW) - Generate descriptive topic names
# ============================================================================
write_log(log_file, "Generating topic labels...")
topic_labels = {}
for topic_id in sorted(set(topics) - {-1}):
    topic_words = topic_model.get_topic(topic_id)
    # Create label from top 3 words
    top_words = [word for word, _ in topic_words[:3]]
    topic_labels[topic_id] = f"Topic_{topic_id}: {', '.join(top_words)}"

topic_labels_df = pd.DataFrame([
    {'Topic': tid, 'Label': label} 
    for tid, label in topic_labels.items()
])
topic_labels_file = os.path.join(OUTPUT_FOLDER, "topic_labels.csv")
topic_labels_df.to_csv(topic_labels_file, index=False)
write_log(log_file, f"Topic labels saved to: {topic_labels_file}")

# ============================================================================
# REPRESENTATIVE DOCUMENTS
# ============================================================================
def get_enhanced_representative_docs(topic_model, topics, documents, file_names, top_n=5):
    """Select representative documents based on multiple criteria"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    rep_docs = {}
    write_log(log_file, f"Finding representative documents (top {top_n} per topic)...")
    
    for topic_id in tqdm(sorted(set(topics) - {-1}), desc="Processing topics"):
        topic_words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
        indices = [i for i, t in enumerate(topics) if t == topic_id]
        
        if not indices:
            continue
            
        topic_documents = [documents[idx] for idx in indices]
        
        # Keyword overlap scoring
        keyword_scores = []
        for idx in indices:
            doc_tokens = set(word_tokenize(documents[idx].lower()))
            overlap = len(set(topic_words) & doc_tokens)
            normalized_overlap = overlap / len(doc_tokens) if doc_tokens else 0
            keyword_scores.append(overlap + normalized_overlap)
        
        # Document length scoring
        length_scores = [len(documents[idx].split()) for idx in indices]
        max_length = max(length_scores) if length_scores else 1
        normalized_length_scores = [length / max_length for length in length_scores]
        
        # TF-IDF similarity
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(topic_documents)
            
            feature_names = vectorizer.get_feature_names_out()
            topic_vector = np.zeros(len(feature_names))
            for word in topic_words:
                if word in feature_names:
                    word_idx = list(feature_names).index(word)
                    topic_vector[word_idx] = 1.0
            
            similarities = cosine_similarity(tfidf_matrix, topic_vector.reshape(1, -1)).flatten()
        except:
            similarities = [0.5] * len(indices)
        
        # Combine scores
        combined_scores = []
        for i in range(len(indices)):
            combined_score = (
                0.5 * keyword_scores[i] +
                0.2 * normalized_length_scores[i] +
                0.3 * similarities[i]
            )
            combined_scores.append((indices[i], combined_score, keyword_scores[i]))
        
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        rep_docs[topic_id] = []
        for idx, score, keyword_overlap in combined_scores[:top_n]:
            rep_docs[topic_id].append({
                'filename': file_names[idx],
                'document': documents[idx],
                'combined_score': score,
                'keyword_overlap': keyword_overlap,
                'similarity': similarities[indices.index(idx)]
            })
    
    return rep_docs

write_log(log_file, "Finding representative documents...")
rep_docs_enhanced = get_enhanced_representative_docs(topic_model, topics, documents, file_names, top_n=5)

rep_docs_data = []
for topic_id, docs in rep_docs_enhanced.items():
    for i, doc_info in enumerate(docs):
        rep_docs_data.append({
            'Topic': topic_id,
            'Topic_Label': topic_labels.get(topic_id, f"Topic {topic_id}"),
            'Rank': i + 1,
            'Filename': doc_info['filename'],
            'Combined_Score': doc_info['combined_score'],
            'Keyword_Overlap': doc_info['keyword_overlap'],
            'TF_IDF_Similarity': doc_info['similarity'],
            'Document_Preview': doc_info['document'][:500]
        })

rep_docs_df = pd.DataFrame(rep_docs_data)
rep_docs_file = os.path.join(OUTPUT_FOLDER, "representative_documents.csv")
rep_docs_df.to_csv(rep_docs_file, index=False)
write_log(log_file, f"Representative documents saved to: {rep_docs_file}")

# ============================================================================
# TOPIC COHERENCE ANALYSIS
# ============================================================================
def analyze_topic_coherence(topic_model, rep_docs_enhanced, output_file):
    """Analyze and save topic coherence"""
    coherence_data = []
    
    for topic_id in sorted(rep_docs_enhanced.keys()):
        topic_words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
        
        all_doc_words = set()
        for doc_info in rep_docs_enhanced[topic_id]:
            doc_words = set(word_tokenize(doc_info['document'].lower()))
            all_doc_words.update(doc_words)
        
        covered_words = [word for word in topic_words if word in all_doc_words]
        coverage = len(covered_words) / len(topic_words) if topic_words else 0
        
        coherence_data.append({
            'Topic': topic_id,
            'Topic_Label': topic_labels.get(topic_id, f"Topic {topic_id}"),
            'Total_Keywords': len(topic_words),
            'Covered_Keywords': len(covered_words),
            'Coverage_Percent': coverage * 100,
            'Top_Keywords': ', '.join(topic_words[:5]),
            'Covered_Keywords_List': ', '.join(covered_words[:5])
        })
    
    coherence_df = pd.DataFrame(coherence_data)
    coherence_df.to_csv(output_file, index=False)
    write_log(log_file, f"Topic coherence analysis saved to: {output_file}")
    
    return coherence_df

coherence_file = os.path.join(OUTPUT_FOLDER, "topic_coherence.csv")
coherence_df = analyze_topic_coherence(topic_model, rep_docs_enhanced, coherence_file)

# ============================================================================
# TOPIC-TOPIC SIMILARITY MATRIX (NEW)
# ============================================================================
write_log(log_file, "Calculating topic-topic similarities...")
topic_ids = sorted(set(topics) - {-1})
if len(topic_ids) > 1:
    # Get topic embeddings
    topic_embeddings = []
    for topic_id in topic_ids:
        topic_docs_indices = [i for i, t in enumerate(topics) if t == topic_id]
        topic_emb = embeddings[topic_docs_indices].mean(axis=0)
        topic_embeddings.append(topic_emb)
    
    topic_embeddings = np.array(topic_embeddings)
    topic_similarity_matrix = cosine_similarity(topic_embeddings)
    
    # Save as CSV
    topic_sim_df = pd.DataFrame(
        topic_similarity_matrix,
        index=[f"Topic_{tid}" for tid in topic_ids],
        columns=[f"Topic_{tid}" for tid in topic_ids]
    )
    topic_sim_file = os.path.join(OUTPUT_FOLDER, "topic_similarity_matrix.csv")
    topic_sim_df.to_csv(topic_sim_file)
    write_log(log_file, f"Topic similarity matrix saved to: {topic_sim_file}")

# ============================================================================
# SAVE BERTOPIC MODEL (NEW)
# ============================================================================
write_log(log_file, "Saving BERTopic model for future use...")
model_path = os.path.join(OUTPUT_FOLDER, "bertopic_model")
topic_model.save(model_path, serialization="pytorch", save_ctfidf=True, save_embedding_model=embedding_model)
write_log(log_file, f"Model saved to: {model_path}")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
write_log(log_file, "Generating summary report...")

# Calculate topic statistics
topic_stats = []
for topic_id in sorted(set(topics) - {-1}):
    topic_doc_count = list(topics).count(topic_id)
    topic_percentage = (topic_doc_count / len(documents)) * 100
    topic_words = [word for word, _ in topic_model.get_topic(topic_id)[:5]]
    topic_stats.append(f"  Topic {topic_id} ({topic_labels.get(topic_id, '')}): {topic_doc_count} docs ({topic_percentage:.1f}%) - {', '.join(topic_words)}")

summary_report = f"""
{'='*80}
BERTOPIC ANALYSIS SUMMARY REPORT
{'='*80}
Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Output Folder: {OUTPUT_FOLDER}

DATASET STATISTICS:
- Total documents processed: {len(documents)}
- Failed documents: {len(failed_files)}
- Average document length: {np.mean([len(doc.split()) for doc in documents]):.1f} words
- Min document length: {min([len(doc.split()) for doc in documents])} words
- Max document length: {max([len(doc.split()) for doc in documents])} words

TOPIC MODELING CONFIGURATION:
- Algorithm: HDBSCAN (adaptive clustering)
- Min topic size: {MIN_TOPIC_SIZE} documents
- UMAP neighbors: {UMAP_N_NEIGHBORS}
- UMAP components: {UMAP_N_COMPONENTS}
- Representation: MaximalMarginalRelevance (diversity=0.3)

TOPIC MODELING RESULTS:
- Number of topics discovered: {num_topics}
- Documents in outlier topic (-1): {list(topics).count(-1)} ({(list(topics).count(-1)/len(documents))*100:.1f}%)
- Average documents per topic: {len([t for t in topics if t != -1]) / max(num_topics, 1):.1f}
- Topic size range: {topic_counts[topic_counts.index != -1].min()}-{topic_counts[topic_counts.index != -1].max()} documents

TOPIC BREAKDOWN:
{chr(10).join(topic_stats)}

PROCESSING PERFORMANCE:
- Embedding generation time: {embedding_time:.2f} seconds
- Topic modeling time: {modeling_time:.2f} seconds
- Total processing time: {embedding_time + modeling_time:.2f} seconds
- Processing speed: {len(documents)/(embedding_time + modeling_time):.2f} docs/sec

HARDWARE USED:
- Device: {device.upper()}
- Model: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}

TOPIC COHERENCE SUMMARY:
Average topic coherence: {coherence_df['Coverage_Percent'].mean():.1f}%
Min coherence: {coherence_df['Coverage_Percent'].min():.1f}%
Max coherence: {coherence_df['Coverage_Percent'].max():.1f}%

OUTPUT FILES GENERATED:
1. processing_log.txt - Detailed processing log with GPU monitoring
2. failed_files.csv - List of files that failed to process
3. document_embeddings.npy - Document embeddings array (reusable)
4. topic_info.csv - Summary information for each topic
5. topic_sizes.csv - Document counts and percentages per topic
6. topic_words.csv - Keywords and scores for each topic
7. topic_labels.csv - Descriptive labels for each topic
8. document_topic_assignments.csv - Document-to-topic mappings
9. representative_documents.csv - Top 5 representative docs per topic
10. topic_coherence.csv - Topic coherence metrics
11. hierarchical_topics.csv - Hierarchical topic relationships
12. topic_similarity_matrix.csv - Topic-to-topic similarity scores
13. topic_distance_map.html - Interactive topic visualization
14. topic_hierarchy.html - Interactive hierarchical view
15. topic_barchart.html - Topic sizes visualization
16. topic_heatmap.html - Topic similarity heatmap
17. bertopic_model/ - Saved model for future use
18. summary_report.txt - This comprehensive summary

COMPARISON WITH LDA:
- LDA found: 28 topics (user reported)
- BERTopic found: {num_topics} topics
- BERTopic advantages: 
  * Semantic understanding via transformers
  * Automatic outlier detection
  * Hierarchical topic structure
  * Better handling of short/varied documents
- Note: BERTopic may find fewer but more coherent topics

NEXT STEPS:
1. Review interactive visualizations (HTML files)
2. Check topic_labels.csv for topic descriptions
3. Examine representative_documents.csv for topic examples
4. If topics seem too broad, decrease MIN_TOPIC_SIZE
5. If too many small topics, increase MIN_TOPIC_SIZE
6. Model saved and can be reloaded for further analysis
{'='*80}
"""

print(summary_report)
summary_file = os.path.join(OUTPUT_FOLDER, "summary_report.txt")
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write(summary_report)

write_log(log_file, f"Summary report saved to: {summary_file}")

# ============================================================================
# TOPIC QUALITY METRICS (NEW)
# ============================================================================
write_log(log_file, "Calculating topic quality metrics...")

# Calculate topic diversity
topic_diversity_scores = []
for topic_id in sorted(set(topics) - {-1}):
    topic_words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
    # Calculate uniqueness of words
    all_topic_words = []
    for tid in set(topics) - {-1}:
        all_topic_words.extend([w for w, _ in topic_model.get_topic(tid)[:10]])
    
    unique_words = len([w for w in topic_words if all_topic_words.count(w) == 1])
    diversity = unique_words / len(topic_words) if topic_words else 0
    topic_diversity_scores.append({
        'Topic': topic_id,
        'Topic_Label': topic_labels.get(topic_id, f"Topic {topic_id}"),
        'Unique_Words': unique_words,
        'Total_Words': len(topic_words),
        'Diversity_Score': diversity * 100
    })

diversity_df = pd.DataFrame(topic_diversity_scores)
diversity_file = os.path.join(OUTPUT_FOLDER, "topic_diversity.csv")
diversity_df.to_csv(diversity_file, index=False)
write_log(log_file, f"Topic diversity scores saved to: {diversity_file}")
write_log(log_file, f"Average topic diversity: {diversity_df['Diversity_Score'].mean():.1f}%")

# ============================================================================
# OUTLIER ANALYSIS (NEW)
# ============================================================================
write_log(log_file, "Analyzing outlier documents...")

outlier_indices = [i for i, t in enumerate(topics) if t == -1]
if outlier_indices:
    outlier_analysis = []
    
    # Analyze outlier characteristics
    for idx in outlier_indices[:100]:  # Sample first 100 outliers
        doc_length = len(documents[idx].split())
        
        # Find distance to nearest topic
        doc_embedding = embeddings[idx]
        topic_centroids = {}
        for topic_id in set(topics) - {-1}:
            topic_doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
            topic_centroids[topic_id] = embeddings[topic_doc_indices].mean(axis=0)
        
        if topic_centroids:
            distances = [(tid, cosine_similarity([doc_embedding], [centroid])[0][0]) 
                        for tid, centroid in topic_centroids.items()]
            nearest_topic, max_similarity = max(distances, key=lambda x: x[1])
            
            outlier_analysis.append({
                'Filename': file_names[idx],
                'Document_Length': doc_length,
                'Nearest_Topic': nearest_topic,
                'Similarity_to_Nearest': max_similarity,
                'Document_Preview': documents[idx][:200]
            })
    
    if outlier_analysis:
        outlier_df = pd.DataFrame(outlier_analysis)
        outlier_file = os.path.join(OUTPUT_FOLDER, "outlier_analysis.csv")
        outlier_df.to_csv(outlier_file, index=False)
        write_log(log_file, f"Outlier analysis saved to: {outlier_file}")
        
        # Log statistics
        avg_length = outlier_df['Document_Length'].mean()
        avg_similarity = outlier_df['Similarity_to_Nearest'].mean()
        write_log(log_file, f"Outlier avg length: {avg_length:.1f} words")
        write_log(log_file, f"Outlier avg similarity to nearest topic: {avg_similarity:.3f}")
else:
    write_log(log_file, "No outliers to analyze!")

# ============================================================================
# DOCUMENT LENGTH ANALYSIS BY TOPIC (NEW)
# ============================================================================
write_log(log_file, "Analyzing document lengths by topic...")

doc_length_by_topic = []
for topic_id in sorted(set(topics) - {-1}):
    topic_doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
    topic_doc_lengths = [len(documents[i].split()) for i in topic_doc_indices]
    
    doc_length_by_topic.append({
        'Topic': topic_id,
        'Topic_Label': topic_labels.get(topic_id, f"Topic {topic_id}"),
        'Num_Documents': len(topic_doc_lengths),
        'Avg_Length': np.mean(topic_doc_lengths),
        'Min_Length': min(topic_doc_lengths),
        'Max_Length': max(topic_doc_lengths),
        'Std_Length': np.std(topic_doc_lengths)
    })

doc_length_df = pd.DataFrame(doc_length_by_topic)
doc_length_file = os.path.join(OUTPUT_FOLDER, "document_length_by_topic.csv")
doc_length_df.to_csv(doc_length_file, index=False)
write_log(log_file, f"Document length analysis saved to: {doc_length_file}")

# ============================================================================
# PARAMETER RECOMMENDATIONS (NEW)
# ============================================================================
write_log(log_file, "Generating parameter recommendations...")

recommendations = []

# Check if too few topics
if num_topics < 10:
    recommendations.append("⚠ Very few topics detected. Consider:")
    recommendations.append(f"  - Decreasing MIN_TOPIC_SIZE (currently {MIN_TOPIC_SIZE})")
    recommendations.append("  - Increasing UMAP_N_COMPONENTS for more representation")
    recommendations.append("  - Using smaller UMAP_MIN_DIST for tighter clusters")

# Check if too many topics
elif num_topics > 35:
    recommendations.append(f"⚠ Many topics detected ({num_topics}). Consider:")
    recommendations.append(f"  - Increasing MIN_TOPIC_SIZE (currently {MIN_TOPIC_SIZE})")
    recommendations.append(f"  - Set MAX_TOPICS to {max(20, int(num_topics * 0.6))} for automatic reduction")
    recommendations.append("  - Examine hierarchical_topics.csv to identify mergeable topics")

# Check outlier ratio with better thresholds
outlier_ratio = list(topics).count(-1) / len(documents)
if outlier_ratio > 0.20:
    recommendations.append(f"🔴 CRITICAL: Very high outlier ratio ({outlier_ratio*100:.1f}%). Try:")
    recommendations.append(f"  1. Decrease MIN_TOPIC_SIZE from {MIN_TOPIC_SIZE} to {max(20, int(MIN_TOPIC_SIZE * 0.6))}")
    recommendations.append(f"  2. Decrease HDBSCAN_MIN_SAMPLES from {HDBSCAN_MIN_SAMPLES} to 1")
    recommendations.append(f"  3. Set HDBSCAN_CLUSTER_SELECTION_METHOD to 'leaf' (currently: {HDBSCAN_CLUSTER_SELECTION_METHOD})")
    recommendations.append(f"  4. Increase UMAP_MIN_DIST from {UMAP_MIN_DIST} to 0.2")
    recommendations.append("  5. Consider if documents are genuinely too diverse/noisy")
elif outlier_ratio > 0.10:
    recommendations.append(f"⚠ Moderate outlier ratio ({outlier_ratio*100:.1f}%). Consider:")
    recommendations.append(f"  - Decrease HDBSCAN_MIN_SAMPLES from {HDBSCAN_MIN_SAMPLES} to {max(1, HDBSCAN_MIN_SAMPLES - 2)}")
    recommendations.append("  - Outlier reassignment already attempted (check logs)")
elif outlier_ratio > 0.05:
    recommendations.append(f"✓ Acceptable outlier ratio ({outlier_ratio*100:.1f}%)")
else:
    recommendations.append(f"✓ Excellent outlier ratio ({outlier_ratio*100:.1f}%)")

# Check topic balance
topic_counts_no_outlier = topic_counts[topic_counts.index != -1]
if len(topic_counts_no_outlier) > 0:
    imbalance_ratio = topic_counts_no_outlier.max() / topic_counts_no_outlier.min()
    if imbalance_ratio > 15:
        recommendations.append(f"⚠ Highly imbalanced topics (ratio {imbalance_ratio:.1f}:1). Consider:")
        recommendations.append(f"  - Increasing MIN_TOPIC_SIZE to {int(MIN_TOPIC_SIZE * 1.5)} to eliminate tiny topics")
        recommendations.append("  - Using topic reduction to merge small similar topics")
    elif imbalance_ratio > 8:
        recommendations.append(f"✓ Some topic imbalance ({imbalance_ratio:.1f}:1) - expected with real data")

# Provide positive feedback if settings are good
if 15 <= num_topics <= 35 and outlier_ratio <= 0.15 and (len(topic_counts_no_outlier) == 0 or imbalance_ratio <= 8):
    recommendations.append("✓ Excellent configuration! Results look well-balanced.")
    recommendations.append(f"  - {num_topics} topics discovered")
    recommendations.append(f"  - {outlier_ratio*100:.1f}% outliers")
    recommendations.append(f"  - Topic balance ratio: {imbalance_ratio:.1f}:1")

# Save recommendations
recommendations_text = "\n".join(recommendations) if recommendations else "✓ No major issues detected. Parameters seem well-tuned."
recommendations_file = os.path.join(OUTPUT_FOLDER, "parameter_recommendations.txt")
with open(recommendations_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("PARAMETER TUNING RECOMMENDATIONS\n")
    f.write("="*80 + "\n\n")
    f.write("Current Configuration:\n")
    f.write(f"  - MIN_TOPIC_SIZE: {MIN_TOPIC_SIZE}\n")
    f.write(f"  - UMAP_N_NEIGHBORS: {UMAP_N_NEIGHBORS}\n")
    f.write(f"  - UMAP_N_COMPONENTS: {UMAP_N_COMPONENTS}\n")
    f.write(f"  - UMAP_MIN_DIST: {UMAP_MIN_DIST}\n\n")
    f.write("Results:\n")
    f.write(f"  - Topics discovered: {num_topics}\n")
    f.write(f"  - Outlier ratio: {outlier_ratio*100:.1f}%\n")
    f.write(f"  - Avg topic coherence: {coherence_df['Coverage_Percent'].mean():.1f}%\n")
    f.write(f"  - Avg topic diversity: {diversity_df['Diversity_Score'].mean():.1f}%\n\n")
    f.write("Recommendations:\n")
    f.write(recommendations_text + "\n\n")
    f.write("="*80 + "\n")
    f.write("To modify parameters, edit the CONFIGURATION SECTION at the top of the script.\n")
    f.write("="*80 + "\n")

write_log(log_file, f"Parameter recommendations saved to: {recommendations_file}")

if recommendations:
    print("\n" + "="*80)
    print("PARAMETER RECOMMENDATIONS:")
    print("="*80)
    print(recommendations_text)
    print("="*80 + "\n")

# ============================================================================
# CLEANUP
# ============================================================================
gpu_monitor.log_status("Processing Complete - Final Status")

if device == "cuda":
    torch.cuda.empty_cache()
    write_log(log_file, "GPU memory cleared")

write_log(log_file, f"All outputs saved to: {OUTPUT_FOLDER}")
write_log(log_file, "Processing completed successfully!")

print(f"\n{'='*80}")
print(f"✓ ALL PROCESSING COMPLETE!")
print(f"✓ Discovered {num_topics} topics from {len(documents)} documents")
print(f"✓ Check output folder: {OUTPUT_FOLDER}")
print(f"✓ Start with: topic_labels.csv and interactive HTML visualizations")
print(f"{'='*80}")