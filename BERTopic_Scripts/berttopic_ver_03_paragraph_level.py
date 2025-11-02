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
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import torch
import warnings
from datetime import datetime
from tqdm import tqdm
import psutil
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
warnings.filterwarnings("ignore")

# Download NLTK data (run once)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ============================================================================
# CONFIGURATION SECTION - OPTIMIZED FOR PARAGRAPH-LEVEL MODELING
# ============================================================================
# REPRODUCIBILITY SETTINGS
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Get the base path
USAWC = os.getenv('USAWC')
INPUT_FOLDER = f"{USAWC}\\Desktop\\Data\\USAWC_EDU\\Research\\SQRL\\SRR\\Extracted_Texts\\Analysis_Ready_TXTs"
OUTPUT_FOLDER = f"{USAWC}\\Desktop\\Data\\USAWC_EDU\\Research\\SQRL\\SRR\\BERTopic_Results_Paragraph_Level_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# TOPIC MODELING PARAMETERS - Optimized for paragraph-level modeling
NUM_TOPICS = None
MIN_TOPIC_SIZE = 100  # Increased to reduce fragmentation (ensures at least MIN_TOPIC_SIZE paragraphs would contain the identified topic)
MAX_TOPICS = 50     # Increased to allow more topics before aggressive reduction (affects the total number of topics idendified before reduction takes place)
UMAP_N_NEIGHBORS = 35  # Increased for broader structure
UMAP_N_COMPONENTS = 8  # Reduced for better reduction and speed
UMAP_MIN_DIST = 0.1  # Tighter clusters for paragraphs
HDBSCAN_MIN_CLUSTER_SIZE = 100  # Increased to match min_topic_size
HDBSCAN_MIN_SAMPLES = 3  # Increased to reduce noise sensitivity
HDBSCAN_METRIC = 'euclidean'  # Aligned with UMAP for consistency
HDBSCAN_CLUSTER_SELECTION_METHOD = 'eom'
HDBSCAN_PREDICTION_DATA = True
MAX_PARAGRAPH_OUTLIER_PERCENTAGE = 20 # Maximum allowed paragraph outlier percentage
PARAGRAPH_LENGTH = 30

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Output folder created: {OUTPUT_FOLDER}")

# ============================================================================
# LOGGING AND MONITORING UTILITIES
# ============================================================================
class GPUMonitor:
    def __init__(self, log_file):
        self.log_file = log_file
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def log_status(self, stage):
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
write_log(log_file, f"  Random seed: {RANDOM_SEED}")
write_log(log_file, f"  Target topics: {NUM_TOPICS if NUM_TOPICS else 'Auto-detect'}")
write_log(log_file, f"  Min topic size: {MIN_TOPIC_SIZE}")
write_log(log_file, f"  Max topics: {MAX_TOPICS}")
write_log(log_file, f"  UMAP neighbors: {UMAP_N_NEIGHBORS}")
write_log(log_file, f"  UMAP components: {UMAP_N_COMPONENTS}")
write_log(log_file, f"  UMAP min distance: {UMAP_MIN_DIST}")
write_log(log_file, f"  HDBSCAN min cluster size: {HDBSCAN_MIN_CLUSTER_SIZE}")
write_log(log_file, f"  HDBSCAN min_samples: {HDBSCAN_MIN_SAMPLES}")
write_log(log_file, f"  HDBSCAN metric: {HDBSCAN_METRIC}")
write_log(log_file, f"  HDBSCAN selection: {HDBSCAN_CLUSTER_SELECTION_METHOD}")
write_log(log_file, f"  HDBSCAN prediction data: {HDBSCAN_PREDICTION_DATA}")
write_log(log_file, f"  Maximum paragraph outlier percentage: {MAX_PARAGRAPH_OUTLIER_PERCENTAGE}")
write_log(log_file, f"  Paragraph length: {PARAGRAPH_LENGTH}")


gpu_monitor.log_status("Initial Setup")

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text, use_lemmatization=True, use_stemming=False):
    # Convert to lowercase but keep original for sentence splitting
    text_lower = text.lower()
    tokens = word_tokenize(text_lower)
    stop_words = set(stopwords.words('english'))
    # Keep punctuation for sentence splitting; only filter stopwords
    tokens = [t for t in tokens if t not in stop_words]
    if use_lemmatization:
        tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
        tokens = [lemmatizer.lemmatize(token, pos='n') for token in tokens]
    elif use_stemming:
        tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens), text  # Return preprocessed and original text

# ============================================================================
# DOCUMENT LOADING
# ============================================================================
write_log(log_file, f"Loading documents from: {INPUT_FOLDER}")
documents = []
raw_documents = []  # Store original text for sentence splitting
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
                        cleaned_content, raw_content = preprocess_text(content)
                        if cleaned_content:
                            documents.append(cleaned_content)
                            raw_documents.append(raw_content)
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
write_log(log_file, f"Sample preprocessed document: {documents[0][:100]}...")
write_log(log_file, f"Sample raw document: {raw_documents[0][:100]}...")

if failed_files:
    failed_df = pd.DataFrame(failed_files, columns=['Filename', 'Error'])
    failed_df.to_csv(os.path.join(OUTPUT_FOLDER, "failed_files.csv"), index=False)

gpu_monitor.log_status("Documents Loaded")

# ============================================================================
# SENTENCE-LEVEL SPLITTING
# ============================================================================
write_log(log_file, "Splitting documents into sentences for sentence-level modeling...")
sentence_docs = []
sentence_to_doc = []
sentence_file_names = []
total_sentences = 0

for doc_idx, doc in enumerate(raw_documents):  # Use raw_documents
    sentences = sent_tokenize(doc)
    if not sentences:
        write_log(log_file, f"Warning: No sentences in document {file_names[doc_idx]}: {doc[:100]}...")
        sentences = doc.split('\n')  # Fallback to newline splitting
    for sent in sentences:
        if sent.strip():
            sentence_docs.append(sent)
            sentence_to_doc.append(doc_idx)
            sentence_file_names.append(file_names[doc_idx])
            total_sentences += 1
    write_log(log_file, f"Document {file_names[doc_idx]}: {len(sentences)} sentences")
write_log(log_file, f"Split {len(documents)} documents into {total_sentences} sentences")
write_log(log_file, f"First 5 sentences: {sentence_docs[:5]}")

if not sentence_docs:
    raise ValueError("No valid sentences found after splitting.")

# ============================================================================
# PARAGRAPH CREATION
# ============================================================================
write_log(log_file, f"Combining sentences into paragraphs ({PARAGRAPH_LENGTH} sentences each)...")
paragraph_docs = []
paragraph_to_doc = []
paragraph_file_names = []
total_paragraphs = 0

for doc_idx in range(len(documents)):
    doc_sent_indices = [i for i, d in enumerate(sentence_to_doc) if d == doc_idx]
    doc_sentences = [sentence_docs[i] for i in doc_sent_indices]
    for start in range(0, len(doc_sentences), PARAGRAPH_LENGTH):
        para_sents = doc_sentences[start:start+8]
        paragraph = ' '.join(para_sents)  # Combine with spaces; use '\n'.join if newlines preferred
        if len(paragraph.split()) < 50:  # Skip short paragraphs
            continue
        paragraph_docs.append(paragraph)
        paragraph_to_doc.append(doc_idx)
        paragraph_file_names.append(file_names[doc_idx])
        total_paragraphs += 1
write_log(log_file, f"Created {total_paragraphs} paragraphs from {len(documents)} documents")
write_log(log_file, f"First 5 paragraphs: {[p[:100] + '...' for p in paragraph_docs[:5]]}")

if not paragraph_docs:
    raise ValueError("No valid paragraphs created.")

# Preprocess paragraphs
write_log(log_file, "Preprocessing paragraphs...")
paragraph_docs_preprocessed = []
for paragraph in paragraph_docs:
    preprocessed_para, _ = preprocess_text(paragraph, use_lemmatization=True, use_stemming=False)
    paragraph_docs_preprocessed.append(preprocessed_para)
paragraph_docs = paragraph_docs_preprocessed  # Overwrite for embedding/modeling

# ============================================================================
# EMBEDDING GENERATION
# ============================================================================
write_log(log_file, "Initializing embedding model...")
embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)
write_log(log_file, f"Embedding model loaded on {device}")

gpu_monitor.log_status("Embedding Model Loaded")

# Generate document-level embeddings for visualizations and similarities
write_log(log_file, f"Generating document-level embeddings for {len(documents)} documents...")
start_time = time.time()
doc_embeddings = embedding_model.encode(documents, show_progress_bar=True, batch_size=64)
doc_embedding_time = time.time() - start_time
write_log(log_file, f"Document embeddings generated in {doc_embedding_time:.2f} seconds ({len(documents)/doc_embedding_time:.2f} docs/sec)")

# Generate paragraph-level embeddings for topic modeling
write_log(log_file, f"Generating paragraph-level embeddings for {len(paragraph_docs)} paragraphs...")
start_time = time.time()
paragraph_embeddings = embedding_model.encode(paragraph_docs, show_progress_bar=True, batch_size=64)
paragraph_embedding_time = time.time() - start_time
write_log(log_file, f"paragraph embeddings generated in {paragraph_embedding_time:.2f} seconds ({len(paragraph_docs)/paragraph_embedding_time:.2f} paragraphs/sec)")

gpu_monitor.log_status("Embeddings Generated")

write_log(log_file, "Calculating document similarities...")
similarities = cosine_similarity(doc_embeddings)
avg_similarity = similarities.mean()
write_log(log_file, f"Average document similarity (cosine): {avg_similarity:.3f}")
if avg_similarity > 0.9:
    write_log(log_file, "WARNING: High document similarity may limit distinct topics")

doc_embeddings_file = os.path.join(OUTPUT_FOLDER, "document_embeddings.npy")
np.save(doc_embeddings_file, doc_embeddings)
write_log(log_file, f"Document embeddings saved to: {doc_embeddings_file}")

paragraph_embeddings_file = os.path.join(OUTPUT_FOLDER, "paragraph_embeddings.npy")
np.save(paragraph_embeddings_file, paragraph_embeddings)
write_log(log_file, f"paragraph embeddings saved to: {paragraph_embeddings_file}")

# ============================================================================
# TOPIC MODELING - OPTIMIZED FOR paragraph-LEVEL
# ============================================================================
write_log(log_file, "Initializing UMAP with optimized parameters...")
try:
    from cuml import UMAP as cumlUMAP
    umap_model = cumlUMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric='cosine',
        random_state=RANDOM_SEED
    )
    write_log(log_file, "Using GPU-accelerated UMAP (cuML)")
except ImportError:
    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric='cosine',
        random_state=RANDOM_SEED
    )
    write_log(log_file, "Using CPU UMAP (cuML not available)")

write_log(log_file, "Using HDBSCAN with paragraph-level settings...")
hdbscan_model = HDBSCAN(
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
    metric=HDBSCAN_METRIC,
    cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
    prediction_data=HDBSCAN_PREDICTION_DATA,
    cluster_selection_epsilon=0.0
)

representation_model = {
    "KeyBERT": KeyBERTInspired(),
    "MMR": MaximalMarginalRelevance(diversity=0.3)
}

vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

gpu_monitor.log_status("Before Topic Modeling")

write_log(log_file, "Initializing BERTopic model with paragraph-level parameters...")
topic_model = BERTopic(
    language="english",
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    representation_model=representation_model,
    vectorizer_model=vectorizer_model,
    verbose=True,
    min_topic_size=MIN_TOPIC_SIZE,
    nr_topics=NUM_TOPICS,
    calculate_probabilities=True,  # Enable for topic distribution
    top_n_words=15
)

write_log(log_file, "Fitting BERTopic model on paragraphs...")
start_time = time.time()
paragraph_topics, paragraph_probs = topic_model.fit_transform(paragraph_docs, embeddings=paragraph_embeddings)
modeling_time = time.time() - start_time
write_log(log_file, f"Topic modeling completed in {modeling_time:.2f} seconds")

# Reduce outliers at paragraph level
write_log(log_file, "Reducing paragraph-level outliers...")
num_paragraph_outliers = list(paragraph_topics).count(-1)
paragraph_outlier_percentage = (num_paragraph_outliers / len(paragraph_docs)) * 100
write_log(log_file, f"Initial paragraph outliers: {num_paragraph_outliers} ({paragraph_outlier_percentage:.1f}%)")
if paragraph_outlier_percentage > MAX_PARAGRAPH_OUTLIER_PERCENTAGE:
    paragraph_topics = topic_model.reduce_outliers(
        paragraph_docs, paragraph_topics, probabilities=paragraph_probs, strategy="probabilities", threshold=0.05
    )
    write_log(log_file, f"After reduction, paragraph outliers: {list(paragraph_topics).count(-1)} ({(list(paragraph_topics).count(-1)/len(paragraph_docs))*100:.1f}%)")
topic_model.topics_ = paragraph_topics

# Aggregate paragraph topics to document level with topic distribution
write_log(log_file, "Aggregating paragraph topics to document level...")
doc_topics = []
doc_topic_distributions = []
unique_topics = sorted(set(paragraph_topics) - {-1})
for doc_idx in range(len(documents)):
    para_indices = [i for i, d in enumerate(paragraph_to_doc) if d == doc_idx]
    if para_indices:
        para_topics = [paragraph_topics[i] for i in para_indices]
        para_probs = [paragraph_probs[i] for i in para_indices] if paragraph_probs is not None else [None] * len(para_indices)
        topic_counts = pd.Series(para_topics).value_counts(normalize=True)
        # Dominant topic (highest proportion, non-outlier if possible)
        non_outlier_topics = [t for t in para_topics if t != -1]
        if non_outlier_topics:
            dominant_topic = pd.Series(non_outlier_topics).mode()[0]
        else:
            dominant_topic = pd.Series(para_topics).mode()[0] if para_topics else -1
        doc_topics.append(dominant_topic)
        # Topic distribution
        distribution = {f"Topic_{t}": topic_counts.get(t, 0.0) for t in unique_topics + [-1]}
        distribution['Doc_Index'] = doc_idx
        doc_topic_distributions.append(distribution)
    else:
        doc_topics.append(-1)
        distribution = {f"Topic_{t}": 0.0 for t in unique_topics + [-1]}
        distribution['Doc_Index'] = doc_idx
        doc_topic_distributions.append(distribution)

topics = doc_topics
doc_topic_dist_df = pd.DataFrame(doc_topic_distributions)
doc_topic_dist_file = os.path.join(OUTPUT_FOLDER, "document_topic_distributions.csv")
doc_topic_dist_df.to_csv(doc_topic_dist_file, index=False)
write_log(log_file, f"Document topic distributions saved to: {doc_topic_dist_file}")

# Log discovered topics
num_topics = len(set(topics)) - (1 if -1 in topics else 0)
num_outliers = list(topics).count(-1)
outlier_percentage = (num_outliers / len(documents)) * 100
write_log(log_file, f"Discovered {num_topics} topics from {len(documents)} documents (aggregated from paragraphs)")
write_log(log_file, f"Outlier documents (topic -1): {num_outliers} ({outlier_percentage:.1f}%)")

gpu_monitor.log_status("Topic Modeling Completed")

# ============================================================================
# TOPIC REDUCTION
# ============================================================================
if num_topics > MAX_TOPICS:
    write_log(log_file, f"Reducing {num_topics} topics to {MAX_TOPICS}...")
    pre_reduction_topics = topics.copy()
    pre_reduction_count = num_topics
    topic_model.reduce_topics(paragraph_docs, nr_topics=MAX_TOPICS)
    paragraph_topics = topic_model.topics_
    # Re-aggregate
    doc_topics = []
    doc_topic_distributions = []
    for doc_idx in range(len(documents)):
        para_indices = [i for i, d in enumerate(paragraph_to_doc) if d == doc_idx]
        if para_indices:
            para_topics = [paragraph_topics[i] for i in para_indices]
            topic_counts = pd.Series(para_topics).value_counts(normalize=True)
            non_outlier_topics = [t for t in para_topics if t != -1]
            if non_outlier_topics:
                dominant_topic = pd.Series(non_outlier_topics).mode()[0]
            else:
                dominant_topic = pd.Series(para_topics).mode()[0] if para_topics else -1
            doc_topics.append(dominant_topic)
            distribution = {f"Topic_{t}": topic_counts.get(t, 0.0) for t in unique_topics + [-1]}
            distribution['Doc_Index'] = doc_idx
            doc_topic_distributions.append(distribution)
        else:
            doc_topics.append(-1)
            distribution = {f"Topic_{t}": 0.0 for t in unique_topics + [-1]}
            distribution['Doc_Index'] = doc_idx
            doc_topic_distributions.append(distribution)
    topics = doc_topics
    doc_topic_dist_df = pd.DataFrame(doc_topic_distributions)
    doc_topic_dist_df.to_csv(doc_topic_dist_file, index=False)
    num_topics = len(set(topics)) - (1 if -1 in topics else 0)
    write_log(log_file, f"Reduced from {pre_reduction_count} to {num_topics} topics")
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
# HIERARCHICAL TOPIC MODELING
# ============================================================================
write_log(log_file, "Building hierarchical topic structure...")
hierarchical_topics = topic_model.hierarchical_topics(paragraph_docs)
hierarchical_file = os.path.join(OUTPUT_FOLDER, "hierarchical_topics.csv")
hierarchical_topics.to_csv(hierarchical_file, index=False)
write_log(log_file, f"Hierarchical topics saved to: {hierarchical_file}")

# ============================================================================
# TOPIC VISUALIZATION
# ============================================================================
write_log(log_file, "Generating topic visualizations...")

# 1. Document Scatter Plot
try:
    write_log(log_file, "Creating document scatter plot...")
    umap_2d = UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=15, min_dist=0.1)
    embeddings_2d = umap_2d.fit_transform(doc_embeddings)
    plt.figure(figsize=(16, 12))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    unique_topics = sorted(set(topics) - {-1})
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_topics)))
    for i, topic_id in enumerate(unique_topics):
        mask = np.array(topics) == topic_id
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[colors[i]], label=f'Topic {topic_id}',
                   alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    if -1 in topics:
        mask = np.array(topics) == -1
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c='none', label='Outliers',
                   alpha=0.3, s=30, edgecolors='black', linewidth=1.5)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title(f'Document Distribution by Topic (n={len(documents)} docs, {num_topics} topics)',
             fontsize=14, fontweight='bold')
    topic_centroids = {}
    for topic_id in unique_topics:
        mask = np.array(topics) == topic_id
        if mask.sum() > 0:
            centroid = embeddings_2d[mask].mean(axis=0)
            topic_centroids[topic_id] = centroid
    topic_labels_dict = {}
    for topic_id in unique_topics:
        topic_words = topic_model.get_topic(topic_id)
        top_words = [word for word, _ in topic_words[:3]]
        topic_labels_dict[topic_id] = ', '.join(top_words)
    N_TOPICS_TO_LABEL = min(30, num_topics)
    topic_counts_for_labels = pd.Series(topics).value_counts()
    largest_topics = topic_counts_for_labels[topic_counts_for_labels.index != -1].head(N_TOPICS_TO_LABEL).index.tolist()
    for topic_id in largest_topics:
        if topic_id in topic_centroids:
            centroid = topic_centroids[topic_id]
            label = f"T{topic_id}: {topic_labels_dict.get(topic_id, '')}"
            xy = (centroid[0], centroid[1])
            plt.annotate(label,
                        xy=xy,
                        textcoords='data',
                        fontsize=10,
                        color='white' if (0.299 * colors[unique_topics.index(topic_id)][0] +
                                        0.587 * colors[unique_topics.index(topic_id)][1] +
                                        0.114 * colors[unique_topics.index(topic_id)][2]) < 0.5 else 'black',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor=colors[unique_topics.index(topic_id)],
                                  alpha=0.7,
                                  edgecolor='black',
                                  linewidth=0.5))
    # plt.xlim(-2, 14.2)
    # plt.ylim(-2, 10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    scatter_file = os.path.join(OUTPUT_FOLDER, "document_scatter_plot.png")
    plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
    plt.close()
    write_log(log_file, f"Document scatter plot saved to: {scatter_file}")
    embeddings_2d_file = os.path.join(OUTPUT_FOLDER, "embeddings_2d.npy")
    np.save(embeddings_2d_file, embeddings_2d)
    write_log(log_file, f"2D embeddings saved to: {embeddings_2d_file}")
except Exception as e:
    write_log(log_file, f"Could not create scatter plot: {e}")

# 2. paragraph Scatter Plot
try:
    write_log(log_file, "Creating paragraph scatter plot...")
    umap_2d_paragraphs = UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=15, min_dist=0.1)
    paragraph_embeddings_2d = umap_2d_paragraphs.fit_transform(paragraph_embeddings)
    plt.figure(figsize=(16, 12))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    unique_paragraph_topics = sorted(set(paragraph_topics) - {-1})
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_paragraph_topics)))
    for i, topic_id in enumerate(unique_paragraph_topics):
        mask = np.array(paragraph_topics) == topic_id
        plt.scatter(paragraph_embeddings_2d[mask, 0], paragraph_embeddings_2d[mask, 1],
                   c=[colors[i]], label=f'Topic {topic_id}',
                   alpha=0.6, s=20, edgecolors='none')
    if -1 in paragraph_topics:
        mask = np.array(paragraph_topics) == -1
        plt.scatter(paragraph_embeddings_2d[mask, 0], paragraph_embeddings_2d[mask, 1],
                   c='none', label='Outliers', 
                   alpha=0.2, s=10, edgecolors='none')
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title(f'Paragraph Distribution by Topic (n={len(paragraph_docs)} paragraphs, {len(unique_paragraph_topics)} topics)',
             fontsize=14, fontweight='bold')
    
    # Calculate topic centroids and labels for paragraphs
    paragraph_topic_centroids = {}
    paragraph_topic_labels_dict = {}
    for topic_id in unique_paragraph_topics:
        mask = np.array(paragraph_topics) == topic_id
        if mask.sum() > 0:
            centroid = paragraph_embeddings_2d[mask].mean(axis=0)
            paragraph_topic_centroids[topic_id] = centroid
            topic_words = topic_model.get_topic(topic_id)
            top_words = [word for word, _ in topic_words[:3]]
            paragraph_topic_labels_dict[topic_id] = ', '.join(top_words)

    # Annotate largest topics
    N_TOPICS_TO_LABEL = len(unique_paragraph_topics) # min(30, len(unique_paragraph_topics))
    paragraph_topic_counts = pd.Series(paragraph_topics).value_counts()
    largest_paragraph_topics = paragraph_topic_counts[paragraph_topic_counts.index != -1].head(N_TOPICS_TO_LABEL).index.tolist()
    for topic_id in largest_paragraph_topics:
        if topic_id in paragraph_topic_centroids:
            centroid = paragraph_topic_centroids[topic_id]
            label = f"T{topic_id}: {paragraph_topic_labels_dict.get(topic_id, '')}"
            plt.annotate(
                label,
                xy=(centroid[0], centroid[1]),
                textcoords='data',
                fontsize=8,
                color='white' if (0.299 * colors[unique_paragraph_topics.index(topic_id)][0] +
                                0.587 * colors[unique_paragraph_topics.index(topic_id)][1] +
                                0.114 * colors[unique_paragraph_topics.index(topic_id)][2]) < 0.5 else 'black',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=colors[unique_paragraph_topics.index(topic_id)],
                    alpha=0.7,
                    edgecolor='black',
                    linewidth=0.5
                )
            )
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    paragraph_scatter_file = os.path.join(OUTPUT_FOLDER, "paragraph_scatter_plot.png")
    plt.savefig(paragraph_scatter_file, dpi=300, bbox_inches='tight')
    plt.close()
    write_log(log_file, f"paragraph scatter plot saved to: {paragraph_scatter_file}")
    paragraph_embeddings_2d_file = os.path.join(OUTPUT_FOLDER, "paragraph_embeddings_2d.npy")
    np.save(paragraph_embeddings_2d_file, paragraph_embeddings_2d)
    write_log(log_file, f"2D paragraph embeddings saved to: {paragraph_embeddings_2d_file}")
except Exception as e:
    write_log(log_file, f"Could not create paragraph scatter plot: {e}")

# 3. Topic Hierarchy
try:
    fig2 = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    fig2.write_html(os.path.join(OUTPUT_FOLDER, "topic_hierarchy.html"))
    write_log(log_file, "Topic hierarchy saved")
except Exception as e:
    write_log(log_file, f"Could not create hierarchy: {e}")

# 4. Topic Barchart
try:
    fig3 = topic_model.visualize_barchart(top_n_topics=num_topics, n_words=8)
    fig3.write_html(os.path.join(OUTPUT_FOLDER, "topic_barchart.html"))
    write_log(log_file, f"Topic barchart saved (showing all {num_topics} topics)")
except Exception as e:
    write_log(log_file, f"Could not create barchart: {e}")

# 5. Topic Heatmap
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

# Save paragraph-topic assignments
paragraph_topics_df = pd.DataFrame({
    'Filename': paragraph_file_names,
    'paragraph': paragraph_docs,
    'Topic': paragraph_topics,
    'Original_Doc_Index': paragraph_to_doc
})
paragraph_topics_file = os.path.join(OUTPUT_FOLDER, "paragraph_topic_assignments.csv")
paragraph_topics_df.to_csv(paragraph_topics_file, index=False)
write_log(log_file, f"paragraph-topic assignments saved to: {paragraph_topics_file}")

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
# TOPIC LABELING
# ============================================================================
write_log(log_file, "Generating topic labels...")
topic_labels = {}
for topic_id in sorted(set(topics) - {-1}):
    topic_words = topic_model.get_topic(topic_id)
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
    from sklearn.feature_extraction.text import TfidfVectorizer
    rep_docs = {}
    write_log(log_file, f"Finding representative documents (top {top_n} per topic)...")
    for topic_id in tqdm(sorted(set(topics) - {-1}), desc="Processing topics"):
        topic_words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
        indices = [i for i, t in enumerate(topics) if t == topic_id]
        if not indices:
            continue
        topic_documents = [documents[idx] for idx in indices]
        keyword_scores = []
        for idx in indices:
            doc_tokens = set(word_tokenize(documents[idx].lower()))
            overlap = len(set(topic_words) & doc_tokens)
            normalized_overlap = overlap / len(doc_tokens) if doc_tokens else 0
            keyword_scores.append(overlap + normalized_overlap)
        length_scores = [len(documents[idx].split()) for idx in indices]
        max_length = max(length_scores) if length_scores else 1
        normalized_length_scores = [length / max_length for length in length_scores]
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
# TOPIC-TOPIC SIMILARITY MATRIX
# ============================================================================
write_log(log_file, "Calculating topic-topic similarities...")
topic_ids = sorted(set(topics) - {-1})
if len(topic_ids) > 1:
    topic_embeddings = []
    for topic_id in topic_ids:
        topic_docs_indices = [i for i, t in enumerate(topics) if t == topic_id]
        topic_emb = doc_embeddings[topic_docs_indices].mean(axis=0)
        topic_embeddings.append(topic_emb)
    topic_embeddings = np.array(topic_embeddings)
    topic_similarity_matrix = cosine_similarity(topic_embeddings)
    topic_sim_df = pd.DataFrame(
        topic_similarity_matrix,
        index=[f"Topic_{tid}" for tid in topic_ids],
        columns=[f"Topic_{tid}" for tid in topic_ids]
    )
    topic_sim_file = os.path.join(OUTPUT_FOLDER, "topic_similarity_matrix.csv")
    topic_sim_df.to_csv(topic_sim_file)
    write_log(log_file, f"Topic similarity matrix saved to: {topic_sim_file}")

# ============================================================================
# SAVE BERTOPIC MODEL
# ============================================================================
write_log(log_file, "Saving BERTopic model for future use...")
model_path = os.path.join(OUTPUT_FOLDER, "bertopic_model")
topic_model.save(model_path, serialization="pytorch", save_ctfidf=True, save_embedding_model=embedding_model)
write_log(log_file, f"Model saved to: {model_path}")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
write_log(log_file, "Generating summary report...")
topic_stats = []
for topic_id in sorted(set(topics) - {-1}):
    topic_doc_count = list(topics).count(topic_id)
    topic_percentage = (topic_doc_count / len(documents)) * 100
    topic_words = [word for word, _ in topic_model.get_topic(topic_id)[:5]]
    topic_stats.append(f"  Topic {topic_id} ({topic_labels.get(topic_id, '')}): {topic_doc_count} docs ({topic_percentage:.1f}%) - {', '.join(topic_words)}")

summary_report = f"""
{'='*80}
BERTOPIC ANALYSIS SUMMARY REPORT (paragraph-LEVEL MODELING)
{'='*80}
Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Output Folder: {OUTPUT_FOLDER}

DATASET STATISTICS:
- Total documents processed: {len(documents)}
- Total paragraphs: {len(paragraph_docs)}
- Failed documents: {len(failed_files)}
- Average document length: {np.mean([len(doc.split()) for doc in documents]):.1f} words
- Min document length: {min([len(doc.split()) for doc in documents])} words
- Max document length: {max([len(doc.split()) for doc in documents])} words
- paragraph-level outliers: {list(paragraph_topics).count(-1)} ({(list(paragraph_topics).count(-1)/len(paragraph_docs))*100:.1f}%)

REPRODUCIBILITY SETTINGS:
- Random seed: {RANDOM_SEED}
- NumPy seed: {RANDOM_SEED}
- PyTorch seed: {RANDOM_SEED}
- CUDA deterministic: {torch.backends.cudnn.deterministic if device == 'cuda' else 'N/A'}

TOPIC MODELING CONFIGURATION:
- Algorithm: HDBSCAN (adaptive clustering)
- Min topic size: {MIN_TOPIC_SIZE} paragraphs
- UMAP neighbors: {UMAP_N_NEIGHBORS}
- UMAP components: {UMAP_N_COMPONENTS}
- Representation: MaximalMarginalRelevance (diversity=0.3)
- Modeling Level: paragraph-level with probabilistic aggregation to documents

TOPIC MODELING RESULTS:
- Number of topics discovered: {num_topics}
- Documents in outlier topic (-1): {list(topics).count(-1)} ({(list(topics).count(-1)/len(documents))*100:.1f}%)
- Average documents per topic: {len([t for t in topics if t != -1]) / max(num_topics, 1):.1f}
- Topic size range: {topic_counts[topic_counts.index != -1].min() if not topic_counts[topic_counts.index != -1].empty else 0}-{topic_counts[topic_counts.index != -1].max() if not topic_counts[topic_counts.index != -1].empty else 0} documents

TOPIC BREAKDOWN:
{chr(10).join(topic_stats)}

PROCESSING PERFORMANCE:
- Document embedding generation time: {doc_embedding_time:.2f} seconds
- paragraph embedding generation time: {paragraph_embedding_time:.2f} seconds
- Topic modeling time: {modeling_time:.2f} seconds
- Total processing time: {doc_embedding_time + paragraph_embedding_time + modeling_time:.2f} seconds
- Processing speed: {len(documents)/(doc_embedding_time + paragraph_embedding_time + modeling_time):.2f} docs/sec

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
3. document_embeddings.npy - Document embeddings array
4. paragraph_embeddings.npy - paragraph embeddings array
5. topic_info.csv - Summary information for each topic
6. topic_sizes.csv - Document counts and percentages per topic
7. topic_words.csv - Keywords and scores for each topic
8. topic_labels.csv - Descriptive labels for each topic
9. document_topic_assignments.csv - Document-to-topic mappings
10. document_topic_distributions.csv - Topic distribution per document
11. paragraph_topic_assignments.csv - paragraph-to-topic mappings
12. representative_documents.csv - Top 5 representative docs per topic
13. topic_coherence.csv - Topic coherence metrics
14. hierarchical_topics.csv - Hierarchical topic relationships
15. topic_similarity_matrix.csv - Topic-to-topic similarity scores
16. document_scatter_plot.png - Document-level topic visualization
17. paragraph_scatter_plot.png - paragraph-level topic visualization
18. topic_hierarchy.html - Interactive hierarchical view
19. topic_barchart.html - Topic sizes visualization
20. topic_heatmap.html - Topic similarity heatmap
21. bertopic_model/ - Saved model for future use
22. summary_report.txt - This comprehensive summary

COMPARISON WITH LDA:
- LDA found: 28 topics (user reported)
- BERTopic found: {num_topics} topics
- BERTopic advantages:
  * Semantic understanding via transformers
  * Automatic outlier detection
  * Hierarchical topic structure
  * paragraph-level modeling for multi-topic documents
- Note: paragraph-level modeling may reduce outliers and capture finer topics

NEXT STEPS:
1. Review paragraph_topic_assignments.csv for paragraph-level topics
2. Check document_topic_distributions.csv for multi-topic analysis
3. Examine paragraph_scatter_plot.png to see paragraph clustering
4. If too many outliers, decrease MIN_TOPIC_SIZE or HDBSCAN_MIN_SAMPLES
5. If topics too broad, adjust UMAP_MIN_DIST or increase MAX_TOPICS
{'='*80}
"""

print(summary_report)
summary_file = os.path.join(OUTPUT_FOLDER, "summary_report.txt")
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write(summary_report)

write_log(log_file, f"Summary report saved to: {summary_file}")

# ============================================================================
# TOPIC QUALITY METRICS
# ============================================================================
write_log(log_file, "Calculating topic quality metrics...")
topic_diversity_scores = []
for topic_id in sorted(set(topics) - {-1}):
    topic_words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
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
# OUTLIER ANALYSIS
# ============================================================================
write_log(log_file, "Analyzing outlier documents...")
outlier_indices = [i for i, t in enumerate(topics) if t == -1]
if outlier_indices:
    outlier_analysis = []
    for idx in outlier_indices: # If you want to restrict to, say 100 outlier documents, specifiy outlier_indices[:100]:
        doc_length = len(documents[idx].split())
        doc_embedding = doc_embeddings[idx]
        topic_centroids = {}
        for topic_id in set(topics) - {-1}:
            topic_doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
            topic_centroids[topic_id] = doc_embeddings[topic_doc_indices].mean(axis=0)
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
        avg_length = outlier_df['Document_Length'].mean()
        avg_similarity = outlier_df['Similarity_to_Nearest'].mean()
        write_log(log_file, f"Outlier avg length: {avg_length:.1f} words")
        write_log(log_file, f"Outlier avg similarity to nearest topic: {avg_similarity:.3f}")
else:
    write_log(log_file, "No outliers to analyze!")

# ============================================================================
# DOCUMENT LENGTH ANALYSIS BY TOPIC
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
        'Avg_Length': np.mean(topic_doc_lengths) if topic_doc_lengths else 0,
        'Min_Length': min(topic_doc_lengths) if topic_doc_lengths else 0,
        'Max_Length': max(topic_doc_lengths) if topic_doc_lengths else 0,
        'Std_Length': np.std(topic_doc_lengths) if topic_doc_lengths else 0
    })

doc_length_df = pd.DataFrame(doc_length_by_topic)
doc_length_file = os.path.join(OUTPUT_FOLDER, "document_length_by_topic.csv")
doc_length_df.to_csv(doc_length_file, index=False)
write_log(log_file, f"Document length analysis saved to: {doc_length_file}")

# ============================================================================
# PARAMETER RECOMMENDATIONS
# ============================================================================
write_log(log_file, "Generating parameter recommendations...")
recommendations = []
if num_topics < 10:
    recommendations.append("⚠ Very few topics detected. Consider:")
    recommendations.append(f"  - Decreasing MIN_TOPIC_SIZE (currently {MIN_TOPIC_SIZE})")
    recommendations.append("  - Increasing UMAP_N_COMPONENTS for more representation")
    recommendations.append("  - Using smaller UMAP_MIN_DIST for tighter clusters")
elif num_topics > 50:
    recommendations.append(f"⚠ Many topics detected ({num_topics}). Consider:")
    recommendations.append(f"  - Increasing MIN_TOPIC_SIZE (currently {MIN_TOPIC_SIZE})")
    recommendations.append(f"  - Set MAX_TOPICS to {max(20, int(num_topics * 0.6))} for automatic reduction")
    recommendations.append("  - Examine hierarchical_topics.csv to identify mergeable topics")
outlier_ratio = list(topics).count(-1) / len(documents)
if outlier_ratio > 0.20:
    recommendations.append(f"🔴 CRITICAL: High outlier ratio ({outlier_ratio*100:.1f}%). Try:")
    recommendations.append(f"  1. Decrease MIN_TOPIC_SIZE from {MIN_TOPIC_SIZE} to {max(10, int(MIN_TOPIC_SIZE * 0.5))}")
    recommendations.append(f"  2. Decrease HDBSCAN_MIN_SAMPLES from {HDBSCAN_MIN_SAMPLES} to 1")
    recommendations.append(f"  3. Increase UMAP_MIN_DIST from {UMAP_MIN_DIST} to 0.1")
    recommendations.append("  4. Consider if documents are genuinely too diverse/noisy")
elif outlier_ratio > 0.10:
    recommendations.append(f"⚠ Moderate outlier ratio ({outlier_ratio*100:.1f}%). Consider:")
    recommendations.append(f"  - Decrease HDBSCAN_MIN_SAMPLES from {HDBSCAN_MIN_SAMPLES} to {max(1, HDBSCAN_MIN_SAMPLES - 1)}")
    recommendations.append("  - Outlier reassignment already attempted at paragraph level")
elif outlier_ratio > 0.05:
    recommendations.append(f"✓ Acceptable outlier ratio ({outlier_ratio*100:.1f}%)")
else:
    recommendations.append(f"✓ Excellent outlier ratio ({outlier_ratio*100:.1f}%)")
topic_counts_no_outlier = topic_counts[topic_counts.index != -1]
if len(topic_counts_no_outlier) > 0:
    imbalance_ratio = topic_counts_no_outlier.max() / topic_counts_no_outlier.min()
    if imbalance_ratio > 15:
        recommendations.append(f"⚠ Highly imbalanced topics (ratio {imbalance_ratio:.1f}:1). Consider:")
        recommendations.append(f"  - Increasing MIN_TOPIC_SIZE to {int(MIN_TOPIC_SIZE * 1.5)} to eliminate tiny topics")
        recommendations.append("  - Using topic reduction to merge small similar topics")
    elif imbalance_ratio > 8:
        recommendations.append(f"✓ Some topic imbalance ({imbalance_ratio:.1f}:1) - expected with real data")
if 15 <= num_topics <= 50 and outlier_ratio <= 0.15 and (len(topic_counts_no_outlier) == 0 or imbalance_ratio <= 8):
    recommendations.append("✓ Excellent configuration! Results look well-balanced.")
    recommendations.append(f"  - {num_topics} topics discovered")
    recommendations.append(f"  - {outlier_ratio*100:.1f}% outliers")
    recommendations.append(f"  - Topic balance ratio: {imbalance_ratio:.1f}:1")
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
write_log(log_file, "\n" + "="*80)
write_log(log_file, "REPRODUCIBILITY NOTE:")
write_log(log_file, f"Random seed {RANDOM_SEED} was used for all stochastic components.")
write_log(log_file, "Results should be identical across runs with the same:")
write_log(log_file, "  - Input documents and order")
write_log(log_file, "  - Python/library versions")
write_log(log_file, "  - Hardware (GPU results may vary slightly from CPU)")
if device == "cuda":
    write_log(log_file, "  - Note: Some GPU operations may have minor non-determinism")
    write_log(log_file, "           despite seeds due to parallel processing")
write_log(log_file, "="*80)

print(f"\n{'='*80}")
print(f"✓ ALL PROCESSING COMPLETE!")
print(f"✓ Discovered {num_topics} topics from {len(documents)} documents")
print(f"✓ Outliers: {list(topics).count(-1)} ({(list(topics).count(-1)/len(documents))*100:.1f}%)")
print(f"✓ Random seed: {RANDOM_SEED} (results are reproducible)")
print(f"✓ Check output folder: {OUTPUT_FOLDER}")
print(f"✓ Start with: document_topic_distributions.csv, paragraph_topic_assignments.csv, paragraph_scatter_plot.png")
print(f"{'='*80}")