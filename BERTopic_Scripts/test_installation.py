"""
GPU and Package Installation Verification Script
Save as: test_installation.py
Run: python test_installation.py
"""

import sys

print("=" * 60)
print("BERTopic Environment Installation Check")
print("=" * 60)

# Test 1: Python version
print(f"\n1. Python Version: {sys.version}")

# Test 2: Core packages
print("\n2. Checking core packages...")
packages_to_check = [
    'numpy', 'pandas', 'sklearn', 'scipy', 'nltk',
    'bertopic', 'sentence_transformers', 'umap', 'hdbscan',
    'torch', 'transformers', 'matplotlib'
]

missing_packages = []
for package in packages_to_check:
    try:
        __import__(package)
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} - NOT FOUND")
        missing_packages.append(package)

# Test 3: PyTorch and CUDA
print("\n3. PyTorch and CUDA Check...")
try:
    import torch
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Test GPU computation
        x = torch.rand(5, 5).cuda()
        print(f"   GPU Computation Test: ✓ SUCCESS")
    else:
        print("   Running on CPU (no GPU detected)")
except Exception as e:
    print(f"   Error: {e}")

# Test 4: Sentence Transformers
print("\n4. Testing Sentence Transformers...")
try:
    from sentence_transformers import SentenceTransformer
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    # Quick embedding test
    test_sentence = ["This is a test sentence"]
    embedding = model.encode(test_sentence)
    
    print(f"   Model loaded on: {device}")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   ✓ Sentence Transformers working correctly")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 5: NLTK data
print("\n5. Checking NLTK data...")
try:
    import nltk
    required_nltk_data = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
    
    for data in required_nltk_data:
        try:
            nltk.data.find(f'corpora/{data}' if data in ['stopwords', 'wordnet', 'omw-1.4'] else f'tokenizers/{data}')
            print(f"   ✓ {data}")
        except LookupError:
            print(f"   ✗ {data} - NOT FOUND (run: nltk.download('{data}'))")
except Exception as e:
    print(f"   Error: {e}")

# Test 6: BERTopic
print("\n6. Testing BERTopic...")
try:
    from bertopic import BERTopic
    print(f"   BERTopic Version: {BERTopic.__version__ if hasattr(BERTopic, '__version__') else 'Unknown'}")
    print(f"   ✓ BERTopic ready")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 7: Optional cuML (GPU-accelerated UMAP)
print("\n7. Checking cuML (optional GPU acceleration)...")
try:
    from cuml import UMAP as cumlUMAP
    print("   ✓ cuML available - GPU-accelerated UMAP enabled")
except ImportError:
    print("   ✗ cuML not available - will use CPU UMAP (this is OK)")

# Summary
print("\n" + "=" * 60)
if missing_packages:
    print(f"⚠ WARNING: {len(missing_packages)} package(s) missing:")
    for pkg in missing_packages:
        print(f"  - {pkg}")
    print("\nInstall missing packages with:")
    print(f"  conda install {' '.join(missing_packages)}")
    print("  or")
    print(f"  pip install {' '.join(missing_packages)}")
else:
    print("✓ All required packages installed successfully!")
    
print("=" * 60)