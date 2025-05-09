import os
import re
import joblib
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    BertTokenizer, 
    BertForSequenceClassification,
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification
)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import kagglehub

# Install required packages
# !pip install torch transformers datasets sklearn nltk kagglehub bs4

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to download the dataset
def download_datasets():
    """Download multiple datasets using kagglehub"""
    # Using more reliable/accessible datasets with better coverage
    datasets = {
        "politics": "linkgish/indonesian-fact-and-hoax-political-news",
        "economy": "muhammadghazimuharam/indonesiafalsenews",  # Changed to available dataset
        "education": "clmentbisaillon/fake-and-real-news-dataset",  # Common fake news dataset
        "religion": "hamzaghanmi/malawi-news-classification-challenge"  # Dataset that worked in previous run
    }
    
    # For fallback datasets in case main ones fail
    fallback_datasets = {
        "politics_fallback": "stevenpeutz/misinformation-fake-news-datasets",
        "economy_fallback": "hasanulkarim/economic-and-financial-news-w-sentiment-analysis",
        "education_fallback": "shivamb/real-or-fake-news",
        "religion_fallback": "hemanthsai7/religion-news" 
    }
    
    dataset_paths = {}
    
    # Function to try downloading a dataset with fallback
    def try_download_dataset(category, dataset_id, fallback_id=None):
        try:
            print(f"Attempting to download {category} dataset: {dataset_id}")
            path = kagglehub.dataset_download(dataset_id)
            print(f"Dataset for {category} downloaded to: {path}")
            print(f"Files in directory: {os.listdir(path)}")
            dataset_paths[category] = path
            return True
        except Exception as e:
            print(f"Error downloading {category} dataset: {e}")
            if fallback_id:
                print(f"Trying fallback dataset for {category}: {fallback_id}")
                try:
                    path = kagglehub.dataset_download(fallback_id)
                    print(f"Fallback dataset for {category} downloaded to: {path}")
                    print(f"Files in directory: {os.listdir(path)}")
                    dataset_paths[category] = path
                    return True
                except Exception as e2:
                    print(f"Error downloading fallback dataset for {category}: {e2}")
            return False
    
    # Try manual dataset creation from files if kagglehub fails
    def create_fake_dataset(category):
        """Create a simple dataset for testing when downloads fail"""
        print(f"Creating synthetic {category} dataset for testing")
        
        # Create a directory for the synthetic dataset
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix=f"synthetic_{category}_")
        
        # Create a small sample dataframe
        if category == "economy":
            # Sample business news titles (real)
            data = {
                'title': [
                    "US Economy Grows 2.5% in Q1 2023",
                    "Federal Reserve Holds Interest Rates Steady",
                    "Unemployment Rate Falls to 3.8% in April",
                    "Stock Market Reaches All-Time High",
                    "Inflation Rate Decreases to 3.2% Annually",
                    "Global Supply Chain Issues Improving, Report Says",
                    "Housing Market Shows Signs of Cooling as Mortgage Rates Rise",
                    "Corporate Profits Exceed Expectations in Q2"
                ],
                'label': [0, 0, 0, 0, 0, 0, 0, 0]  # All real news
            }
        elif category == "education":
            # Mix of real and fake educational news
            data = {
                'title': [
                    "New Study Shows Benefits of Early Childhood Education",
                    "University Enrollment Drops 5% Nationwide",
                    "BREAKING: All Student Loans to be Forgiven Next Week",
                    "Teacher Shortage Reaches Critical Levels in Rural Areas",
                    "Study Finds Link Between Homework and Improved Test Scores",
                    "Schools Ban All Textbooks in Favor of Tablets",
                    "Education Department Announces New STEM Initiative",
                    "Government to Replace All Teachers with AI by 2025"
                ],
                'label': [0, 0, 1, 0, 0, 1, 0, 1]  # Mix of real and fake
            }
        elif category == "religion":
            # Mix of real and fake religious news
            data = {
                'title': [
                    "Pope Calls for Interfaith Dialogue at Annual Conference",
                    "Survey Shows Decline in Religious Affiliation Among Young Adults",
                    "Ancient Religious Text Discovered in Cave",
                    "Religious Leaders Meet to Discuss Climate Change",
                    "Vatican Announces They've Made Contact with Aliens",
                    "Study Links Religious Participation to Longer Lifespan",
                    "Government to Ban All Religious Symbols in Public",
                    "Secret Religious Manuscript Reveals End of World Date"
                ],
                'label': [0, 0, 0, 0, 1, 0, 1, 1]  # Mix of real and fake
            }
        else:  # Default/technology
            # Mix of real and fake tech news
            data = {
                'title': [
                    "Apple Announces New iPhone Model",
                    "Google Develops AI that Can Predict Stock Market",
                    "Microsoft Releases Windows Update",
                    "New Chip Promises 500% Faster Computing",
                    "Facebook Found to Cure Cancer with New Algorithm",
                    "Tesla Unveils New Electric Vehicle",
                    "Scientists Develop First Quantum Computer for Home Use",
                    "5G Networks Expand to Rural Areas"
                ],
                'label': [0, 1, 0, 0, 1, 0, 1, 0]  # Mix of real and fake
            }
        
        # Create dataframe and save to CSV
        df = pd.DataFrame(data)
        csv_path = os.path.join(temp_dir, f"{category}_data.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"Created synthetic dataset at {temp_dir}")
        print(f"Files in directory: {os.listdir(temp_dir)}")
        
        return temp_dir
    
    # Try to download main datasets
    for category, dataset_id in datasets.items():
        fallback_id = fallback_datasets.get(f"{category}_fallback")
        success = try_download_dataset(category, dataset_id, fallback_id)
        
        # If both main and fallback fail, create synthetic dataset
        if not success:
            synthetic_path = create_fake_dataset(category)
            dataset_paths[category] = synthetic_path
    
    # Add a technology category with synthetic data if it doesn't exist
    if "technology" not in dataset_paths:
        tech_path = create_fake_dataset("technology")
        dataset_paths["technology"] = tech_path
    
    return dataset_paths

# Load and preprocess the data
def load_and_preprocess_data(dataset_paths):
    """Load and preprocess data from multiple datasets"""
    all_dfs = []
    
    # Process politics dataset
    if "politics" in dataset_paths:
        politics_path = dataset_paths["politics"]
        politics_dfs = []
        
        # Look for Excel files in Cleaned, Summarized, or RAW folders
        folders = ["Cleaned", "Summarized", "RAW"]
        for folder in folders:
            folder_path = os.path.join(politics_path, folder)
            if os.path.exists(folder_path):
                file_list = [f for f in os.listdir(folder_path) if f.endswith(".xls") or f.endswith(".xlsx")]
                
                for file_name in file_list:
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        df = pd.read_excel(file_path, engine='openpyxl')
                        # Make sure we have 'title' and 'label' columns
                        if 'title' not in df.columns and 'judul' in df.columns:
                            df['title'] = df['judul']
                        if 'label' not in df.columns and 'kategori' in df.columns:
                            # Map 'Fakta' to 0 (real) and 'Hoaks' to 1 (fake)
                            df['label'] = df['kategori'].apply(
                                lambda x: 1 if str(x).lower() == 'hoaks' else 0
                            )
                        
                        df["source"] = file_name
                        df["dataset_category"] = "politics"  # Explicitly mark the source dataset
                        politics_dfs.append(df)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        
        if politics_dfs:
            politics_df = pd.concat(politics_dfs, ignore_index=True)
            all_dfs.append(politics_df)
            print(f"Politics dataset: {len(politics_df)} entries")
    
    # Process economy dataset
    if "economy" in dataset_paths:
        econ_path = dataset_paths["economy"]
        econ_dfs = []
        
        # Check for different file types
        for ext in ['.csv', '.xlsx', '.xls']:
            files = [f for f in os.listdir(econ_path) if f.endswith(ext)]
            for file in files:
                file_path = os.path.join(econ_path, file)
                try:
                    if ext == '.csv':
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_excel(file_path, engine='openpyxl')
                    
                    # For the business Indonesian dataset specific mappings
                    if 'category' in df.columns:
                        # Standardize to binary classification (real news)
                        df['label'] = 0  # All business news assumed to be real
                    
                    if 'title' not in df.columns:
                        if 'judul' in df.columns:  # Indonesian
                            df['title'] = df['judul']
                        elif 'berita' in df.columns:  # Indonesian
                            df['title'] = df['berita']
                    
                    if 'title' not in df.columns:
                        # Try to find other potential title columns
                        potential_title_cols = ['headline', 'header', 'news_title', 'Title']
                        for col in potential_title_cols:
                            if col in df.columns:
                                df['title'] = df[col]
                                break
                    
                    # Ensure required columns exist
                    if 'title' in df.columns:  # Only add if we have a title column
                        df["source"] = file
                        df["dataset_category"] = "economy"
                        econ_dfs.append(df[['title', 'label', 'source', 'dataset_category']])
                    else:
                        print(f"Warning: No title column found in {file}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        if econ_dfs:
            # Concatenate all economy dataframes
            econ_df = pd.concat(econ_dfs, ignore_index=True)
            
            # Check if we have label column, if not (all real news) add it
            if 'label' not in econ_df.columns:
                econ_df['label'] = 0  # Assuming business news from official sources are real
            
            all_dfs.append(econ_df)
            print(f"Economy dataset: {len(econ_df)} entries")
    
    # Process education dataset
    if "education" in dataset_paths:
        edu_path = dataset_paths["education"]
        edu_dfs = []
        
        # Look specifically for True.csv and Fake.csv
        true_path = os.path.join(edu_path, 'True.csv')
        fake_path = os.path.join(edu_path, 'Fake.csv')
        
        if os.path.exists(true_path):
            try:
                true_df = pd.read_csv(true_path)
                # Assign label 0 for real news
                true_df['label'] = 0
                # Use title or text as the title
                if 'title' not in true_df.columns and 'text' in true_df.columns:
                    true_df['title'] = true_df['text'].apply(lambda x: x[:100])  # Take first 100 chars as title
                true_df["source"] = 'True.csv'
                true_df["dataset_category"] = "education"
                edu_dfs.append(true_df)
            except Exception as e:
                print(f"Error reading {true_path}: {e}")
                
        if os.path.exists(fake_path):
            try:
                fake_df = pd.read_csv(fake_path)
                # Assign label 1 for fake news
                fake_df['label'] = 1
                # Use title or text as the title
                if 'title' not in fake_df.columns and 'text' in fake_df.columns:
                    fake_df['title'] = fake_df['text'].apply(lambda x: x[:100])  # Take first 100 chars as title
                fake_df["source"] = 'Fake.csv'
                fake_df["dataset_category"] = "education"
                edu_dfs.append(fake_df)
            except Exception as e:
                print(f"Error reading {fake_path}: {e}")
        
        # Check for any other CSV files in the directory
        if not edu_dfs:
            csv_files = [f for f in os.listdir(edu_path) if f.endswith('.csv')]
            for file in csv_files:
                file_path = os.path.join(edu_path, file)
                try:
                    df = pd.read_csv(file_path)
                    # Check if this is likely a fake news dataset with appropriate columns
                    if 'label' in df.columns or 'fake' in df.columns or 'real' in df.columns:
                        # If no explicit label, look for columns to determine
                        if 'label' not in df.columns:
                            for col in ['fake', 'is_fake', 'is_real']:
                                if col in df.columns:
                                    if col == 'is_real':
                                        df['label'] = df[col].apply(lambda x: 0 if x == 1 else 1)
                                    else:
                                        df['label'] = df[col].apply(lambda x: 1 if x == 1 else 0)
                                    break
                    
                    # Ensure we have a title
                    if 'title' not in df.columns:
                        # Try to find or construct a title
                        potential_title_cols = ['text', 'content', 'news', 'article']
                        for col in potential_title_cols:
                            if col in df.columns:
                                df['title'] = df[col].apply(lambda x: str(x)[:100])
                                break
                    
                    if 'title' in df.columns and 'label' in df.columns:
                        df["source"] = file
                        df["dataset_category"] = "education"
                        edu_dfs.append(df[['title', 'label', 'source', 'dataset_category']])
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        if edu_dfs:
            # Concatenate all education dataframes
            edu_df = pd.concat(edu_dfs, ignore_index=True)
            all_dfs.append(edu_df)
            print(f"Education dataset: {len(edu_df)} entries")
    
    # Process religion dataset
    if "religion" in dataset_paths:
        religion_path = dataset_paths["religion"]
        religion_dfs = []
        
        # Look for CSV files
        csv_files = [f for f in os.listdir(religion_path) if f.endswith('.csv')]
        for file in csv_files:
            file_path = os.path.join(religion_path, file)
            try:
                df = pd.read_csv(file_path)
                
                # Check for label columns
                if 'label' not in df.columns:
                    for col in ['fake', 'is_fake', 'class']:
                        if col in df.columns:
                            df['label'] = df[col].apply(lambda x: 1 if x == 1 or x == True else 0)
                            break
                
                # Check for title columns
                if 'title' not in df.columns:
                    for col in ['text', 'content', 'news', 'headline']:
                        if col in df.columns:
                            df['title'] = df[col].apply(lambda x: str(x)[:100])
                            break
                
                # Only include if we have both title and label
                if 'title' in df.columns and 'label' in df.columns:
                    df["source"] = file
                    df["dataset_category"] = "religion"
                    religion_dfs.append(df[['title', 'label', 'source', 'dataset_category']])
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # For h5 files (if there are any in the religion dataset)
        h5_files = [f for f in os.listdir(religion_path) if f.endswith('.h5')]
        for file in h5_files:
            file_path = os.path.join(religion_path, file)
            try:
                # Try using pandas to read h5 file
                df = pd.read_hdf(file_path)
                
                # Check for label columns
                if 'label' not in df.columns:
                    for col in ['fake', 'is_fake', 'class']:
                        if col in df.columns:
                            df['label'] = df[col].apply(lambda x: 1 if x == 1 or x == True else 0)
                            break
                
                # Check for title columns
                if 'title' not in df.columns:
                    for col in ['text', 'content', 'news', 'headline']:
                        if col in df.columns:
                            df['title'] = df[col].apply(lambda x: str(x)[:100])
                            break
                
                # Only include if we have both title and label
                if 'title' in df.columns and 'label' in df.columns:
                    df["source"] = file
                    df["dataset_category"] = "religion"
                    religion_dfs.append(df[['title', 'label', 'source', 'dataset_category']])
            except Exception as e:
                print(f"Error reading h5 file {file_path}: {e}")
        
        if religion_dfs:
            # Concatenate all religion dataframes
            religion_df = pd.concat(religion_dfs, ignore_index=True)
            all_dfs.append(religion_df)
            print(f"Religion dataset: {len(religion_df)} entries")
    
    # Merge all datasets
    if not all_dfs:
        raise ValueError("No valid datasets were loaded")
    
    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total data after merging all datasets: {len(merged_df)}")
    
    # Ensure required columns exist
    required_cols = ['title', 'label']
    for col in required_cols:
        if col not in merged_df.columns:
            raise ValueError(f"Required column '{col}' not found in merged dataset")
    
    # Drop rows with NaN in essential columns
    merged_df.dropna(subset=required_cols, inplace=True)
    
    # Convert label to float for consistency
    merged_df['label'] = merged_df['label'].astype(float)
    
    # If dataset_category is not already specified, infer it from content
    merged_df['category'] = merged_df['dataset_category']
    
    # Clean text
    merged_df['cleaned_title'] = merged_df['title'].astype(str).apply(clean_text)
    
    # Display category distribution after loading
    print("\nCategory distribution after processing:")
    print(merged_df['category'].value_counts())
    
    # Display label distribution after loading
    print("\nLabel distribution after processing:")
    print(merged_df['label'].value_counts())
    print(f"Label distribution percentage: {merged_df['label'].value_counts(normalize=True) * 100}")
    
    return merged_df

def categorize_news(row):
    """Categorize news based on keywords in title and content"""
    # Handle both Indonesian and English keywords
    text = str(row['title']).lower()
    if 'content' in row and pd.notna(row['content']):
        text += " " + str(row['content']).lower()
    
    # Define category keywords in both Indonesian and English
    politics_keywords = [
        # Indonesian
        'politik', 'pemilu', 'presiden', 'pemerintah', 'partai', 'menteri', 'dpr', 'gubernur', 'walikota',
        # English
        'politic', 'election', 'president', 'government', 'party', 'minister', 'governor', 'mayor',
        'democrat', 'republican', 'parliament', 'congress', 'senate', 'vote', 'campaign', 'ballot',
        'legislative', 'administration', 'cabinet'
    ]
    
    economy_keywords = [
        # Indonesian
        'ekonomi', 'bisnis', 'rupiah', 'saham', 'bank', 'inflasi', 'investasi', 'keuangan', 'pasar', 'bursa',
        # English
        'economy', 'business', 'stock', 'market', 'finance', 'inflation', 'investment', 'dollar', 'euro',
        'trade', 'fiscal', 'monetary', 'gdp', 'recession', 'economic', 'financial', 'budget', 'debt',
        'loan', 'interest rate', 'banking', 'commerce', 'corporation', 'industry'
    ]
    
    education_keywords = [
        # Indonesian
        'pendidikan', 'sekolah', 'universitas', 'kuliah', 'belajar', 'siswa', 'mahasiswa', 'ujian',
        # English
        'education', 'school', 'university', 'college', 'student', 'learn', 'academic', 'teacher',
        'professor', 'exam', 'degree', 'course', 'campus', 'classroom', 'curriculum', 'study',
        'educational', 'scholarship', 'graduate', 'faculty', 'lecture', 'tuition'
    ]
    
    religion_keywords = [
        # Indonesian
        'agama', 'islam', 'kristen', 'hindu', 'buddha', 'masjid', 'gereja', 'ibadah', 'doa',
        # English
        'religion', 'religious', 'faith', 'church', 'mosque', 'temple', 'prayer', 'god', 'christian',
        'muslim', 'islamic', 'hindu', 'buddhist', 'worship', 'scripture', 'holy', 'spiritual',
        'priest', 'imam', 'pastor', 'ritual', 'sermon', 'bible', 'quran', 'torah'
    ]
    
    technology_keywords = [
        # Indonesian
        'teknologi', 'digital', 'internet', 'aplikasi', 'komputer', 'smartphone', 'inovasi', 'perangkat',
        # English
        'technology', 'tech', 'digital', 'internet', 'software', 'hardware', 'app', 'computer',
        'smartphone', 'innovation', 'device', 'online', 'artificial intelligence', 'ai', 'machine learning',
        'cyber', 'coding', 'programming', 'network', 'data', 'algorithm', 'robot', 'automation',
        'silicon valley', 'startup', 'gadget', 'mobile', 'web', 'cloud', 'electronic', 'semiconductor',
        'biotech', 'blockchain', 'cryptocurrency', 'virtual reality', 'vr', 'augmented reality', 'ar'
    ]
    
    # Check for keywords in text with score-based approach
    category_scores = {
        'politics': sum(1 for keyword in politics_keywords if keyword in text),
        'economy': sum(1 for keyword in economy_keywords if keyword in text),
        'education': sum(1 for keyword in education_keywords if keyword in text),
        'religion': sum(1 for keyword in religion_keywords if keyword in text),
        'technology': sum(1 for keyword in technology_keywords if keyword in text)
    }
    
    # Use the highest scoring category if score is > 0
    max_category = max(category_scores.items(), key=lambda x: x[1])
    if max_category[1] > 0:
        return max_category[0]
    else:
        return 'other'

def clean_text(text):
    """Clean text from non-letter characters and convert to lowercase"""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def advanced_clean_text(text):
    """Advanced text cleaning with stopword removal"""
    # Basic cleaning
    text = clean_text(text)
    
    # Tokenize
    word_tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    filtered_text = [word for word in word_tokens if word not in stop_words]
    
    return ' '.join(filtered_text)

def get_headline_from_url(url):
    """Extract news headline from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Search for common headline tags
        headline_tags = ['h1', 'h2', '.headline', '.title', 'title']
        for tag in headline_tags:
            if tag.startswith('.'):
                headline = soup.select_one(tag)
            else:
                headline = soup.find(tag)
                
            if headline:
                return headline.text.strip()

        return "Headline not found."
    except requests.exceptions.RequestException as e:
        return f"Failed to retrieve data: {e}"

# Traditional ML approach with TF-IDF + Naive Bayes
def train_traditional_model(df, category=None):
    """Train a traditional ML model (TF-IDF + Naive Bayes)"""
    # Filter by category if specified
    if category and category != 'all':
        df = df[df['category'] == category]
        if len(df) < 10:
            print(f"Not enough data for category: {category}")
            return None, None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_title'], df['label'], test_size=0.2, random_state=42
    )
    
    # Create pipeline
    model = make_pipeline(TfidfVectorizer(max_features=5000), MultinomialNB())
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    
    print(f"Traditional Model Results for {'all categories' if not category or category == 'all' else category}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(report)
    
    return model, accuracy, f1

# BERT Model dataset class
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten() if 'token_type_ids' in encoding else None,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Function to train transformer models
def train_transformer_model(df, model_name, category=None):
    """Train a transformer model (BERT, mBERT, XLM-R)"""
    # Filter by category if specified
    if category and category != 'all':
        df = df[df['category'] == category]
        if len(df) < 10:
            print(f"Not enough data for category: {category}")
            return None, None, None
    
    # Define label mapping
    unique_labels = df['label'].unique()
    label_map = {label: i for i, label in enumerate(unique_labels)}
    df['label_id'] = df['label'].map(label_map)
    
    # Split data
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Initialize tokenizer and model
    if 'bert-base-multilingual' in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=len(unique_labels)
        )
    elif 'xlm-roberta' in model_name:
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        model = XLMRobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=len(unique_labels)
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(unique_labels)
        )
    
    # Create datasets
    train_dataset = NewsDataset(train_df['cleaned_title'], train_df['label_id'], tokenizer)
    test_dataset = NewsDataset(test_df['cleaned_title'], test_df['label_id'], tokenizer)
    
    # Define training arguments - Fix the error with evaluation_strategy
    training_args = TrainingArguments(
        output_dir=f'./results_{model_name.replace("/", "_")}',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        # Fix for older Transformers versions that don't support evaluation_strategy
        # Check if the version supports evaluation_strategy
        **({'evaluation_strategy': 'epoch', 'save_strategy': 'epoch'} 
           if hasattr(TrainingArguments, 'evaluation_strategy') else {}),
        load_best_model_at_end=True,
    )
    
    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate
    eval_result = trainer.evaluate()
    
    # Make predictions
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    
    # Convert label IDs back to original labels
    id_to_label = {v: k for k, v in label_map.items()}
    true_labels = [id_to_label[id] for id in test_df['label_id'].values]
    pred_labels = [id_to_label[id] for id in preds]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    report = classification_report(true_labels, pred_labels)
    
    print(f"{model_name} Results for {'all categories' if not category or category == 'all' else category}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(report)
    
    # Save model and tokenizer
    model_save_path = f'./saved_models/{model_name.replace("/", "_")}'
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    return model, accuracy, f1

def predict_with_traditional(model, text, category=None):
    """Predict using traditional ML model"""
    cleaned_text = clean_text(text)
    prediction = model.predict([cleaned_text])[0]
    return prediction

def predict_with_transformer(model_path, text):
    """Predict using transformer model"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=1).item()
    
    # Get label from model config
    id_to_label = {i: label for i, label in enumerate(model.config.id2label.values())}
    return id_to_label[predicted_class]

def run_full_analysis():
    """Run full analysis with all models and categories"""
    # Download multiple datasets
    dataset_paths = download_datasets()
    
    # Load and preprocess data
    df = load_and_preprocess_data(dataset_paths)
    
    # Summary of data by category and dataset source
    print("\nCategory distribution:")
    print(df['category'].value_counts())
    
    print("\nDataset source distribution:")
    if 'dataset_category' in df.columns:
        print(df['dataset_category'].value_counts())
    
    # Check for class imbalance
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"Label distribution percentage: {df['label'].value_counts(normalize=True) * 100}")
    
    # Results storage
    results = []
    
    # Create directory for model storage
    if not os.path.exists('./models'):
        os.makedirs('./models')
    
    # Train traditional model on all data
    print("\n===== Training Traditional Model (TF-IDF + Naive Bayes) on All Data =====")
    trad_model, trad_acc, trad_f1 = train_traditional_model(df, 'all')
    results.append({
        'model': 'TF-IDF + NB',
        'category': 'all',
        'accuracy': trad_acc,
        'f1_score': trad_f1
    })
    
    # Save the model
    if trad_model is not None:
        joblib.dump(trad_model, './models/traditional_model_all.pkl')
        print("Traditional model saved to ./models/traditional_model_all.pkl")
    
    # Train on individual categories
    categories = ['politics', 'economy', 'education', 'religion', 'technology', 'other']
    for category in categories:
        category_df = df[df['category'] == category]
        if len(category_df) >= 100:  # Only train if we have enough data
            print(f"\n===== Training Traditional Model on {category.upper()} Category =====")
            print(f"Data size: {len(category_df)} entries")
            cat_model, cat_acc, cat_f1 = train_traditional_model(category_df, 'all')
            
            if cat_model is not None:
                results.append({
                    'model': 'TF-IDF + NB',
                    'category': category,
                    'accuracy': cat_acc,
                    'f1_score': cat_f1
                })
                
                # Save category-specific model
                joblib.dump(cat_model, f'./models/traditional_model_{category}.pkl')
                print(f"Category model saved to ./models/traditional_model_{category}.pkl")
        else:
            print(f"\nSkipping {category} category due to insufficient data ({len(category_df)} entries)")
    
    # Train transformer models - only if enough data is available
    if len(df) >= 1000:  # Transformers need more data for meaningful training
        transformer_models = {
            'bert': 'indolem/indobert-base-uncased',  # For Indonesian
            'mbert': 'bert-base-multilingual-cased',  # Good for mixed language
            'xlm-r': 'xlm-roberta-base'               # State-of-the-art multilingual
        }
        
        for model_type, model_name in transformer_models.items():
            print(f"\n===== Training {model_type.upper()} Model on All Categories =====")
            
            try:
                trans_model, trans_acc, trans_f1 = train_transformer_model(df, model_name, 'all')
                
                if trans_model is not None:
                    results.append({
                        'model': model_type.upper(),
                        'category': 'all',
                        'accuracy': trans_acc,
                        'f1_score': trans_f1
                    })
            except Exception as e:
                print(f"Error training {model_type} model: {e}")
                continue
            
            # Optional: Train on individual categories with large enough data
            for category in categories:
                category_df = df[df['category'] == category]
                if len(category_df) >= 500:  # Need substantial data for transformers per category
                    print(f"\n===== Training {model_type.upper()} model on {category.upper()} category =====")
                    try:
                        cat_trans_model, cat_trans_acc, cat_trans_f1 = train_transformer_model(
                            category_df, model_name, 'all'
                        )
                        
                        if cat_trans_model is not None:
                            results.append({
                                'model': f"{model_type.upper()}_{category}",
                                'category': category,
                                'accuracy': cat_trans_acc,
                                'f1_score': cat_trans_f1
                            })
                    except Exception as e:
                        print(f"Error training {model_type} model for {category}: {e}")
    else:
        print("\nSkipping transformer models due to insufficient data")
        print("Transformers typically need 1000+ examples for effective training")
    
    # Create results dataframe and save to CSV
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='f1_score', ascending=False)
    print("\nFinal Results Summary (sorted by F1 score):")
    print(results_df)
    
    # Save results
    results_df.to_csv('./model_performance_results.csv', index=False)
    print("\nResults saved to model_performance_results.csv")
    
    # Plot performance comparison
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='model', y='f1_score', hue='category', data=results_df)
        plt.title('Model Performance Comparison (F1 Score)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('./model_performance_comparison.png')
        print("Performance comparison plot saved to model_performance_comparison.png")
    except Exception as e:
        print(f"Could not generate performance plot: {e}")
    
    return results_df

def main():
    """Main function to run the program"""
    print("Indonesian Fake News Detection System")
    print("====================================")
    
    choice = input("Select an option:\n1. Train all models and analyze results\n2. Check a news headline\nEnter choice (1 or 2): ")
    
    if choice == '1':
        results = run_full_analysis()
        print("\nAnalysis complete!")
        
    elif choice == '2':
        # Load traditional model for quick prediction
        try:
            trad_model = joblib.load('traditional_model.pkl')
        except:
            print("Model not found. Training a new model...")
            path = download_dataset()
            df = load_and_preprocess_data(path)
            trad_model, _, _ = train_traditional_model(df)
            
        # Get news headline
        url_or_headline = input("Enter news URL or headline text: ")
        
        if url_or_headline.startswith(('http://', 'https://')):
            headline = get_headline_from_url(url_or_headline)
            if "Failed to retrieve data" in headline:
                print(headline)
                headline = input("Please enter the headline manually: ")
        else:
            headline = url_or_headline
            
        # Make prediction
        category = categorize_news({'title': headline, 'content': ''})
        prediction = predict_with_traditional(trad_model, headline)
        
        print(f"\nHeadline: {headline}")
        print(f"Detected Category: {category}")
        print(f"Prediction: {'Hoax' if prediction == 1 else 'Not Hoax'}")
        
    else:
        print("Invalid choice. Please run the program again.")

if __name__ == "__main__":
    main()
