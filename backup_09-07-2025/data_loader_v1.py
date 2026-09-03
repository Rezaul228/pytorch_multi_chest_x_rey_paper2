import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from PIL import Image
import pickle
import glob
import gc
import paths
import re
from collections import defaultdict

class SimpleTokenizer:
    def __init__(self, oov_token="<unk>"):
        self.word2idx = {"<pad>": 0, oov_token: 1}
        self.idx2word = {0: "<pad>", 1: oov_token}
        self.word_counts = defaultdict(int)
        self.oov_token = oov_token
        self.num_words = None

    def fit_on_texts(self, texts):
        """Build vocabulary from list of texts"""
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                self.word_counts[word.lower()] += 1
        
        # Sort words by frequency
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add words to vocabulary
        for word, count in sorted_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def texts_to_sequences(self, texts, maxlen=115):
        """Convert texts to sequences with padding"""
        sequences = []
        for text in texts:
            words = self._tokenize(text)
            sequence = []
            for word in words[:maxlen]:
                if word in self.word2idx:
                    sequence.append(self.word2idx[word])
                else:
                    sequence.append(self.word2idx[self.oov_token])
            
            # Pad sequence to maxlen
            if len(sequence) < maxlen:
                sequence.extend([0] * (maxlen - len(sequence)))
            
            sequences.append(sequence)
        return torch.tensor(sequences, dtype=torch.long)
    
    def texts_to_sequences_compatible(self, texts, maxlen=115):
        """Compatible version that handles both tokenizer types"""
        if hasattr(self, 'word_index'):
            # EnhancedTokenizer - returns list of lists
            sequences = self.texts_to_sequences(texts)
            # Pad sequences to maxlen
            padded_sequences = []
            for seq in sequences:
                if len(seq) > maxlen:
                    padded_sequences.append(seq[:maxlen])
                else:
                    padded_seq = seq + [0] * (maxlen - len(seq))
                    padded_sequences.append(padded_seq)
            return torch.tensor(padded_sequences, dtype=torch.long)
        else:
            # SimpleTokenizer - already returns torch tensor
            return self.texts_to_sequences(texts, maxlen)
    
    def _tokenize(self, text):
        """Simple tokenization by splitting on whitespace and removing punctuation"""
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def __len__(self):
        """Return vocabulary size"""
        return len(self.word2idx)

class IndianaDataset(Dataset):
    def __init__(self, shard_paths, max_samples=None):
        self.shard_paths = shard_paths
        self.max_samples = max_samples
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        samples_loaded = 0
        for shard_path in self.shard_paths:
            if self.max_samples and samples_loaded >= self.max_samples:
                break
                
            with open(shard_path, 'rb') as f:
                shard_data = pickle.load(f)
            
            shard_size = len(shard_data['images'])
            for i in range(shard_size):
                if self.max_samples and samples_loaded >= self.max_samples:
                    break
                    
                self.samples.append({
                    'images': shard_data['images'][i],
                    'captions': shard_data['captions'][i],
                    'study_ids': shard_data['study_ids'][i]
                })
                samples_loaded += 1
            
            del shard_data
            gc.collect()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'images': torch.FloatTensor(sample['images']),
            'captions': torch.LongTensor(sample['captions']),
            'study_ids': sample['study_ids']
        }

class IndianaDataLoader:
    def __init__(self, batch_size=32, use_shards=True, shard_size=100, shard_subfolder="indiana_shards"):
        self.batch_size = batch_size
        self.data = None
        self.tokenizer = None
        self.dataset = None
        self.use_shards = use_shards
        self.shard_size = shard_size
        self.shard_subfolder = shard_subfolder
        
        # Use centralized path configuration
        self.shard_dir = paths.get_shard_base_path(shard_subfolder)
        self.metadata_path = paths.get_metadata_path(shard_subfolder)
        
        # Print current configuration
        paths.print_current_config(shard_subfolder)
        
        # Legacy paths for backward compatibility (deprecated)
        # These are only used for MIMIC data processing from raw data
        self.mimic_base_path = paths.MIMIC_DATA_PATH
        self.metadata_csv_path = os.path.join(self.mimic_base_path, "metadata.csv")
        self.image_dir = os.path.join(self.mimic_base_path, "images")
        self.reports_dir = os.path.join(self.mimic_base_path, "reports")
        
        self.mimic_data = None
    
    def load_data(self, max_samples=None, skip_processing=False):
        shards_exist = os.path.exists(self.metadata_path)
        
        if skip_processing and shards_exist:
            print("Shards already exist. Loading from existing shards...")
            try:
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.tokenizer = metadata.get('tokenizer')
                if self.tokenizer is None:
                    raise ValueError("No tokenizer found in metadata")
                
                # Handle EnhancedTokenizer compatibility
                if hasattr(self.tokenizer, 'word_index') and not hasattr(self.tokenizer, 'word2idx'):
                    # Add compatibility attributes for EnhancedTokenizer
                    self.tokenizer.word2idx = self.tokenizer.word_index
                    self.tokenizer.idx2word = self.tokenizer.index_word
                    print(f"EnhancedTokenizer detected - added compatibility attributes")
                    print(f"Vocabulary size: {len(self.tokenizer.word2idx)}")
                
                print("Successfully loaded existing tokenizer.")
            except Exception as e:
                print(f"Warning: Could not load existing tokenizer ({e}). Creating new one...")
                # Create a new tokenizer with a reasonable vocabulary
                self.tokenizer = SimpleTokenizer(oov_token="<unk>")
                # Build a basic vocabulary (this will be fine for loading existing shards)
                basic_vocab = ["the", "and", "or", "with", "without", "chest", "x-ray", "normal", "abnormal", 
                              "lung", "heart", "impression", "findings", "patient", "examination", "view", "shows"]
                self.tokenizer.fit_on_texts(basic_vocab)
                print("Created new tokenizer with basic vocabulary.")
            
            self.data = {'images': [], 'captions': [], 'study_ids': []}
            print("Tokenizer ready. Training data will be loaded by get_data() method.")
        else:
            print(f"Processing {self.shard_subfolder} data from scratch...")
            self._process_mimic_data(max_samples)
            
        print(f"Loaded {len(self.data['images'])} samples from {self.shard_subfolder} dataset")
        gc.collect()
        return self.data
    
    def _process_mimic_data(self, max_samples=None):
        print(f"Reading metadata from: {self.metadata_csv_path}")
        print(f"Image directory: {self.image_dir}")
        print(f"Reports directory: {self.reports_dir}")
        
        df = pd.read_csv(self.metadata_csv_path)
        
        if max_samples:
            df = df.head(max_samples)
        
        print(f"Processing {len(df)} {self.shard_subfolder} samples using streaming approach...")
        
        sample_texts = self._create_tokenizer_from_sample(df.head(min(1000, len(df))))
        self._create_shards_streaming(df)
        
        total_samples = len(df)
        self.data = {
            'images': [],
            'captions': [],
            'study_ids': []
        }
        
        print(f"Streaming processing completed. {total_samples} samples processed into shards.")
        print("Data shards created - actual data will be loaded on-demand during training")
        gc.collect()

    def _create_tokenizer_from_sample(self, sample_df):
        print("Creating tokenizer from sample data...")
        
        sample_texts = []
        for idx, row in sample_df.iterrows():
            report_path = os.path.join(self.reports_dir, row['report_file'])
            if os.path.exists(report_path):
                try:
                    with open(report_path, 'r', encoding='utf-8') as f:
                        report_text = f.read().strip()
                    sample_texts.append(report_text)
                except Exception as e:
                    continue
        
        self.tokenizer = SimpleTokenizer(oov_token="<unk>")
        self.tokenizer.fit_on_texts(sample_texts)
        
        vocab_size = len(self.tokenizer)
        print(f"Tokenizer created. Vocabulary size: {vocab_size}")
        
        return sample_texts

    def _create_shards_streaming(self, df):
        print(f"Creating shards in: {self.shard_dir}")
        os.makedirs(self.shard_dir, exist_ok=True)
        
        train_dir = paths.get_train_shards_dir(self.shard_subfolder)
        val_dir = paths.get_val_shards_dir(self.shard_subfolder)
        test_dir = paths.get_test_shards_dir(self.shard_subfolder)
        
        for d in [train_dir, val_dir, test_dir]:
            os.makedirs(d, exist_ok=True)
        
        unique_study_ids = df['study_id'].unique()
        np.random.seed(42)
        shuffled_ids = np.random.permutation(unique_study_ids)
        
        train_idx = int(0.8 * len(shuffled_ids))
        val_idx = int(0.9 * len(shuffled_ids))
        
        train_study_ids = set(shuffled_ids[:train_idx])
        val_study_ids = set(shuffled_ids[train_idx:val_idx])
        test_study_ids = set(shuffled_ids[val_idx:])
        
        print(f"Study ID splits - Train: {len(train_study_ids)}, Val: {len(val_study_ids)}, Test: {len(test_study_ids)}")
        
        train_count = self._process_split_streaming(df, train_study_ids, train_dir, 'train')
        val_count = self._process_split_streaming(df, val_study_ids, val_dir, 'val')  
        test_count = self._process_split_streaming(df, test_study_ids, test_dir, 'test')
        
        metadata = {
            'tokenizer': self.tokenizer,
            'vocab_size': len(self.tokenizer),
            'total_samples': len(df),
            'train_samples': train_count,
            'val_samples': val_count,
            'test_samples': test_count,
            'train_study_ids': list(train_study_ids),
            'val_study_ids': list(val_study_ids),
            'test_study_ids': list(test_study_ids),
            'shard_subfolder': self.shard_subfolder
        }
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"Data splits completed - Train: {train_count}, Val: {val_count}, Test: {test_count}")

    def _process_split_streaming(self, df, study_ids_set, split_dir, split_name):
        print(f"Processing {split_name} split...")
        
        split_df = df[df['study_id'].isin(study_ids_set)]
        
        shard_images = []
        shard_captions = []
        shard_study_ids = []
        shard_count = 0
        processed_count = 0
        
        for idx, row in split_df.iterrows():
            study_id = row['study_id']
            img_path = os.path.join(self.image_dir, row['image_file'])
            report_path = os.path.join(self.reports_dir, row['report_file'])
            
            if os.path.exists(img_path) and os.path.exists(report_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    
                    with open(report_path, 'r', encoding='utf-8') as f:
                        report_text = f.read().strip()
                    
                    # Convert text to sequence using tokenizer (returns tensor)
                    sequence_tensor = self.tokenizer.texts_to_sequences([report_text])
                    # Convert tensor to numpy array for storage
                    caption = sequence_tensor.numpy()[0]  # Get first (and only) sequence
                    
                    shard_images.append(img_array)
                    shard_captions.append(caption)
                    shard_study_ids.append(study_id)
                    processed_count += 1
                    
                    if len(shard_images) >= self.shard_size:
                        self._save_shard_streaming(shard_images, shard_captions, shard_study_ids, 
                                                  split_dir, shard_count)
                        
                        shard_images = []
                        shard_captions = []
                        shard_study_ids = []
                        shard_count += 1
                        gc.collect()
                        
                        if processed_count % 500 == 0:
                            print(f"  {split_name}: Processed {processed_count} samples, created {shard_count} shards")
                    
                except Exception as e:
                    print(f"Error processing {study_id}: {e}")
                    continue
        
        if len(shard_images) > 0:
            self._save_shard_streaming(shard_images, shard_captions, shard_study_ids, 
                                      split_dir, shard_count)
            shard_count += 1
        
        print(f"{split_name} split completed: {processed_count} samples in {shard_count} shards")
        return processed_count

    def _save_shard_streaming(self, images, captions, study_ids, split_dir, shard_idx):
        shard_data = {
            'images': np.array(images),
            'captions': np.array(captions),  # Now all captions are numpy arrays
            'study_ids': np.array(study_ids)
        }
        
        shard_path = os.path.join(split_dir, f'shard_{shard_idx:04d}.pkl')
        with open(shard_path, 'wb') as f:
            pickle.dump(shard_data, f)
        
        del shard_data
        gc.collect()
    
    def preprocess_captions(self):
        print("Caption preprocessing already completed in IndianaDatasetLoader")
        return
    
    def get_data(self, max_samples=None):
        train_dir = paths.get_train_shards_dir(self.shard_subfolder)
        train_shards = sorted(glob.glob(os.path.join(train_dir, '*.pkl')))
        
        if not train_shards:
            raise FileNotFoundError(f"No training shards found in {train_dir}")
        
        print(f"Creating PyTorch Dataset from {len(train_shards)} shards")
        print(f"   Max samples: {max_samples if max_samples else 'ALL'}")
        
        dataset = IndianaDataset(train_shards, max_samples)
        total_samples = len(dataset)
        self.dataset_size = total_samples
        
        self.data = {
            'images': np.array([]),
            'captions': np.array([]),
            'study_ids': np.array([])
        }
        
        print(f"PyTorch Dataset created with {total_samples} samples")
        print(f"   Memory usage: ~{self._estimate_memory_usage():.1f} MB (vs ~{total_samples * 0.6:.0f} MB for full loading)")
        
        return dataset
    
    def _count_total_samples(self, shard_paths, max_samples=None):
        total = 0
        for shard_path in shard_paths:
            if max_samples and total >= max_samples:
                break
                
            with open(shard_path, 'rb') as f:
                shard_data = pickle.load(f)
                shard_size = len(shard_data['images'])
                
                if max_samples:
                    remaining_needed = min(max_samples - total, shard_size)
                    total += remaining_needed
                else:
                    total += shard_size
                    
                del shard_data
                gc.collect()
        
        return total
    
    def _estimate_memory_usage(self):
        return 50.0
    
    def get_validation_data(self, num_samples=None):
        val_dir = paths.get_val_shards_dir(self.shard_subfolder)
        val_shards = sorted(glob.glob(os.path.join(val_dir, '*.pkl')))
        
        if not val_shards:
            raise FileNotFoundError(f"No validation shards found in {val_dir}")
        
        print(f"Creating PyTorch Validation Dataset from {len(val_shards)} shards")
        print(f"   Max samples: {num_samples if num_samples else 'ALL'}")
        
        val_dataset = IndianaDataset(val_shards, num_samples)
        total_val_samples = len(val_dataset)
        
        print(f"PyTorch Validation Dataset created with {total_val_samples} samples")
        
        return val_dataset

    def get_test_data(self, num_samples=None):
        test_dir = paths.get_test_shards_dir(self.shard_subfolder)
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory {test_dir} not found. Run training without --skip_processing once.")
        
        test_shards = sorted(glob.glob(os.path.join(test_dir, '*.pkl')))
        if not test_shards:
            raise FileNotFoundError(f"No test shards found in {test_dir}.")
        
        print(f"Creating PyTorch Test Dataset from {len(test_shards)} shards")
        print(f"   Max samples: {num_samples if num_samples else 'ALL'}")
        
        test_dataset = IndianaDataset(test_shards, num_samples)
        total_test_samples = len(test_dataset)
        
        print(f"PyTorch Test Dataset created with {total_test_samples} samples")
        
        return test_dataset

    def get_validation_data_for_training(self, num_samples=2500):
        print(f"Loading validation data for training (numpy arrays, {num_samples} samples)")
        
        val_dataset = self.get_validation_data(num_samples=num_samples)
        
        all_images = []
        all_captions = []
        all_study_ids = []
        
        for i, sample in enumerate(val_dataset):
            if num_samples is not None and i >= num_samples:
                break
                
            all_images.append(sample['images'].numpy())
            all_captions.append(sample['captions'].numpy())
            
            study_id = sample['study_ids']
            if isinstance(study_id, bytes):
                study_id = study_id.decode('utf-8')
            else:
                study_id = str(study_id)
            all_study_ids.append(study_id)
            
            if (i + 1) % 50 == 0:
                target_str = f"/{num_samples}" if num_samples else ""
                print(f"  Loaded {i + 1}{target_str} validation samples...")
        
        val_data = {
            'images': np.array(all_images),
            'captions': np.array(all_captions),
            'study_ids': np.array(all_study_ids),
            'tokenizer': self.tokenizer
        }
        
        print(f"Loaded {len(val_data['images'])} validation samples as numpy arrays")
        return val_data

    def get_test_data_for_evaluation(self, num_samples=1500):
        print(f"Loading test data for evaluation (numpy arrays, {num_samples if num_samples else 'ALL'} samples)")
        
        test_dataset = self.get_test_data(num_samples=num_samples)
        
        all_images = []
        all_captions = []
        all_study_ids = []
        
        for i, sample in enumerate(test_dataset):
            if num_samples is not None and i >= num_samples:
                break
                
            all_images.append(sample['images'].numpy())
            all_captions.append(sample['captions'].numpy())
            
            study_id = sample['study_ids']
            if isinstance(study_id, bytes):
                study_id = study_id.decode('utf-8')
            else:
                study_id = str(study_id)
            all_study_ids.append(study_id)
            
            if (i + 1) % 25 == 0:
                target_str = f"/{num_samples}" if num_samples else ""
                print(f"  Loaded {i + 1}{target_str} test samples...")
        
        test_data = {
            'images': np.array(all_images),
            'captions': np.array(all_captions),
            'study_ids': np.array(all_study_ids),
            'tokenizer': self.tokenizer
        }
        
        print(f"Loaded {len(test_data['images'])} test samples as numpy arrays")
        return test_data

if __name__ == "__main__":
    # Test with default indiana_shards
    data_loader = IndianaDataLoader(batch_size=16, shard_size=50, shard_subfolder="indiana_shards")
    
    data_loader.load_data(max_samples=300)
    
    dataset = data_loader.get_data()
    
    val_data = data_loader.get_validation_data(num_samples=50)
    
    print(f"Dataset created successfully")
    print(f"Validation data shape: {len(val_data)} samples")
    
    # Test switching to mimic_shards
    print("\n" + "="*50)
    print("Testing MIMIC shards...")
    mimic_loader = IndianaDataLoader(batch_size=16, shard_size=50, shard_subfolder="mimic_shards")
    print("MIMIC loader created successfully!") 