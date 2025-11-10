# Import required libraries
import pandas as pd
import dask.dataframe as dd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
import time
import gc
import psutil
import os

# Hardware Configuration
HARDWARE_CONFIG = {
    'device': 'cpu',  
    'num_threads': 16,  # CPU threads
    'memory_limit_gb': 20,  # Memory limit in GB
    'num_workers': 1,  # 1 worker for data loading
    'deterministic': True  # Ensure deterministic operations
}

RANDOM_SEEDS = [1, 2, 3, 4, 5]
DATASET = './CICIoT2023.parquet'

def configure_hardware():
    print("Configuring hardware...")
    
    # Set CPU threads
    torch.set_num_threads(HARDWARE_CONFIG['num_threads'])
    torch.set_num_interop_threads(HARDWARE_CONFIG['num_threads'])
    
    # Enable deterministic operations
    if HARDWARE_CONFIG['deterministic']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set device
    device = torch.device(HARDWARE_CONFIG['device'])
    
    # Memory monitoring setup
    process = psutil.Process(os.getpid())
    memory_limit_bytes = HARDWARE_CONFIG['memory_limit_gb'] * 1024 * 1024 * 1024
    
    print(f"Hardware Configuration:")
    print(f"Device: {device}")
    print(f"CPU Threads: {HARDWARE_CONFIG['num_threads']}")
    print(f"RAM Memory Limit: {HARDWARE_CONFIG['memory_limit_gb']} GB")
    print(f"Deterministic: {HARDWARE_CONFIG['deterministic']}")
    
    return device, memory_limit_bytes

def check_memory_usage(memory_limit_bytes):
    process = psutil.Process(os.getpid())
    current_memory = process.memory_info().rss
    
    if current_memory > memory_limit_bytes:
        print(f"WARNING: Memory usage ({current_memory / (1024**3):.2f} GB) exceeds limit ({memory_limit_bytes / (1024**3):.2f} GB)")
        gc.collect()
    
    return current_memory / (1024 * 1024)  # Return in MB

class LSTMAutoencoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, num_layers=1, sequence_length=10):
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):

        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
    
    def encode(self, x):

        # LSTM encoder
        encoded, (hidden_state, cell_state) = self.encoder_lstm(x)
        
        # Return the last hidden state as the encoded representation
        return encoded[:, -1, :], (hidden_state, cell_state)
    
    def decode(self, encoded, target_length, hidden_state=None):

        batch_size = encoded.size(0)
        
        # Initialize decoder input with encoded representation
        decoder_input = encoded.unsqueeze(1)  
        
        if hidden_state is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, 
                            device=encoded.device, dtype=encoded.dtype)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim,
                            device=encoded.device, dtype=encoded.dtype)
            hidden_state = (h_0, c_0)
        
        # Decode sequence step by step
        outputs = []
        current_hidden = hidden_state
        
        for t in range(target_length):

            lstm_out, current_hidden = self.decoder_lstm(decoder_input, current_hidden)
            
            output = self.output_projection(lstm_out)
            outputs.append(output)
            
            decoder_input = lstm_out
        
        # Concatenate all outputs
        decoded = torch.cat(outputs, dim=1)  
        
        return decoded
    
    def forward(self, x):

        # Encode
        encoded, hidden_state = self.encode(x)
        
        # Decode
        reconstructed = self.decode(encoded, x.size(1), hidden_state)
        
        return reconstructed, encoded

def create_sequences(data, sequence_length, overlapping=True):

    n_samples, n_features = data.shape
    
    if n_samples < sequence_length:

        n_repeats = (sequence_length // n_samples) + 1
        data = np.tile(data, (n_repeats, 1))
        n_samples = data.shape[0]
    
    sequences = []
    
    if overlapping:
        # Create overlapping sequences (for training)
        for i in range(n_samples - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
    else:
        # Create non-overlapping sequences (for evaluation)
        for i in range(0, n_samples - sequence_length + 1, sequence_length):
            sequences.append(data[i:i + sequence_length])
    
    # Convert to numpy array and then to tensor
    sequences = np.array(sequences)
    return torch.FloatTensor(sequences)

def get_model_size_kb(model):
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size = (param_size + buffer_size) / 1024  # Convert to KB
    return model_size

def measure_model_loading_time(model, device):
    # Create identical model architecture
    model_copy = LSTMAutoencoder(
        input_dim=model.input_dim,
        hidden_dim=model.hidden_dim,
        num_layers=model.num_layers,
        sequence_length=model.sequence_length
    )
    
    # Save model state
    state_dict = model.state_dict()
    
    # Clear cache and collect garbage
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    loading_times = []
    num_runs = 5 
    
    for _ in range(num_runs):
        # Create fresh model instance
        test_model = LSTMAutoencoder(
            input_dim=model.input_dim,
            hidden_dim=model.hidden_dim,
            num_layers=model.num_layers,
            sequence_length=model.sequence_length
        )
        
        # Measure state dict loading
        start_time = time.perf_counter()
        test_model.load_state_dict(state_dict)
        test_model.to(device)
        loading_time = time.perf_counter() - start_time
        
        loading_times.append(loading_time)
        
        # Cleanup
        del test_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Return average loading time (ms)
    avg_loading_time = np.mean(loading_times) * 1000
    std_loading_time = np.std(loading_times) * 1000
    
    del model_copy
    return avg_loading_time, std_loading_time

def extract_benign_data(memory_limit_bytes):
    print("Extracting Benign Data...")
    
    # Configure dask for controlled resource usage
    df_dask = dd.read_parquet(DATASET, engine='pyarrow')
    partitions = df_dask.to_delayed()
    benign_chunks = []
    
    print(f"Total partitions to process: {len(partitions)}")
    
    for i, partition in enumerate(partitions):
        if i % 10 == 0:
            print(f"Partition {i+1}/{len(partitions)} - Memory: {check_memory_usage(memory_limit_bytes):.1f} MB")
        
        chunk_df = partition.compute()
        chunk_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk_df.dropna(inplace=True)
        
        benign_mask = chunk_df['label'] == 'Benign_Final'
        if benign_mask.any():
            benign_chunk = chunk_df[benign_mask].copy()
            benign_chunks.append(benign_chunk)
        
        del chunk_df
        gc.collect()
    
    benign_df = pd.concat(benign_chunks, ignore_index=True)
    print(f"Collected {len(benign_df):,} total benign samples")
    return benign_df

def train_autoencoder(X_train, X_val, seed, device, memory_limit_bytes):
    print(f"Training LSTM Autoencoder (seed {seed})...")
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Track memory before training
    memory_before = check_memory_usage(memory_limit_bytes)
    
    # LSTM-specific parameters
    sequence_length = 10
    hidden_dim = 64
    num_layers = 1
    
    # Create sequences from the data (overlapping for training)
    X_train_seq = create_sequences(X_train.numpy(), sequence_length, overlapping=True)
    X_val_seq = create_sequences(X_val.numpy(), sequence_length, overlapping=True)
    
    # Initialize model
    model = LSTMAutoencoder(
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        sequence_length=sequence_length
    )
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()  # Reconstruction loss
    
    # Batch size 
    batch_size = 256
    best_val_loss = float('inf')
    patience = 0
    
    start_time = time.perf_counter()
    peak_memory = memory_before
    
    for epoch in range(50):
        epoch_start = time.perf_counter()
        
        # Training phase
        model.train()
        train_loss = 0
        batch_count = 0
        
        # Shuffle training sequences
        indices = torch.randperm(len(X_train_seq))
        X_train_shuffled = X_train_seq[indices]
        
        for i in range(0, len(X_train_shuffled), batch_size):
            batch = X_train_shuffled[i:i+batch_size].to(device)
            optimizer.zero_grad()
            
            reconstructed, encoded = model(batch)
            
            # Loss is reconstruction error
            loss = criterion(reconstructed, batch)
            
            loss.backward()
            
            # Gradient clipping for LSTM stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            # Track peak memory
            current_memory = check_memory_usage(memory_limit_bytes)
            peak_memory = max(peak_memory, current_memory)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val_seq), batch_size):
                batch = X_val_seq[i:i+batch_size].to(device)
                
                reconstructed, encoded = model(batch)
                loss = criterion(reconstructed, batch)
                
                val_loss += loss.item()
                val_batch_count += 1
        
        train_loss /= batch_count
        val_loss /= val_batch_count
        
        epoch_time = time.perf_counter() - epoch_start
        
        print(f"Epoch {epoch+1:2d}/50 | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {epoch_time:.1f}s")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            
        if patience >= 10:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.perf_counter() - start_time
    
    # Measure model loading time
    avg_loading_time, std_loading_time = measure_model_loading_time(model, device)
    model_size_kb = get_model_size_kb(model)
    
    # Compute threshold on validation sequences
    model.eval()
    with torch.no_grad():
        val_reconstructed, _ = model(X_val_seq.to(device))
        val_errors = torch.mean((X_val_seq.to(device) - val_reconstructed) ** 2, dim=(1, 2))
        threshold = np.percentile(val_errors.cpu().numpy(), 99)
    
    print(f"Training completed in {training_time:.1f}s")
    print(f"Model loading time: {avg_loading_time:.4f} ± {std_loading_time:.4f}ms")
    print(f"Model size: {model_size_kb:.1f} KB")
    print(f"Peak memory usage: {peak_memory:.2f} MB")
    print(f"Threshold (99th percentile of validation): {threshold:.6f}")
    
    return model, threshold, {
        'training_time': training_time,
        'model_loading_time_mean': avg_loading_time,
        'model_loading_time_std': std_loading_time,
        'model_size_kb': model_size_kb,
        'peak_memory_mb': peak_memory
    }

def evaluate_model(model, scaler, device, memory_limit_bytes):
    print("Evaluating model on full dataset...")
    
    # Evaluate on full CICIoT2023 dataset: 22% held-out test benign + 78% train/val benign + entire attack corpus (97.6% attack ratio)
    df_dask = dd.read_parquet(DATASET, engine='pyarrow')
    partitions = df_dask.to_delayed()
    
    all_errors = []
    all_labels = []
    model.eval()
    
    total_samples_processed = 0
    benign_count = 0
    
    evaluation_start = time.perf_counter()
    
    for i, partition in enumerate(partitions):
        if i % 15 == 0:
            print(f"Evaluating partition {i+1}/{len(partitions)} - Memory: {check_memory_usage(memory_limit_bytes):.1f} MB")
        
        chunk_df = partition.compute()
        chunk_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk_df.dropna(inplace=True)
        
        if len(chunk_df) == 0:
            continue
        
        chunk_benign = (chunk_df['label'] == 'Benign_Final').sum()
        benign_count += chunk_benign
        
        X_chunk = chunk_df.drop('label', axis=1)
        y_binary = (chunk_df['label'] != 'Benign_Final').astype(int)
        
        X_norm = scaler.transform(X_chunk)
        
        # Create non-overlapping sequences for evaluation
        if len(X_norm) >= model.sequence_length:
            X_sequences = create_sequences(X_norm, model.sequence_length, overlapping=False)
            
            batch_size = 256
            sequence_errors = []
            
            inference_start = time.perf_counter()
            with torch.no_grad():
                for j in range(0, len(X_sequences), batch_size):
                    batch_seq = X_sequences[j:j+batch_size].to(device)
                    reconstructed, _ = model(batch_seq)
                    
                    # Calculate reconstruction error for each sequence
                    errors = torch.mean((batch_seq - reconstructed) ** 2, dim=(1, 2))
                    sequence_errors.extend(errors.cpu().numpy())
            
            inference_time = time.perf_counter() - inference_start
            
            # Map sequence errors to samples (middle sample of each sequence)
            for seq_idx, error in enumerate(sequence_errors):
                sample_idx = seq_idx * model.sequence_length + (model.sequence_length // 2)
                if sample_idx < len(y_binary):
                    all_errors.append(error)
                    all_labels.append(y_binary.iloc[sample_idx])
            
            total_samples_processed += len(sequence_errors)
        
        del chunk_df, X_chunk
        gc.collect()
    
    total_evaluation_time = time.perf_counter() - evaluation_start
    inference_time_per_sample = (total_evaluation_time / total_samples_processed) * 1000 if total_samples_processed > 0 else 0
    
    print(f"Evaluation completed in {total_evaluation_time:.1f}s")
    print(f"Total samples evaluated: {total_samples_processed:,}")
    print(f"Benign samples: {benign_count:,}")
    print(f"Inference time per sample: {inference_time_per_sample:.3f} ms")
    
    return np.array(all_errors), np.array(all_labels), {
        'total_evaluation_time': total_evaluation_time,
        'inference_time_per_sample_ms': inference_time_per_sample,
        'total_samples': total_samples_processed
    }

def calculate_metrics(errors, labels, threshold):
    print("Calculating metrics...")
    
    roc_auc = roc_auc_score(labels, errors)
    auprc = average_precision_score(labels, errors)
    
    predictions = (errors > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    print(f"Threshold (from validation set): {threshold:.6f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    return {
        'roc_auc': roc_auc,
        'auprc': auprc,
        'avg_precision': auprc, 
        'f1_score': f1,
        'threshold': threshold,
        'precision': precision,
        'recall': recall
    }

def run_seed_experiment(benign_df, seed, device, memory_limit_bytes):
    print(f"\nSEED: {seed}")
    print(f"Using device: {device}")
    
    X_benign = benign_df.drop('label', axis=1)
    
    X_trainval, X_test = train_test_split(X_benign, test_size=0.22, random_state=seed)
    X_train, X_val = train_test_split(X_trainval, test_size=0.20, random_state=seed)
    
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    scaler = StandardScaler()
    print("Fitting scaler and normalizing data...")
    X_train_norm = torch.FloatTensor(scaler.fit_transform(X_train))
    X_val_norm = torch.FloatTensor(scaler.transform(X_val))
    
    model, threshold, training_metrics = train_autoencoder(X_train_norm, X_val_norm, seed, device, memory_limit_bytes)
    errors, labels, evaluation_metrics = evaluate_model(model, scaler, device, memory_limit_bytes)
    performance_metrics = calculate_metrics(errors, labels, threshold)
    
    result = {
        'seed': seed,
        'samples_evaluated': len(errors),
        **performance_metrics,
        **training_metrics,
        **evaluation_metrics
    }
    
    print(f"FINAL METRICS FOR SEED {seed}:")
    print(f"ROC-AUC: {performance_metrics['roc_auc']:.4f}")
    print(f"AUPRC: {performance_metrics['auprc']:.4f}")
    print(f"F1 Score: {performance_metrics['f1_score']:.4f}")
    print(f"Training Time: {training_metrics['training_time']:.1f}s")
    print(f"Model Loading Time: {training_metrics['model_loading_time_mean']:.4f} ± {training_metrics['model_loading_time_std']:.4f}ms")
    print(f"Inference Time: {evaluation_metrics['inference_time_per_sample_ms']:.3f} ms/sample")
    print(f"Peak Memory: {training_metrics['peak_memory_mb']:.2f} MB")
    
    del model, X_train_norm, X_val_norm, errors, labels
    gc.collect()
    
    return result

def calculate_uncertainty(results):
    roc_aucs = [r['roc_auc'] for r in results]
    auprcs = [r['auprc'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    training_times = [r['training_time'] for r in results]
    model_loading_times_mean = [r['model_loading_time_mean'] for r in results]
    model_loading_times_std = [r['model_loading_time_std'] for r in results]
    inference_times = [r['inference_time_per_sample_ms'] for r in results]
    peak_memories = [r['peak_memory_mb'] for r in results]
    model_size_kb = results[0]['model_size_kb']
    
    print("INDIVIDUAL SEED RESULTS:")
    for result in results:
        print(f"Seed {result['seed']}: ROC-AUC={result['roc_auc']:.4f}, AUPRC={result['auprc']:.4f}, F1={result['f1_score']:.4f}")
    
    return {
        'model_name': 'LSTM Autoencoder',
        'hardware_config': HARDWARE_CONFIG,
        'roc_auc_mean': np.mean(roc_aucs),
        'roc_auc_std': np.std(roc_aucs),
        'auprc_mean': np.mean(auprcs),
        'auprc_std': np.std(auprcs),
        'f1_score_mean': np.mean(f1_scores),
        'f1_score_std': np.std(f1_scores),
        'precision_mean': np.mean(precisions),
        'precision_std': np.std(precisions),
        'recall_mean': np.mean(recalls),
        'recall_std': np.std(recalls),
        'training_time_mean': np.mean(training_times),
        'training_time_std': np.std(training_times),
        'model_loading_time_mean': np.mean(model_loading_times_mean),
        'model_loading_time_std': np.std(model_loading_times_mean),
        'model_loading_time_variability': np.mean(model_loading_times_std),
        'inference_time_mean': np.mean(inference_times),
        'inference_time_std': np.std(inference_times),
        'peak_memory_mean': np.mean(peak_memories),
        'peak_memory_std': np.std(peak_memories),
        'model_size_kb': model_size_kb
    }

def run():
    # Configure hardware first
    device, memory_limit_bytes = configure_hardware()
    
    total_start = time.perf_counter()
    
    # Extract benign data
    benign_df = extract_benign_data(memory_limit_bytes)
    
    # Run experiments with controlled hardware
    results = []
    for seed in RANDOM_SEEDS:
        result = run_seed_experiment(benign_df, seed, device, memory_limit_bytes)
        results.append(result)
    
    # Calculate final results
    final_results = calculate_uncertainty(results)
    
    total_time = time.perf_counter() - total_start
    
    print("\nTRAINING FINISHED:")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    print("\nFINAL BENCHMARK RESULTS:")
    print("Performance Metrics:")
    print(f"ROC-AUC: {final_results['roc_auc_mean']:.4f} ± {final_results['roc_auc_std']:.4f}")
    print(f"AUPRC: {final_results['auprc_mean']:.4f} ± {final_results['auprc_std']:.4f}")
    print(f"F1 Score: {final_results['f1_score_mean']:.4f} ± {final_results['f1_score_std']:.4f}")
    print(f"Precision: {final_results['precision_mean']:.4f} ± {final_results['precision_std']:.4f}")
    print(f"Recall: {final_results['recall_mean']:.4f} ± {final_results['recall_std']:.4f}")
    
    print("\nEfficiency Metrics:")
    print(f"Training Time: {final_results['training_time_mean']:.1f} ± {final_results['training_time_std']:.1f}s")
    print(f"Model Loading Time: {final_results['model_loading_time_mean']:.4f} ± {final_results['model_loading_time_std']:.4f}ms")
    print(f"Inference Time: {final_results['inference_time_mean']:.3f} ± {final_results['inference_time_std']:.3f} ms/sample")
    print(f"Peak Memory: {final_results['peak_memory_mean']:.2f} ± {final_results['peak_memory_std']:.2f} MB")
    print(f"Model Size: {final_results['model_size_kb']:.1f} KB")
    
    return final_results

if __name__ == "__main__":
    final_results = run()