# Import required libraries
import pandas as pd
import dask.dataframe as dd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.covariance import EllipticEnvelope
import time
import gc
import psutil
import os
import pickle

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
    
    memory_limit_bytes = HARDWARE_CONFIG['memory_limit_gb'] * 1024 * 1024 * 1024
    
    print(f"Hardware Configuration:")
    print(f"Device: {HARDWARE_CONFIG['device']}")
    print(f"CPU Threads: {HARDWARE_CONFIG['num_threads']}")
    print(f"RAM Memory Limit: {HARDWARE_CONFIG['memory_limit_gb']} GB")
    print(f"Deterministic: {HARDWARE_CONFIG['deterministic']}")
    
    return memory_limit_bytes

def check_memory_usage(memory_limit_bytes):
    process = psutil.Process(os.getpid())
    current_memory = process.memory_info().rss
    
    if current_memory > memory_limit_bytes:
        print(f"WARNING: Memory usage ({current_memory / (1024**3):.2f} GB) exceeds limit ({memory_limit_bytes / (1024**3):.2f} GB)")
        gc.collect()
    
    return current_memory / (1024 * 1024)

def get_model_size_kb(model):

    model_bytes = pickle.dumps(model)
    model_size_kb = len(model_bytes) / 1024
    return model_size_kb

def measure_model_loading_time(model, num_runs=5):

    model_bytes = pickle.dumps(model)
    
    loading_times = []
    
    for _ in range(num_runs):
        gc.collect()
        
        start_time = time.perf_counter()
        loaded_model = pickle.loads(model_bytes)
        loading_time = time.perf_counter() - start_time
        
        loading_times.append(loading_time)
        del loaded_model
    
    avg_loading_time = np.mean(loading_times) * 1000  # Convert to ms
    std_loading_time = np.std(loading_times) * 1000
    
    return avg_loading_time, std_loading_time

def extract_benign_data(memory_limit_bytes):
    print("Extracting Benign Data...")
    
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

def train_elliptic_envelope(X_train, X_val, seed, memory_limit_bytes):

    print(f"Training Elliptic Envelope (seed {seed})...")
    
    memory_before = check_memory_usage(memory_limit_bytes)
    
    # Elliptic Envelope parameters
    model = EllipticEnvelope(
        contamination=0.01,  # Expected proportion of outliers
        support_fraction=None, 
        random_state=seed
    )
    
    start_time = time.perf_counter()
    peak_memory = memory_before
    
    # Train on benign data only
    model.fit(X_train)
    
    training_time = time.perf_counter() - start_time
    
    current_memory = check_memory_usage(memory_limit_bytes)
    peak_memory = max(peak_memory, current_memory)
    
    avg_loading_time, std_loading_time = measure_model_loading_time(model)
    model_size_kb = get_model_size_kb(model)
    
    val_mahalanobis_dist = model.score_samples(X_val)
    val_scores = -val_mahalanobis_dist
    threshold = np.percentile(val_scores, 99)
    
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

def evaluate_model(model, scaler, memory_limit_bytes):
    print("Evaluating model on full dataset...")
    
    # Evaluate on full CICIoT2023 dataset: 22% held-out test benign + 78% train/val benign + entire attack corpus (97.6% attack ratio)
    df_dask = dd.read_parquet(DATASET, engine='pyarrow')
    partitions = df_dask.to_delayed()
    
    all_errors = []
    all_labels = []
    
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
        
        inference_start = time.perf_counter()
        
        # Anomaly score
        mahalanobis_dist = model.score_samples(X_norm)
        scores = -mahalanobis_dist
        all_errors.extend(scores)
        all_labels.extend(y_binary.values)
        
        inference_time = time.perf_counter() - inference_start
        total_samples_processed += len(chunk_df)
        
        del chunk_df, X_chunk
        gc.collect()
    
    total_evaluation_time = time.perf_counter() - evaluation_start
    inference_time_per_sample = (total_evaluation_time / total_samples_processed) * 1000
    
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

def run_seed_experiment(benign_df, seed, memory_limit_bytes):
    print(f"\nSEED: {seed}")
    
    X_benign = benign_df.drop('label', axis=1)
    
    X_trainval, X_test = train_test_split(X_benign, test_size=0.22, random_state=seed)
    X_train, X_val = train_test_split(X_trainval, test_size=0.20, random_state=seed)
    
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    scaler = StandardScaler()
    print("Fitting scaler and normalizing data...")
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    
    model, threshold, training_metrics = train_elliptic_envelope(X_train_norm, X_val_norm, seed, memory_limit_bytes)
    errors, labels, evaluation_metrics = evaluate_model(model, scaler, memory_limit_bytes)
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
        'model_name': 'Elliptic Envelope',
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
    memory_limit_bytes = configure_hardware()
    
    total_start = time.perf_counter()
    
    benign_df = extract_benign_data(memory_limit_bytes)
    
    results = []
    for seed in RANDOM_SEEDS:
        result = run_seed_experiment(benign_df, seed, memory_limit_bytes)
        results.append(result)
    
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