import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

try:
    from analysis_all_visualization_v1 import visualize_retrieval_examples
    visualization_available = True
except ImportError:
    visualization_available = False
    print("Visualization module not available. Install matplotlib for visualization features.")

def create_rank_distribution_plots(i2t_ranks, t2i_ranks, results, output_dir="evaluation_plots"):
    """Create comprehensive rank distribution plots for evaluation results"""
    print("📊 Creating rank distribution plots...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create rank distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Rank Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Rank histograms
    axes[0, 0].hist(i2t_ranks, bins=50, alpha=0.7, label='Image→Text', color='blue', density=True)
    axes[0, 0].hist(t2i_ranks, bins=50, alpha=0.7, label='Text→Image', color='red', density=True)
    axes[0, 0].set_xlabel('Rank')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Rank Distribution Histogram')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Cumulative rank distribution
    max_rank = int(max(max(i2t_ranks), max(t2i_ranks)))
    rank_range = range(1, min(max_rank + 1, 101))
    
    # Convert ranks to integers for bincount
    i2t_ranks_int = i2t_ranks.astype(int)
    t2i_ranks_int = t2i_ranks.astype(int)
    
    # Ensure max_rank is an integer
    max_rank = int(max_rank)
    
    i2t_cumulative = np.cumsum(np.bincount(i2t_ranks_int, minlength=max_rank+1)[1:]) / len(i2t_ranks)
    t2i_cumulative = np.cumsum(np.bincount(t2i_ranks_int, minlength=max_rank+1)[1:]) / len(t2i_ranks)
    
    # Truncate to reasonable range
    plot_range = min(len(rank_range), len(i2t_cumulative), len(t2i_cumulative))
    rank_range = rank_range[:plot_range]
    i2t_cumulative = i2t_cumulative[:plot_range]
    t2i_cumulative = t2i_cumulative[:plot_range]
    
    axes[0, 1].plot(rank_range, i2t_cumulative, label='Image→Text', color='blue', linewidth=2)
    axes[0, 1].plot(rank_range, t2i_cumulative, label='Text→Image', color='red', linewidth=2)
    axes[0, 1].set_xlabel('Rank')
    axes[0, 1].set_ylabel('Cumulative Probability')
    axes[0, 1].set_title('Cumulative Rank Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(1, min(50, max_rank))  # Focus on first 50 ranks
    
    # Plot 3: Log-scale rank distribution
    axes[1, 0].hist(i2t_ranks, bins=50, alpha=0.7, label='Image→Text', color='blue', density=True)
    axes[1, 0].hist(t2i_ranks, bins=50, alpha=0.7, label='Text→Image', color='red', density=True)
    axes[1, 0].set_xlabel('Rank')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Rank Distribution (Log Scale)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Performance comparison
    metrics = ['R@1', 'R@5', 'R@10', 'MRR']
    i2t_values = [results.get('i2t_recall@1', 0), results.get('i2t_recall@5', 0), 
                  results.get('i2t_recall@10', 0), results.get('i2t_mrr', 0)]
    t2i_values = [results.get('t2i_recall@1', 0), results.get('t2i_recall@5', 0), 
                  results.get('t2i_recall@10', 0), results.get('t2i_mrr', 0)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, i2t_values, width, label='Image→Text', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, t2i_values, width, label='Text→Image', color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add text annotations with statistics
    stats_text = f"""
    Image→Text:
    Mean Rank: {results.get('i2t_mean_rank', 0):.2f}
    Median Rank: {results.get('i2t_median_rank', 0):.2f}
    
    Text→Image:
    Mean Rank: {results.get('t2i_mean_rank', 0):.2f}
    Median Rank: {results.get('t2i_median_rank', 0):.2f}
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'rank_distribution_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Rank distribution plot saved to: {plot_path}")
    
    return plot_path

def evaluate_cross_modal_retrieval(model, test_data, k_values=[1, 5, 10], visualize=False, num_vis_examples=3, output_dir=None, create_rank_plots=False):
    """Evaluate cross-modal retrieval performance - PyTorch version"""
    print("Evaluating cross-modal retrieval performance...")
    
    # Move model to eval mode
    model.eval()
    
    with torch.no_grad():
        # Convert data to PyTorch tensors if needed
        if isinstance(test_data['images'], np.ndarray):
            images = torch.FloatTensor(test_data['images'])
            if len(images.shape) == 4 and images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
        else:
            images = test_data['images']
            
        if isinstance(test_data['captions'], np.ndarray):
            captions = torch.LongTensor(test_data['captions'])
        else:
            captions = test_data['captions']
        
        # Move to device if model is on GPU
        device = next(model.parameters()).device
        images = images.to(device)
        captions = captions.to(device)
        
        image_emb, text_emb = model((images, captions), training=False)
    
    results = {}
    i2t_ranks = []
    t2i_ranks = []
    
    print("Computing similarity matrices...")
    
    i2t_sim = torch.matmul(image_emb, text_emb.transpose(0, 1))
    t2i_sim = torch.matmul(text_emb, image_emb.transpose(0, 1))
    
    print("Calculating metrics...")
    
    for direction, sim_matrix, direction_name in [
        ("i2t", i2t_sim, "Image-to-Text"), 
        ("t2i", t2i_sim, "Text-to-Image")
    ]:
        print(f"Evaluating {direction_name} Retrieval:")
        
        correct_indices = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        
        # Convert to numpy for rank calculations
        sim_np = sim_matrix.cpu().numpy()
        
        # Calculate ranks
        ranks = []
        for i in range(len(sim_np)):
            # Get similarities for this query
            similarities = sim_np[i]
            
            # Get the score for the correct match
            correct_score = similarities[i]
            
            # Calculate rank (1-indexed)
            rank = 1 + np.sum(similarities > correct_score)
            
            # Handle ties by averaging ranks
            tie_count = np.sum(np.isclose(similarities, correct_score)) - 1
            if tie_count > 0:
                # Average rank for tied items
                rank = rank + tie_count / 2
                
            ranks.append(rank)
        
        ranks = np.array(ranks)
        
        # Store ranks for plotting
        if direction == "i2t":
            i2t_ranks = ranks
        else:
            t2i_ranks = ranks
        
        # Calculate MRR (Mean Reciprocal Rank)
        mrr = np.mean(1.0 / ranks)
        results[f"{direction}_mrr"] = float(mrr)
        
        # Calculate mean and median rank
        mean_rank = np.mean(ranks)
        results[f"{direction}_mean_rank"] = float(mean_rank)
        
        median_rank = np.median(ranks)
        results[f"{direction}_median_rank"] = float(median_rank)
        
        # Calculate rank percentages
        rank_thresholds = [1, 5, 10, 50, 100]
        for threshold in rank_thresholds:
            pct = 100 * np.mean(ranks <= threshold)
            results[f"{direction}_rank<={threshold}"] = float(pct)
            
        # Calculate Recall@K and Precision@K
        for k in k_values:
            _, top_k_indices = torch.topk(sim_matrix, k=k, dim=1)
            
            correct_in_topk = torch.any(
                top_k_indices == correct_indices.unsqueeze(1),
                dim=1
            )
            
            recall = torch.mean(correct_in_topk.float())
            results[f"{direction}_recall@{k}"] = float(recall.cpu())
            
            # Calculate Precision@K
            # For each query, count how many of the top-k results are correct
            precision_scores = []
            for i in range(sim_matrix.size(0)):
                correct_count = torch.sum(
                    top_k_indices[i] == correct_indices[i]
                ).float()
                precision = correct_count / k
                precision_scores.append(precision)
            
            precision = torch.mean(torch.stack(precision_scores))
            results[f"{direction}_precision@{k}"] = float(precision.cpu())
        
        # Print results for this direction
        print(f"  MRR: {float(mrr):.4f}")
        print(f"  Mean Rank: {float(mean_rank):.2f}")
        print(f"  Median Rank: {float(median_rank):.2f}")
        
        for k in k_values:
            print(f"  Recall@{k}: {float(results[f'{direction}_recall@{k}']):.4f}")
            print(f"  Precision@{k}: {float(results[f'{direction}_precision@{k}']):.4f}")
        
        print("  Rank percentages:")
        for threshold in rank_thresholds:
            print(f"    <= {threshold}: {float(results[f'{direction}_rank<={threshold}']):.2f}%")
    
    # Calculate overall averages
    print("\nOverall Cross-Modal Retrieval Performance:")
    
    avg_mrr = (results[f"i2t_mrr"] + results[f"t2i_mrr"]) / 2
    results[f"avg_mrr"] = avg_mrr
    print(f"  Average MRR: {avg_mrr:.4f}")
    
    avg_mean_rank = (results[f"i2t_mean_rank"] + results[f"t2i_mean_rank"]) / 2
    results[f"avg_mean_rank"] = avg_mean_rank
    print(f"  Average Mean Rank: {avg_mean_rank:.2f}")
    
    avg_median_rank = (results[f"i2t_median_rank"] + results[f"t2i_median_rank"]) / 2
    results[f"avg_median_rank"] = avg_median_rank
    print(f"  Average Median Rank: {avg_median_rank:.2f}")
    
    for k in k_values:
        avg_recall = (results[f"i2t_recall@{k}"] + results[f"t2i_recall@{k}"]) / 2
        results[f"avg_recall@{k}"] = avg_recall
        print(f"  Average Recall@{k}: {avg_recall:.4f}")
    
    # Create summary string
    try:
        summary = (f"MRR: {avg_mrr:.4f}, "
                f"Mean Rank: {avg_mean_rank:.1f}, "
                f"Median Rank: {avg_median_rank:.1f}, " + 
                ", ".join([f"R@{k}: {results[f'avg_recall@{k}']:.4f}" for k in k_values]))
    except Exception as e:
        summary = f"Summary generation error: {str(e)}"
    
    results["summary"] = summary
    
    # Create rank distribution plots if requested
    if create_rank_plots and len(i2t_ranks) > 0 and len(t2i_ranks) > 0:
        plot_path = create_rank_distribution_plots(i2t_ranks, t2i_ranks, results, output_dir)
        results["rank_plot_path"] = plot_path
    
    # Generate visualizations if requested
    if visualize and visualization_available and num_vis_examples > 0:
        # Ensure tokenizer is available
        if hasattr(test_data, 'tokenizer') or 'tokenizer' in test_data:
            pass
        elif hasattr(test_data, 'indiana_loader') and hasattr(test_data.indiana_loader, 'tokenizer'):
            test_data['tokenizer'] = test_data.indiana_loader.tokenizer
        
        visualize_retrieval_examples(model, test_data, num_examples=num_vis_examples, k=3, output_dir=output_dir)
    
    return results

def evaluate_cross_modal_retrieval_streaming(model, test_dataset, k_values=[1, 5, 10], batch_size=64, visualize=False, num_vis_examples=3, output_dir=None, create_rank_plots=False):
    """Evaluate cross-modal retrieval performance on streaming dataset - PyTorch version"""
    print(f"Processing in batches of {batch_size}...")
    
    # Move model to eval mode
    model.eval()
    device = next(model.parameters()).device
    
    all_image_embeddings = []
    all_text_embeddings = []
    all_images_for_viz = []
    all_captions_for_viz = []
    all_study_ids = []
    
    sample_count = 0
    batch_count = 0
    
    # Handle PyTorch DataLoader vs TensorFlow Dataset
    if hasattr(test_dataset, 'batch'):
        # TensorFlow Dataset
        batched_dataset = test_dataset.batch(batch_size)
        iterator = iter(batched_dataset)
    else:
        # PyTorch DataLoader
        from torch.utils.data import DataLoader
        if not isinstance(test_dataset, DataLoader):
            batched_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            batched_dataset = test_dataset
        iterator = iter(batched_dataset)
    
    with torch.no_grad():
        for batch in iterator:
            if hasattr(batch, 'keys'):
                # Dictionary format
                batch_images = batch['images']
                batch_captions = batch['captions']
                batch_study_ids = batch['study_ids']
            else:
                # Tuple format
                batch_images, batch_captions, batch_study_ids = batch
            
            # Convert to PyTorch tensors if needed
            if not isinstance(batch_images, torch.Tensor):
                batch_images = torch.FloatTensor(batch_images)
            if not isinstance(batch_captions, torch.Tensor):
                batch_captions = torch.LongTensor(batch_captions)
            
            # Handle image tensor format conversion
            if len(batch_images.shape) == 4 and batch_images.shape[-1] == 3:
                batch_images = batch_images.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
            
            # Move to device
            batch_images = batch_images.to(device)
            batch_captions = batch_captions.to(device)
            
            # Get embeddings
            batch_image_emb, batch_text_emb = model((batch_images, batch_captions), training=False)
            
            # Store embeddings
            all_image_embeddings.append(batch_image_emb.cpu().numpy())
            all_text_embeddings.append(batch_text_emb.cpu().numpy())
            
            # Store data for visualization
            if len(all_images_for_viz) < num_vis_examples * 2:
                all_images_for_viz.extend(batch_images.cpu().numpy())
                all_captions_for_viz.extend(batch_captions.cpu().numpy())
            
            # Handle study IDs
            if isinstance(batch_study_ids, torch.Tensor):
                study_ids_batch = [str(sid.item()) for sid in batch_study_ids]
            elif hasattr(batch_study_ids, '__iter__') and not isinstance(batch_study_ids, str):
                # Handle list/array of study IDs
                study_ids_batch = []
                for sid in batch_study_ids:
                    if isinstance(sid, str):
                        study_ids_batch.append(sid)
                    elif hasattr(sid, 'numpy'):
                        sid_val = sid.numpy()
                        if isinstance(sid_val, bytes):
                            study_ids_batch.append(sid_val.decode('utf-8'))
                        else:
                            study_ids_batch.append(str(sid_val))
                    else:
                        study_ids_batch.append(str(sid))
            else:
                study_ids_batch = [str(batch_study_ids)]
            all_study_ids.extend(study_ids_batch)
            
            sample_count += len(batch_images)
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f"  Processed {sample_count} samples in {batch_count} batches...")
    
    print(f"Computed embeddings for {sample_count} test samples")
    
    print("Concatenating embeddings and computing similarity matrices...")
    image_embeddings = np.concatenate(all_image_embeddings, axis=0)
    text_embeddings = np.concatenate(all_text_embeddings, axis=0)
    
    print(f"   Image embeddings shape: {image_embeddings.shape}")
    print(f"   Text embeddings shape: {text_embeddings.shape}")
    
    # Convert back to PyTorch tensors for similarity computation
    image_emb_torch = torch.FloatTensor(image_embeddings).to(device)
    text_emb_torch = torch.FloatTensor(text_embeddings).to(device)
    
    print("Computing similarity matrices...")
    i2t_sim = torch.matmul(image_emb_torch, text_emb_torch.transpose(0, 1))
    t2i_sim = torch.matmul(text_emb_torch, image_emb_torch.transpose(0, 1))
    
    results = {}
    i2t_ranks = []
    t2i_ranks = []
    
    print("Calculating metrics...")
    
    for direction, sim_matrix, direction_name in [
        ("i2t", i2t_sim, "Image-to-Text"), 
        ("t2i", t2i_sim, "Text-to-Image")
    ]:
        print(f"Evaluating {direction_name} Retrieval:")
        
        correct_indices = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        sim_np = sim_matrix.cpu().numpy()
        
        # Calculate ranks
        ranks = []
        for i in range(len(sim_np)):
            similarities = sim_np[i]
            correct_score = similarities[i]
            rank = 1 + np.sum(similarities > correct_score)
            
            # Handle ties
            tie_count = np.sum(np.isclose(similarities, correct_score)) - 1
            if tie_count > 0:
                rank = rank + tie_count / 2
                
            ranks.append(rank)
        
        ranks = np.array(ranks)
        
        # Store ranks for plotting
        if direction == "i2t":
            i2t_ranks = ranks
        else:
            t2i_ranks = ranks
        
        # Calculate metrics
        mrr = np.mean(1.0 / ranks)
        results[f"{direction}_mrr"] = float(mrr)
        
        mean_rank = np.mean(ranks)
        results[f"{direction}_mean_rank"] = float(mean_rank)
        
        median_rank = np.median(ranks)
        results[f"{direction}_median_rank"] = float(median_rank)
        
        # Rank percentages
        rank_thresholds = [1, 5, 10, 50, 100]
        for threshold in rank_thresholds:
            pct = 100 * np.mean(ranks <= threshold)
            results[f"{direction}_rank<={threshold}"] = float(pct)
            
        # Recall@K and Precision@K
        for k in k_values:
            _, top_k_indices = torch.topk(sim_matrix, k=k, dim=1)
            
            correct_in_topk = torch.any(
                top_k_indices == correct_indices.unsqueeze(1),
                dim=1
            )
            
            recall = torch.mean(correct_in_topk.float())
            results[f"{direction}_recall@{k}"] = float(recall.cpu())
            
            # Calculate Precision@K
            precision_scores = []
            for i in range(sim_matrix.size(0)):
                correct_count = torch.sum(
                    top_k_indices[i] == correct_indices[i]
                ).float()
                precision = correct_count / k
                precision_scores.append(precision)
            
            precision = torch.mean(torch.stack(precision_scores))
            results[f"{direction}_precision@{k}"] = float(precision.cpu())
        
        # Print results
        print(f"  MRR: {float(mrr):.4f}")
        print(f"  Mean Rank: {float(mean_rank):.2f}")
        print(f"  Median Rank: {float(median_rank):.2f}")
        
        for k in k_values:
            print(f"  Recall@{k}: {float(results[f'{direction}_recall@{k}']):.4f}")
            print(f"  Precision@{k}: {float(results[f'{direction}_precision@{k}']):.4f}")
        
        print("  Rank percentages:")
        for threshold in rank_thresholds:
            print(f"    <= {threshold}: {float(results[f'{direction}_rank<={threshold}']):.2f}%")
    
    # Calculate overall performance
    print("Overall Performance:")
    
    avg_mrr = (results[f"i2t_mrr"] + results[f"t2i_mrr"]) / 2
    results[f"avg_mrr"] = avg_mrr
    print(f"  Average MRR: {avg_mrr:.4f}")
    
    avg_mean_rank = (results[f"i2t_mean_rank"] + results[f"t2i_mean_rank"]) / 2
    results[f"avg_mean_rank"] = avg_mean_rank
    print(f"  Average Mean Rank: {avg_mean_rank:.2f}")
    
    avg_median_rank = (results[f"i2t_median_rank"] + results[f"t2i_median_rank"]) / 2
    results[f"avg_median_rank"] = avg_median_rank
    print(f"  Average Median Rank: {avg_median_rank:.2f}")
    
    for k in k_values:
        avg_recall = (results[f"i2t_recall@{k}"] + results[f"t2i_recall@{k}"]) / 2
        results[f"avg_recall@{k}"] = avg_recall
        print(f"  Average Recall@{k}: {avg_recall:.4f}")
        
        avg_precision = (results[f"i2t_precision@{k}"] + results[f"t2i_precision@{k}"]) / 2
        results[f"avg_precision@{k}"] = avg_precision
        print(f"  Average Precision@{k}: {avg_precision:.4f}")
    
    # Create summary
    try:
        summary = (f"MRR: {avg_mrr:.4f}, "
                f"Mean Rank: {avg_mean_rank:.1f}, "
                f"Median Rank: {avg_median_rank:.1f}, " + 
                ", ".join([f"R@{k}: {results[f'avg_recall@{k}']:.4f}" for k in k_values]))
    except Exception as e:
        summary = f"Summary generation error: {str(e)}"
    
    results["summary"] = summary
    
    # Create rank distribution plots if requested
    if create_rank_plots and len(i2t_ranks) > 0 and len(t2i_ranks) > 0:
        plot_path = create_rank_distribution_plots(i2t_ranks, t2i_ranks, results, output_dir)
        results["rank_plot_path"] = plot_path
    
    # Generate visualizations if requested
    if visualize and visualization_available and num_vis_examples > 0:
        print("\nGenerating retrieval visualizations...")
        
        viz_test_data = {
            'images': np.array(all_images_for_viz[:num_vis_examples * 2]),
            'captions': np.array(all_captions_for_viz[:num_vis_examples * 2]),
            'study_ids': np.array(all_study_ids[:num_vis_examples * 2])
        }
        
        visualize_retrieval_examples(model, viz_test_data, num_examples=num_vis_examples, k=3, output_dir=output_dir)
    
    # Print final summary
    print(f"   Total samples evaluated: {sample_count}")
    print(f"   Memory efficient: Processed in {batch_count} batches")
    
    return results 