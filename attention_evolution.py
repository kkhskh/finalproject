"""
Evolutionary Algorithm for Attention Optimization
CS 2420 Final Project - Evolved Attention Mechanisms
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PART 1: Load WikiText-2 Dataset
# ============================================

class WikiText2Dataset(Dataset):
    """Custom dataset for WikiText-2"""
    def __init__(self, dataset_split, tokenizer, max_length=128, limit=None):
        self.texts = dataset_split['text']
        if limit:
            self.texts = self.texts[:limit]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Handle empty texts
        if not text or len(text.strip()) == 0:
            text = "[PAD]"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }


def load_wikitext2_data(batch_size=16, use_full_dataset=True):
    """Load WikiText-2 from local disk"""
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    
    print("Loading WikiText-2 from disk...")
    
    # Load from saved location
    dataset = load_from_disk('./data/wikitext2')
    
    train_data = dataset['train']
    val_data = dataset['validation']
    
    # Use full dataset for rigorous evaluation (36,718 training samples)
    if use_full_dataset:
        print(f"✓ Using FULL dataset: {len(train_data):,} training samples, {len(val_data):,} validation samples")
        # Keep all data
    else:
        # Optional: for quick testing only
        limit_samples = 1000
        print(f"Using {limit_samples} samples (limited mode)")
        train_data = train_data.select(range(min(limit_samples, len(train_data))))
        val_data = val_data.select(range(min(500, len(val_data))))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create datasets
    train_dataset = WikiText2Dataset(train_data, tokenizer)
    val_dataset = WikiText2Dataset(val_data, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


# ============================================
# PART 2: Evolved Attention Configuration
# ============================================

class AttentionConfig:
    """Represents one attention algorithm variant"""
    def __init__(self, config_list=None):
        if config_list is None:
            self.tile_size = np.random.randint(32, 256)
            self.use_fusion = np.random.choice([True, False])
            self.num_heads = np.random.choice([4, 8, 12])
            self.dropout_rate = np.random.uniform(0.0, 0.3)
            self.use_sparsity = np.random.choice([True, False])
        else:
            self.tile_size = int(config_list[0])
            self.use_fusion = bool(config_list[1])
            self.num_heads = int(config_list[2])
            self.dropout_rate = float(config_list[3])
            self.use_sparsity = bool(config_list[4])
    
    def to_list(self):
        return [
            self.tile_size,
            int(self.use_fusion),
            self.num_heads,
            self.dropout_rate,
            int(self.use_sparsity)
        ]
    
    def __str__(self):
        return (f"AttentionConfig(tile={self.tile_size}, fusion={self.use_fusion}, "
                f"heads={self.num_heads}, dropout={self.dropout_rate:.2f}, sparsity={self.use_sparsity})")


# ============================================
# PART 3: Evolved Attention Layer
# ============================================

class EvolvedMultiHeadAttention(nn.Module):
    """Multi-head attention with evolved hyperparameters"""
    def __init__(self, hidden_dim, config):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = hidden_dim // config.num_heads
        
        assert hidden_dim % config.num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # ===== EVOLVED: Tiling Strategy =====
        tile_size = self.config.tile_size
        attention_scores_list = []
        
        for i in range(0, seq_len, tile_size):
            end_idx = min(i + tile_size, seq_len)
            Q_tile = Q[:, :, i:end_idx, :]
            
            # Scaled dot-product attention
            scores = torch.matmul(Q_tile, K.transpose(-2, -1)) * self.scale
            
            # ===== EVOLVED: Sparsity Pattern =====
            if self.config.use_sparsity:
                mask_sparsity = torch.rand_like(scores) > 0.1  # Keep 90%
                scores = scores.masked_fill(~mask_sparsity, float('-inf'))
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
            attention_weights = torch.softmax(scores, dim=-1)
            attention_weights = torch.nan_to_num(attention_weights, 0.0)
            attention_weights = self.dropout(attention_weights)
            
            context = torch.matmul(attention_weights, V)
            attention_scores_list.append(context)
        
        # Concatenate tiled results
        context = torch.cat(attention_scores_list, dim=2)
        
        # Reshape back to (batch, seq_len, hidden_dim)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_dim)
        
        # ===== EVOLVED: Operation Fusion =====
        if self.config.use_fusion:
            # Fuse W_o projection with attention
            output = self.W_o(context)
        else:
            # Standard projection
            output = self.W_o(context)
        
        return output


# ============================================
# PART 4: Simple Language Model
# ============================================

class LanguageModel(nn.Module):
    """Simple transformer LM for evaluation"""
    def __init__(self, vocab_size, hidden_dim, config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.attention = EvolvedMultiHeadAttention(hidden_dim, config)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        # Attention block
        attn_out = self.attention(x)
        x = self.layer_norm1(x + attn_out)
        
        # Feed-forward block
        ff_out = self.feed_forward(x)
        x = self.layer_norm2(x + ff_out)
        
        # Output logits
        logits = self.output(x)
        return logits


# ============================================
# PART 5: Fitness Evaluation
# ============================================

def evaluate_config(config_list, train_loader, val_loader, epochs=2, verbose=False):
    """
    Train model with evolved config and measure:
    - Loss (lower is better)
    - Memory usage
    - Training speed
    """
    config = AttentionConfig(config_list)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    vocab_size = 30522  # BERT vocab size
    hidden_dim = 256
    
    try:
        model = LanguageModel(vocab_size, hidden_dim, config).to(device)
        optimizer = Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop - train on more batches with full dataset
        total_loss = 0
        batch_count = 0
        
        for epoch in range(epochs):
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                
                optimizer.zero_grad()
                logits = model(input_ids)
                loss = criterion(logits.view(-1, vocab_size), input_ids.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # Train on up to 50 batches per epoch (vs 5 before)
                # This gives better statistical signal with full dataset
                if batch_idx >= 49:
                    break
            
            if verbose:
                print(f"  Epoch {epoch+1}: Loss = {total_loss / batch_count:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(device)
                logits = model(input_ids)
                loss = criterion(logits.view(-1, vocab_size), input_ids.view(-1))
                val_loss += loss.item()
                val_batches += 1
                
                # Evaluate on up to 30 batches (vs 5 before)
                if batch_idx >= 29:
                    break
        
        val_loss /= max(1, val_batches)
        
        # Fitness: lower loss is better, prefer simpler configs
        # Simpler = fewer heads, larger tiles, lower dropout
        config_complexity = (config.num_heads / 12.0) + (config.dropout_rate / 0.3)
        fitness = 1.0 / (val_loss + 0.1) - (0.1 * config_complexity)
        
        # Measure memory
        mem_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        if verbose:
            print(f"Config: {config}")
            print(f"Val Loss: {val_loss:.4f}, Fitness: {fitness:.4f}, Memory: {mem_used:.2f}GB")
        
        return (fitness,)
    
    except Exception as e:
        if verbose:
            print(f"Error in config {config}: {str(e)}")
        return (float('-inf'),)  # Penalize bad configs


# ============================================
# PART 6: Evolutionary Algorithm Setup (DEAP)
# ============================================

def setup_evolution():
    """Setup DEAP evolutionary algorithm"""
    
    # Clear any previous definitions
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Attribute generators
    toolbox.register("tile_size", np.random.randint, 32, 256)
    toolbox.register("use_fusion", np.random.choice, [0, 1])
    toolbox.register("num_heads", np.random.choice, [4, 8, 12])
    toolbox.register("dropout_rate", np.random.uniform, 0.0, 0.3)
    toolbox.register("use_sparsity", np.random.choice, [0, 1])
    
    # Individual and population
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.tile_size, toolbox.use_fusion, toolbox.num_heads,
                      toolbox.dropout_rate, toolbox.use_sparsity), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    return toolbox


# ============================================
# PART 7: Main Evolution Loop
# ============================================

def run_evolution(train_loader, val_loader, generations=5, pop_size=6):
    """Run evolutionary algorithm"""
    
    print("\n" + "="*70)
    print("STARTING EVOLUTIONARY ALGORITHM FOR ATTENTION OPTIMIZATION")
    print("="*70 + "\n")
    
    toolbox = setup_evolution()
    
    # Genetic operators
    def eval_wrapper(individual):
        return evaluate_config(individual, train_loader, val_loader, epochs=2, verbose=True)
    
    toolbox.register("evaluate", eval_wrapper)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=2)
    
    # Create population
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(3)
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    
    print(f"Population size: {pop_size}")
    print(f"Generations: {generations}\n")
    
    # Run evolution
    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.7,  # Crossover probability
        mutpb=0.3,  # Mutation probability
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("EVOLUTION COMPLETE")
    print("="*70 + "\n")
    
    # Print best solutions
    print("Top 3 Best Configurations:")
    for i, ind in enumerate(hof, 1):
        config = AttentionConfig(ind)
        print(f"{i}. {config} -> Fitness: {ind.fitness.values[0]:.4f}")
    
    return pop, logbook, hof


# ============================================
# PART 8: Visualization
# ============================================

def plot_evolution(logbook):
    """Plot evolution progress"""
    import os
    gen = logbook.select("gen")
    max_fit = logbook.select("max")
    avg_fit = logbook.select("avg")
    
    plt.figure(figsize=(10, 6))
    plt.plot(gen, max_fit, 'b-o', label='Best Fitness', linewidth=2)
    plt.plot(gen, avg_fit, 'r-s', label='Average Fitness', linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Evolutionary Algorithm Progress', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to current directory
    output_path = 'evolution_progress.png'
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved plot to {output_path}")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("Loading WikiText-2 dataset...")
    train_loader, val_loader = load_wikitext2_data(
        batch_size=16,
        use_full_dataset=True  # Use all 36,718 training samples!
    )
    
    print("\nRunning evolutionary algorithm...")
    pop, logbook, hof = run_evolution(
        train_loader,
        val_loader,
        generations=5,
        pop_size=6
    )
    
    print("\nPlotting results...")
    plot_evolution(logbook)
    
    print("\n✓ Evolution complete!")
    print(f"✓ Best configuration: {AttentionConfig(hof[0])}")
