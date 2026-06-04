from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import os
import warnings
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from data_jepa_tit_cftr import CompositeImageTextDataset, split_data
from torch.utils.data import DataLoader

def get_matscibert_embeddings(tokenizer, matscibert, text_data, device):
    inputs = tokenizer(text_data, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device=device) for k, v in inputs.items()}
    matscibert_output = matscibert(**inputs)
    text_embeddings = matscibert_output.last_hidden_state  # [batch, seq_len, matscibert_hidden_dim]
    pooled_embeddings = text_embeddings.mean(dim=1).to(device=device)  # [batch, matscibert_hidden_dim]
    return pooled_embeddings



def plot_tsne_embeddings(model, dataloader, save_path='./Results/tsne_embeddings_physics.png', 
                         max_samples=500, perplexity=30, n_iter=1000):
    """
    Visualise les embeddings avec t-SNE, colorés par la variable cible
    
    Args:
        model: Modèle supervisé (SupervisedElasticModulusPredictor)
        dataloader: DataLoader pour les données
        save_path: Chemin pour sauvegarder le plot
        max_samples: Nombre maximum d'échantillons à visualiser
        perplexity: Perplexité pour t-SNE (typiquement 5-50)
        n_iter: Nombre d'itérations pour t-SNE
    """
    print("\n" + "="*80)
    print("VISUALISATION T-SNE DES EMBEDDINGS")
    print("="*80)
    
    model.eval()
    all_embeddings = []
    all_targets = []
    text_model_name = 'm3rg-iitd/matscibert'    
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    matscibert = AutoModel.from_pretrained(text_model_name)
    matscibert = matscibert.to(model.device)
    matscibert_hidden_dim = matscibert.config.hidden_size
    for param in matscibert.parameters():
        param.requires_grad = False

    print("Extraction des embeddings...")
    with torch.no_grad():
        sample_count = 0
        for batch in dataloader:
            if sample_count >= max_samples:
                break
            
            images = batch['image']
            properties = batch['properties']
            text = batch['text']
            targets = batch['target']
            matscibert_embeddings_pooled = get_matscibert_embeddings(tokenizer, matscibert, text,device=model.device) 

            # Obtenir les embeddings du modèle non supervisé
            forward_output = model(images=images, tabular_data=properties, text_data=matscibert_embeddings_pooled, apply_masking=False, mask_ratio=None)
            enhanced_embeddings = forward_output['original_embeddings']
            
            # Pool les embeddings (comme dans le forward du modèle supervisé)
            image_emb = enhanced_embeddings['image']  # [batch, seq_len, combined_dim]
            #tabular_emb = enhanced_embeddings['tabular']  # [batch, 1, combined_dim]
            
            # Attention pooling pour images (comme dans le modèle)
            if hasattr(model, 'image_attention'):
                attn_weights = model.image_attention(image_emb)  # [batch, seq_len, 1]
                attn_weights = F.softmax(attn_weights, dim=1)
                image_pooled = (image_emb * attn_weights).sum(dim=1)  # [batch, combined_dim]
            else:
                image_pooled = image_emb.mean(dim=1)  # [batch, combined_dim]
            
            # Pooling simple pour tabular
            #tabular_pooled = tabular_emb.mean(dim=1)  # [batch, combined_dim]
            
            # Concaténer les embeddings
            #combined_emb = torch.cat([image_pooled, tabular_pooled], dim=1)  # [batch, 2*combined_dim]
            combined_emb = image_pooled
            
            # Stocker
            batch_size = combined_emb.size(0)
            all_embeddings.append(combined_emb.cpu().numpy())
            all_targets.append(targets.numpy())
            
            sample_count += batch_size
    
    # Concaténer tous les embeddings
    all_embeddings = np.vstack(all_embeddings)
    all_targets = np.concatenate(all_targets)
    
    print(f"   Total samples: {len(all_embeddings)}")
    print(f"   Embedding dimension: {all_embeddings.shape[1]}")
    print(f"   Target range: [{np.min(all_targets):.4f}, {np.max(all_targets):.4f}]")
    
    # Détecter si les targets sont discrètes (binaires ou multi-classes) ou continues
    unique_targets = np.unique(all_targets)
    num_unique = len(unique_targets)
    
    # Détecter si c'est discret (binaire ou multi-classe)
    # Considérer comme discret si:
    # 1. <= 10 valeurs uniques ET
    # 2. Les valeurs sont des entiers ou très proches d'entiers (tolérance 0.01)
    is_discrete = False
    if num_unique <= 10:
        # Vérifier si les valeurs sont des entiers (ou très proches)
        are_integers = np.allclose(unique_targets, np.round(unique_targets), atol=0.01)
        if are_integers:
            is_discrete = True
    
    if is_discrete:
        print(f"   ✅ Target détectée comme DISCRÈTE ({num_unique} classes): valeurs uniques = {unique_targets}")
        # Convertir en entiers pour les classes
        discrete_targets = np.round(all_targets).astype(int)
        # Normaliser les classes pour commencer à 0 (si elles commencent à 1, 2, etc.)
        unique_discrete = np.unique(discrete_targets)
        if len(unique_discrete) > 1 and unique_discrete[0] != 0:
            # Remapper pour commencer à 0
            mapping = {old: new for new, old in enumerate(unique_discrete)}
            discrete_targets = np.array([mapping[t] for t in discrete_targets])
            print(f"   Classes remappées: {unique_discrete} -> {np.unique(discrete_targets)}")
    else:
        print(f"   Target détectée comme CONTINUE: {num_unique} valeurs uniques")
        discrete_targets = None
    
    # Limiter le nombre d'échantillons si nécessaire
    if len(all_embeddings) > max_samples:
        indices = np.random.choice(len(all_embeddings), max_samples, replace=False)
        all_embeddings = all_embeddings[indices]
        all_targets = all_targets[indices]
        if is_discrete:
            discrete_targets = discrete_targets[indices]
        print(f"Using {max_samples} random samples for t-SNE")
    
    # Appliquer t-SNE
    print(f"\nApplication de t-SNE (perplexity={perplexity}, n_iter={n_iter})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42, verbose=1)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Créer le plot
    print("Création du plot...")
    plt.figure(figsize=(12, 8))
    
    # Scatter plot avec couleurs adaptées au type de target
    if is_discrete:
        # Pour targets discrètes (binaires ou multi-classes): utiliser des couleurs discrètes
        num_classes = len(np.unique(discrete_targets))
        
        # Palette de couleurs pour multi-classes
        if num_classes == 2:
            # Binaire: rouge et bleu
            color_map = {0: 'red', 1: 'blue'}
        elif num_classes == 3:
            # 3 classes: rouge, vert, bleu
            color_map = {0: 'red', 1: 'green', 2: 'blue'}
        elif num_classes <= 10:
            # Multi-classes: utiliser une palette de couleurs distinctes
            import matplotlib.cm as cm
            colors_list = cm.tab10(np.linspace(0, 1, num_classes))
            color_map = {i: colors_list[i] for i in range(num_classes)}
        else:
            # Trop de classes: utiliser une colormap continue
            color_map = None
        
        if color_map is not None:
            colors = [color_map[t] for t in discrete_targets]
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                 c=colors, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
            
            # Légende au lieu de colorbar
            from matplotlib.patches import Patch
            legend_elements = []
            for class_id in sorted(np.unique(discrete_targets)):
                count = np.sum(discrete_targets == class_id)
                legend_elements.append(
                    Patch(facecolor=color_map[class_id], label=f'Class {class_id} (n={count})')
                )
            plt.legend(handles=legend_elements, loc='upper right')
        else:
            # Trop de classes: utiliser colorbar discrète
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                 c=discrete_targets, cmap='tab10', alpha=0.6, s=50)
            cbar = plt.colorbar(scatter)
            cbar.set_label('Class', rotation=270, labelpad=20)
    else:
        # Pour targets continues: utiliser colorbar
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=all_targets, cmap='viridis', alpha=0.6, s=50)
        
        # Colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Target Value', rotation=270, labelpad=20)
    
    # Labels et titre
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    if is_discrete:
        num_classes = len(np.unique(discrete_targets))
        if num_classes == 2:
            plt.title('t-SNE Visualization of Embeddings (colored by binary target)', fontsize=14, fontweight='bold')
        else:
            plt.title(f't-SNE Visualization of Embeddings (colored by {num_classes}-class target)', fontsize=14, fontweight='bold')
    else:
        plt.title('t-SNE Visualization of Embeddings (colored by continuous target)', fontsize=14, fontweight='bold')
    
    # Ajouter des statistiques adaptées au type de target
    stats_text = f'Samples: {len(all_targets)}\n'
    if is_discrete:
        unique_classes = np.unique(discrete_targets)
        for class_id in sorted(unique_classes):
            count = np.sum(discrete_targets == class_id)
            percentage = 100 * count / len(discrete_targets)
            stats_text += f'Class {class_id}: {count} ({percentage:.1f}%)\n'
        stats_text = stats_text.rstrip('\n')  # Enlever le dernier \n
    else:
        stats_text += f'Target mean: {np.mean(all_targets):.4f}\n'
        stats_text += f'Target std: {np.std(all_targets):.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    
    # Sauvegarder
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Plot sauvegardé: {save_path}")
    
    # Afficher
    plt.show()
    
    # Analyse de la séparabilité
    print("\n📊 Analyse de la séparabilité:")
    # Calculer la corrélation entre distance dans l'espace t-SNE et différence de targets
    
    # Distances dans l'espace t-SNE
    tsne_distances = squareform(pdist(embeddings_2d))
    
    # Différences de targets
    target_diffs = squareform(pdist(all_targets.reshape(-1, 1)))
    
    # Corrélation (sur un échantillon pour éviter la mémoire)
    sample_size = min(1000, len(all_targets))
    sample_indices = np.random.choice(len(all_targets), sample_size, replace=False)
    tsne_sample = tsne_distances[np.ix_(sample_indices, sample_indices)]
    target_sample = target_diffs[np.ix_(sample_indices, sample_indices)]
    
    # Prendre seulement la partie triangulaire supérieure (sans diagonale)
    mask = np.triu(np.ones_like(tsne_sample, dtype=bool), k=1)
    tsne_flat = tsne_sample[mask]
    target_flat = target_sample[mask]
    
    correlation, p_value = pearsonr(tsne_flat, target_flat)
    print(f"   Corrélation distance t-SNE vs différence targets: {correlation:.4f} (p={p_value:.4e})")
    
    if correlation > 0.3:
        print("   ✅ Bonne séparabilité: les embeddings semblent contenir de l'information sur les targets")
    elif correlation > 0.1:
        print("   ⚠️ Séparabilité modérée: les embeddings contiennent un peu d'information")
    else:
        print("   ❌ Mauvaise séparabilité: les embeddings ne semblent pas contenir d'information utile")
    
    model.train()
    return embeddings_2d, all_targets


def check_model_learning(model, dataloader, epoch):
    """
    Vérifier si le modèle apprend des représentations utiles
    Détecte le collapse des embeddings et l'over-alignment
    """
    model.eval()
    embeddings_stats = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 5:  # Vérifier seulement quelques batches
                break
            
            images = batch['image']
            properties = batch['properties']
            
            try:
                output = model(images, properties, apply_masking=False)
                
                # Extraire les embeddings améliorés
                img_emb = output['enhanced_embeddings']['image'].mean(dim=1)  # [batch, dim]
                tab_emb = output['enhanced_embeddings']['tabular'].mean(dim=1)  # [batch, dim]
                
                # Calculer la variance (diversité des embeddings)
                img_var = img_emb.var(dim=0).mean().item()
                tab_var = tab_emb.var(dim=0).mean().item()
                
                # Calculer la similarité cosinus moyenne entre modalités
                cosine_sim = F.cosine_similarity(img_emb, tab_emb, dim=1).mean().item()
                
                # Calculer la distance euclidienne moyenne
                euclidean_dist = torch.norm(img_emb - tab_emb, dim=1).mean().item()
                
                embeddings_stats.append({
                    'img_var': img_var,
                    'tab_var': tab_var,
                    'cosine_sim': cosine_sim,
                    'euclidean_dist': euclidean_dist
                })
            except Exception as e:
                print(f"  ⚠️ Error in check_model_learning: {e}")
                break
    
    if embeddings_stats:
        avg_img_var = np.mean([e['img_var'] for e in embeddings_stats])
        avg_tab_var = np.mean([e['tab_var'] for e in embeddings_stats])
        avg_sim = np.mean([e['cosine_sim'] for e in embeddings_stats])
        avg_dist = np.mean([e['euclidean_dist'] for e in embeddings_stats])
        
        print(f"\n🔍 Model Learning Check (Epoch {epoch + 1}):")
        print(f"   Image embedding variance: {avg_img_var:.6f} (higher = more diverse)")
        print(f"   Tabular embedding variance: {avg_tab_var:.6f} (higher = more diverse)")
        print(f"   Cross-modal cosine similarity: {avg_sim:.4f} (0-1, higher = more aligned)")
        print(f"   Cross-modal euclidean distance: {avg_dist:.4f} (higher = more separated)")
        
        # Détecter les problèmes
        warnings = []
        if avg_img_var < 1e-6:
            warnings.append("⚠️ Image embedding collapse detected! (variance too low)")
        if avg_tab_var < 1e-6:
            warnings.append("⚠️ Tabular embedding collapse detected! (variance too low)")
        if avg_sim > 0.99:
            warnings.append("⚠️ Over-alignment detected! (modalities too similar, may lose information)")
        if avg_sim < 0.1:
            warnings.append("⚠️ Under-alignment detected! (modalities too different, may not learn cross-modal relationships)")
        
        if warnings:
            for warning in warnings:
                print(f"   {warning}")
        else:
            print(f"   ✅ Embeddings look healthy")
    else:
        print(f"   ⚠️ Could not compute embedding statistics")
    
    model.train()



def synthetic_stiffness_rule_of_mixtures(vf, seed=42, rng=None):
    """
    Physics-informed: longitudinal stiffness of a unidirectional composite.
    E_eff(vf) ≈ vf * Ef + (1 - vf) * Em, with a mild nonlinearity.
    
    vf : np.ndarray, shape (N,), volume fraction in [0, 1]
    returns: np.ndarray, shape (N,), effective stiffness (GPa-like, arbitrary scale)
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    # Fiber and matrix moduli (GPa) – physically Ef >> Em
    Ef = rng.uniform(150.0, 250.0)   # e.g. carbon fiber
    Em = rng.uniform(2.0, 5.0)       # e.g. polymer matrix

    # Mild nonlinearity exponent to model imperfect load transfer
    alpha = rng.uniform(0.8, 1.2)

    vf_clipped = np.clip(vf, 0.0, 1.0)
    vf_eff = vf_clipped**alpha

    E_eff = vf_eff * Ef + (1.0 - vf_eff) * Em   # rule of mixtures with adjusted vf

    # Optional small noise
    noise = rng.normal(0.0, 0.02 * E_eff.mean(), size=E_eff.shape)
    return E_eff + noise


def synthetic_strength_optimal_vf(vf, seed=42, rng=None):
    """
    Physics-informed: tensile strength peaking at an optimal vf.
    - Too little fiber → weak.
    - Too much fiber → brittle, processing defects → strength drops.
    
    We model this with a peaked curve around vf_opt.
    
    vf : np.ndarray, shape (N,), volume fraction in [0, 1]
    returns: np.ndarray, shape (N,), ultimate strength (MPa-like, arbitrary scale)
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    vf_clipped = np.clip(vf, 0.0, 1.0)

    # Optimal volume fraction somewhere realistic, e.g. 0.5–0.7
    vf_opt = rng.uniform(0.5, 0.7)

    # Peak strength at vf_opt
    sigma_peak = rng.uniform(600.0, 1200.0)  # MPa-like

    # Width of the peak (how quickly strength drops away from vf_opt)
    width = rng.uniform(0.1, 0.25)

    # Gaussian-like peak centered at vf_opt, zero at extremes
    # This is physically “best at vf_opt, worse if too low/high”
    strength_shape = np.exp(-0.5 * ((vf_clipped - vf_opt) / width) ** 2)

    # Also enforce that strength goes to zero as vf → 0
    strength_shape *= vf_clipped / max(vf_opt, 1e-6)

    sigma_eff = sigma_peak * strength_shape

    noise = rng.normal(0.0, 0.03 * sigma_eff.max(), size=sigma_eff.shape)
    return np.maximum(0.0, sigma_eff + noise)



def synthetic_E_longitudinal(vf, Ef=230e3, Em=3e3, noise_std=0.0, rng=None):
    """
    Physics-informed longitudinal modulus (E_parallel) vs volume fraction.
    
    vf       : array-like, fiber volume fraction in [0, 1]
    Ef, Em   : fiber and matrix moduli (MPa) – defaults ~carbon fiber + epoxy
    noise_std: relative noise level, e.g. 0.02 for 2% std
    """
    if rng is None:
        rng = np.random.default_rng()
    vf = np.clip(np.asarray(vf, dtype=np.float64), 0.0, 1.0)
    
    # Voigt rule of mixtures (iso-strain)
    E_eff = vf * Ef + (1.0 - vf) * Em  # MPa

    if noise_std > 0:
        noise = rng.normal(0.0, noise_std * E_eff, size=E_eff.shape)
        E_eff = E_eff + noise

    return E_eff.astype(np.float32)

def synthetic_E_transverse_halpin_tsai(vf, Ef=230e3, Em=3e3, xi=1.5, noise_std=0.0, rng=None):
 
    """
    Physics-informed transverse modulus (E_perp) vs volume fraction
    using the Halpin–Tsai equation.
    
    vf       : array-like, fiber volume fraction in [0, 1]
    Ef, Em   : fiber and matrix moduli (MPa)
    xi       : Halpin–Tsai parameter (geometry/orientation)
    noise_std: relative noise level
    """
    if rng is None:
        rng = np.random.default_rng()
    vf = np.clip(np.asarray(vf, dtype=np.float64), 0.0, 1.0)

    # stiffness ratio
    r = Ef / Em
    # Halpin–Tsai eta
    eta = (r - 1.0) / (r + xi)

    # Halpin–Tsai transverse modulus
    num = 1.0 + xi * eta * vf
    den = 1.0 - eta * vf
    E_eff = Em * num / den  # MPa

    if noise_std > 0:
        noise = rng.normal(0.0, noise_std * E_eff, size=E_eff.shape)
        E_eff = E_eff + noise

    return E_eff.astype(np.float32)



def setup_logging(log_dir='./Results/Logs'):
    """Configure logging to both file and console"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_physics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def run_classical_ml(train_all_embeddings, train_all_targets,test_all_embeddings, test_all_targets,targets_mean,targets_std):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    X_train, X_test = train_all_embeddings, test_all_embeddings
    y_train, y_test = train_all_targets, test_all_targets
    models = [
        Ridge(alpha=1.0),
        RandomForestRegressor(n_estimators=50),
        KNeighborsRegressor(n_neighbors=5),
        GradientBoostingRegressor(n_estimators=50),
    ]
    for model in models:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        #pred = (pred * targets_std) + targets_mean
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        print(f"R² avec {model.__class__.__name__}: {r2:.4f}")
        print(f"RMSE avec {model.__class__.__name__}: {rmse:.4f}")
        print(f"MAE avec {model.__class__.__name__}: {mae:.4f}")
        print("-"*80)


def debug_modalities(model, train_dataloader, test_dataloader):
  
    target_list = []
    for b in train_dataloader:
        target_list.append(b['target'].float())
    target_list = np.concatenate(target_list)
    targets_mean = target_list.mean()
    targets_std = target_list.std()
    train_embedding_list = []
    train_targets_list = []
    model.eval()
    with torch.no_grad():
        for b in train_dataloader:
            images = b['image'].to(model.device)  # Not used but kept for compatibility
            text = b['text']
            properties = b['properties'].to(model.device)
            targets = b['target'].to(model.device).float()
            
            # Get tabular embeddings from the model
            forward_output = model(None, properties, text, apply_masking=False)
            tabular_embeddings = forward_output['original_embeddings']['tabular']  # [batch, hidden_dim]
            
            train_embedding_list.append(tabular_embeddings.detach().cpu().numpy())
            train_targets_list.append(targets.detach().cpu().numpy())
    train_all_embeddings = np.concatenate(train_embedding_list)
    train_all_targets = np.concatenate(train_targets_list)
    test_embedding_list = []
    test_targets_list = []
    for b in test_dataloader:
        images = b['image'].to(model.device)  # Not used but kept for compatibility
        text = b['text']
        properties = b['properties'].to(model.device)
        targets = b['target'].to(model.device).float()
        
        # Get tabular embeddings from the model
        forward_output = model(None, properties, text, apply_masking=False)
        tabular_embeddings = forward_output['original_embeddings']['tabular']  # [batch, hidden_dim]
        test_embedding_list.append(tabular_embeddings.detach().cpu().numpy())
        test_targets_list.append(targets.detach().cpu().numpy())
    test_all_embeddings = np.concatenate(test_embedding_list)
    test_all_targets = np.concatenate(test_targets_list)
    run_classical_ml(train_all_embeddings, train_all_targets,test_all_embeddings, test_all_targets,targets_mean,targets_std)


def diagnose_unsupervised_embeddings(model, dataloader):
    """
    Diagnostic pour vérifier si les embeddings du modèle non supervisé sont utiles
    (DEPRECATED - Use fine_tune_supervised_property_predictor instead)
    """
    print("WARNING: diagnose_unsupervised_embeddings is deprecated.")
    print("Use fine_tune_supervised_property_predictor for better results with cross-attention.")
    model.eval()
    all_embeddings = []
    all_targets = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10:  # Juste quelques batches
                break
            
            images = batch['image']
            properties = batch['properties']
            text = batch['text']
            targets = batch['target']
            
            
            
            # Get tabular embeddings (for tabular JEPA, image is None)
            tab_emb = original_embeddings['tabular']  # [batch, hidden_dim]
            combined_emb = tab_emb
            
            all_embeddings.append(combined_emb.cpu().numpy())
            all_targets.append(targets.numpy())
    
    all_embeddings = np.vstack(all_embeddings)
    all_targets = np.concatenate(all_targets)
    
    # Vérifier la variance des embeddings
    emb_std = np.std(all_embeddings, axis=0)
    print(f"Embedding variance: mean={np.mean(emb_std):.6f}, min={np.min(emb_std):.6f}, max={np.max(emb_std):.6f}")
    
def check_alingment(model, train_dataloader):
    """Check alignment between aggregator and target aggregator for tabular JEPA"""
    model.eval()
    with torch.no_grad():
        for batch in train_dataloader:
            images = batch['image']  # Not used but kept for compatibility
            text = batch['text']
            properties = batch['properties'].to(model.device)
            
            # Get visible and masked columns
            visible_columns, masked_columns, visible_indices, masked_indices = \
                model._mask_tabular_columns(properties)
            
            # Encode each visible column individually
            visible_col_embeddings = []
            for i, col_idx in enumerate(visible_indices):
                col_value = visible_columns[:, i:i+1]  # [batch, 1]
                col_emb = model.column_encoder(col_value)  # [batch, column_embed_dim]
                col_type_emb = model.column_type_embeddings(col_idx)  # [column_embed_dim]
                col_emb = col_emb + col_type_emb.unsqueeze(0)  # [batch, column_embed_dim]
                col_processed = model.aggregator(col_emb)  # [batch, projected_dim]
                visible_col_embeddings.append(col_processed)
            
            # Aggregate visible columns (mean pooling)
            visible_col_embeddings = torch.stack(visible_col_embeddings, dim=1)  # [batch, 5, projected_dim]
            z_s = visible_col_embeddings.mean(dim=1)  # [batch, projected_dim]
            
            # Encode each masked column individually
            masked_col_embeddings = []
            for i, col_idx in enumerate(masked_indices):
                col_value = masked_columns[:, i:i+1]  # [batch, 1]
                col_emb = model.column_encoder(col_value)  # [batch, column_embed_dim]
                col_type_emb = model.column_type_embeddings(col_idx)  # [column_embed_dim]
                col_emb = col_emb + col_type_emb.unsqueeze(0)  # [batch, column_embed_dim]
                col_processed = model.target_aggregator(col_emb)  # [batch, projected_dim]
                masked_col_embeddings.append(col_processed)
            
            # Aggregate masked columns (mean pooling)
            masked_col_embeddings = torch.stack(masked_col_embeddings, dim=1)  # [batch, 2, projected_dim]
            z_t = masked_col_embeddings.mean(dim=1)  # [batch, projected_dim]
            
            # Predict masked columns from visible
            p_s = model.masked_column_predictor(z_s)  # [batch, projected_dim]
            
            # Compute cosine similarity
            cos_sim = F.cosine_similarity(p_s, z_t, dim=-1).mean().item()
            print("Mean teacher–student cosine:", cos_sim)
            mean_var = z_s.var(dim=0).mean().item()
            print(f"Mean variance of context features: {mean_var:.6f}")
            break  # Just check first batch

def zero_shot_prediction(
    model,
    property_predictor,
    test_dataloader,
    properties_mean,
    properties_std,
    targets_mean,
    targets_std,
    device,
):
    checkpoint_paths = [
        #"./Results/Checkpoint/property_predictor_combined_jepa_tit_no_cf.pth",
        # "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_100.pth",
        "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_10.pth",
        "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_50.pth",
        "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_50_elongation.pth"
        "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_10_elongation.pth'"  # maybe not same as previous?
    ]

    # 1. Collect test targets (once)
    test_targets = []
    with torch.no_grad():
        for batch in test_dataloader:
            targets = batch["target"].to(device).float()
            targets_normalized = (targets - targets_mean) / targets_std
            test_targets.append(targets_normalized.cpu().numpy())
    test_targets = np.concatenate(test_targets).flatten()  # shape (N,)

    # 2. Get predictions from each checkpoint
    all_predictions = []  # list of arrays, each shape (N,)
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # state_dict handling
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        property_predictor.load_state_dict(state_dict)
        property_predictor.to(device)
        property_predictor.eval()

        # Use your existing evaluation function
        preds = evaluate_supervised_property_predictor(
            model,
            property_predictor,
            test_dataloader,
            properties_mean,
            properties_std,
            targets_mean,
            targets_std,
            device,
        )
        # IMPORTANT: make sure preds are on the SAME SCALE as test_targets (both normalized or both de-normalized)
        all_predictions.append(np.array(preds).flatten())

    # 3. Stack to [N, K]
    Y = np.stack(all_predictions, axis=1)  # [N, K]
    N, K = Y.shape
    print(f"Prediction matrix Y shape: {Y.shape}")  # (N, K)

    y = test_targets  # [N,]

    # 4. Solve ridge in prediction space: a = (Y^T Y + lam I)^(-1) Y^T y
    lam = 0.0001
    YtY = Y.T @ Y          # [K, K]
    Yty = Y.T @ y          # [K,]

    A = YtY + lam * np.eye(K)
    a = np.linalg.solve(A, Yty)  # [K,]

    # 5. Combined predictions
    y_pred = Y @ a  # [N,]

    # 6. Metrics (still in normalized space if targets are normalized)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    print("\nZero-shot Combined Prediction Results:")
    print(f"R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    print(f"Combination weights (a): {a}")

    return y_pred, a

def zero_shot_prediction_nonlinear(
    model,
    property_predictor,
    train_dataloader,
    test_dataloader,
    properties_mean,
    properties_std,
    targets_mean,
    targets_std,
    device,
    method='polynomial',  # Options: 'mlp', 'polynomial', 'rbf', 'gradient_boosting', 'random_forest'
    lam=0.01,
    polynomial_degree=2,
    mlp_hidden_dim=64,
    test_with_random=False,
):
    """
    Zero-shot prediction avec régression non-linéaire, utilisant seulement test_dataloader
    comme la version linéaire. Les méthodes non-linéaires utilisent toutes les données de test
    pour l'entraînement et l'évaluation (comme la version linéaire).
    """
    checkpoint_paths = [
        # "./Results/Checkpoint/property_predictor_combined_jepa_tit_no_cf.pth",
        # "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_100.pth",
        # "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_10.pth",
        # "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_50.pth",
        # "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_k5_w10.pth",
        # "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_50_elongation.pth",
        # "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_10_elongation.pth",
        "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_5_synthetic_strength.pth",
        "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_50_synthetic_stiffness.pth",
        "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_50_synthetic_strength.pth",
        "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_10_synthetic_E_longitudinal.pth",
        "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_10_synthetic_E_transverse.pth"
    ]	
    

    # 1. Collecter les targets de test (comme la version linéaire)
    train_targets = []
    with torch.no_grad():
        for batch in train_dataloader:
            targets = batch["target"].to(device).float()
            targets_normalized = (targets - targets_mean) / targets_std
            train_targets.append(targets_normalized.cpu().numpy())
    train_targets = np.concatenate(train_targets).flatten()  # shape (N,)

    # 2. Obtenir les prédictions de chaque checkpoint (comme la version linéaire)
    train_predictions = []  # list of arrays, each shape (N,)
    for i, checkpoint_path in enumerate(checkpoint_paths):
        if test_with_random:
            # TEST: Générer des prédictions aléatoires
            print(f"⚠️  TEST MODE: Using random predictions for predictor {i+1}")
            np.random.seed(42 + i)
            preds = np.random.normal(targets_mean, targets_std, size=len(train_targets))
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            property_predictor.load_state_dict(state_dict)
            property_predictor.to(device)
            property_predictor.eval()

            # Use your existing evaluation function
            preds = evaluate_supervised_property_predictor(
                model,
                property_predictor,
                train_dataloader,
                properties_mean,
                properties_std,
                targets_mean,
                targets_std,
                device,
            )
        # IMPORTANT: make sure preds are on the SAME SCALE as test_targets (both normalized or both de-normalized)
        train_predictions.append(np.array(preds).flatten())

    # 3. Empiler les prédictions [N, K]
    Y_hat = np.stack(train_predictions, axis=1)  # [N, K]
    N, K = Y_hat.shape
    print(f"Prediction matrix Y_hat shape: {Y_hat.shape}")  # (N, K)

    y_gt = train_targets  # [N,]

    # # 4. Appliquer la méthode non-linéaire choisie
    # # Utiliser toutes les données pour entraîner et évaluer (comme la version linéaire)
    Y_train = Y_hat
    y_train = y_gt
    
   
    if method == 'polynomial':
        from sklearn.preprocessing import PolynomialFeatures
        from scipy.linalg import pinv, lstsq
        import warnings
        poly = PolynomialFeatures(degree=polynomial_degree, include_bias=True)
        Y_train_poly = poly.fit_transform(Y_train)
        N, P = Y_train_poly.shape
        print(f"Polynomial features: {N} samples × {P} features")
        try:
            Y_poly_pinv = pinv(Y_train_poly, rcond=1e-15)  # rcond pour le seuil de valeurs singulières
            w = Y_poly_pinv @ y_train
        
        except np.linalg.LinAlgError:
            # Fallback: utiliser lstsq qui est aussi analytique
            print("Using lstsq as fallback")
            w, residual, rank, s = lstsq(Y_train_poly, y_train, rcond=None)
            # y_pred = Y_train_poly @ w
            print(f"Matrix rank: {rank}/{P}, Residual: {residual[0]:.6e}")

        weights = w    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: 'mlp', 'polynomial', 'rbf', 'gradient_boosting', 'random_forest'")

     # 1. Collecter les targets de test (comme la version linéaire)
    test_targets = []
    with torch.no_grad():
        for batch in test_dataloader:
            targets = batch["target"].to(device).float()
            targets_normalized = (targets - targets_mean) / targets_std
            test_targets.append(targets_normalized.cpu().numpy())
    test_targets = np.concatenate(test_targets).flatten()  # shape (N,)

    # 2. Obtenir les prédictions de chaque checkpoint (comme la version linéaire)
    test_predictions = []  # list of arrays, each shape (N,)
    for i, checkpoint_path in enumerate(checkpoint_paths):
        if test_with_random:
            # TEST: Générer des prédictions aléatoires
            print(f"⚠️  TEST MODE: Using random predictions for predictor {i+1}")
            np.random.seed(42 + i)
            preds = np.random.normal(targets_mean, targets_std, size=len(test_targets))
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            property_predictor.load_state_dict(state_dict)
            property_predictor.to(device)
            property_predictor.eval()

            # Use your existing evaluation function
            preds = evaluate_supervised_property_predictor(
                model,
                property_predictor,
                test_dataloader,
                properties_mean,
                properties_std,
                targets_mean,
                targets_std,
                device,
            )
        # IMPORTANT: make sure preds are on the SAME SCALE as test_targets (both normalized or both de-normalized)
        test_predictions.append(np.array(preds).flatten())

    # 3. Empiler les prédictions [N, K]
    Y_test = np.stack(test_predictions, axis=1)  # [N, K]
    N, K = Y_test.shape
    print(f"Prediction matrix Y_test shape: {Y_test.shape}")  # (N, K)
    Y_test_poly = poly.transform(Y_test)
    y_pred_test = Y_test_poly @ weights
    y_pred_test_denormalized = (y_pred_test * targets_std) + targets_mean
    test_targets_denormalized = (test_targets * targets_std) + targets_mean

    # 5. Métriques (sur les mêmes données, comme la version linéaire)
    r2 = r2_score(test_targets_denormalized, y_pred_test_denormalized)
    rmse = np.sqrt(mean_squared_error(test_targets_denormalized, y_pred_test_denormalized))
    mae = mean_absolute_error(test_targets_denormalized, y_pred_test_denormalized)

    print(f"\nZero-shot Non-linear ({method}) Combined Prediction Results:")
    print(f"R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    
    return y_pred_test, weights

# def one_shot_prediction(
#     model,
#     property_predictor,
#     train_dataloader,
#     test_dataloader,
#     properties_mean,
#     properties_std,
#     targets_mean,
#     targets_std,
#     a_zero_shot,
#     device,
# ):
#     """
#     One-shot prediction: résout la régression ridge dans l'espace des paramètres,
#     combine les paramètres, initialise le prédicteur et entraîne pour 1 époque.
#     """
#     checkpoint_paths = [
#         #"./Results/Checkpoint/property_predictor_combined_jepa_tit_no_cf.pth",
#         "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_100.pth",
#         "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_10.pth",
#         "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_50.pth",
#     ]

#     # 1. Collecter les targets d'entraînement pour la régression ridge
#     # train_targets = []
#     # with torch.no_grad():
#     #     for batch in train_dataloader:
#     #         targets = batch["target"].to(device).float()
#     #         targets_normalized = (targets - targets_mean) / targets_std
#     #         train_targets.append(targets_normalized.cpu().numpy())
#     # train_targets = np.concatenate(train_targets).flatten()  # shape (N,)

#     # 2. Extraire les paramètres de chaque checkpoint et obtenir les prédictions
#     all_parameter_vectors = []  # Liste de vecteurs de paramètres aplatis
#     all_predictions = []  # Liste de prédictions pour la régression ridge
    
#     for checkpoint_path in checkpoint_paths:
#         checkpoint = torch.load(checkpoint_path, map_location=device)
        
#         # Gestion du state_dict
#         if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
#             state_dict = checkpoint["model_state_dict"]
#         else:
#             state_dict = checkpoint
        
#         property_predictor.load_state_dict(state_dict)
#         property_predictor.to(device)
#         property_predictor.eval()
        
#         # Extraire les paramètres et les aplatir en vecteur
#         param_vector = []
#         with torch.no_grad():
#             for param in property_predictor.parameters():
#                 param_vector.append(param.cpu().flatten().numpy())
#         param_vector = np.concatenate(param_vector)  # Vecteur aplati
#         all_parameter_vectors.append(param_vector)
        
#     #     # Obtenir les prédictions sur l'ensemble d'entraînement pour la régression ridge
#     #     train_preds = []
#     #     with torch.no_grad():
#     #         for batch in train_dataloader:
#     #             images = batch['image'].to(device)
#     #             text = batch['text']
#     #             properties = batch['properties'].to(device)
#     #             properties_normalized = (properties - properties_mean) / properties_std
                
#     #             forward_output = model(None, properties_normalized, text, apply_masking=False)
#     #             tabular_embeddings = forward_output['original_embeddings']['tabular']
                
#     #             predictions_normalized = property_predictor(tabular_embeddings).squeeze(-1)
#     #             train_preds.append(predictions_normalized.cpu().numpy())
        
#     #     train_preds = np.concatenate(train_preds).flatten()
#     #     all_predictions.append(train_preds)
    
#     # # 3. Empiler les paramètres et prédictions
#     # # Paramètres: [K, P] où K = nombre de checkpoints, P = nombre de paramètres
#     param_matrix = np.stack(all_parameter_vectors, axis=0)  # [K, P]
#     K, P = param_matrix.shape
#     print(f"Parameter matrix shape: {param_matrix.shape}")  # (K, P)
    
#     # # Prédictions: [N, K] où N = nombre d'échantillons d'entraînement
#     # Y = np.stack(all_predictions, axis=1)  # [N, K]
#     # N = Y.shape[0]
#     # print(f"Prediction matrix Y shape: {Y.shape}")  # (N, K)
    
#     # y = train_targets  # [N,]
    
#     # # 4. Résoudre la régression ridge dans l'espace des prédictions pour obtenir les poids
#     # lam = 0.0001
#     # YtY = Y.T @ Y          # [K, K]
#     # Yty = Y.T @ y          # [K,]
    
#     # A = YtY + lam * np.eye(K)
#     # a = np.linalg.solve(A, Yty)  # [K,] - poids de combinaison
    
#     # print(f"Combination weights (a): {a}")
    
#     a = a_zero_shot

#     # 5. Combiner les paramètres selon les poids
#     param_combined = sum(a[k] * param_matrix[k, :] for k in range(K))
#     param_combined = (param_matrix.T @ a).T  # [P,] - paramètres combinés
    
#     # 6. Initialiser un nouveau property_predictor avec les paramètres combinés
#     # Reconstruire le state_dict à partir du vecteur de paramètres combiné
#     combined_state_dict = {}
#     param_idx = 0
#     with torch.no_grad():
#         for name, param in property_predictor.named_parameters():
#             param_size = param.numel()
#             param_shape = param.shape
#             # Extraire les paramètres combinés pour cette couche
#             combined_param = param_combined[param_idx:param_idx + param_size]
#             combined_param = torch.from_numpy(combined_param).reshape(param_shape)
#             combined_state_dict[name] = combined_param
#             param_idx += param_size
    
#     # Créer un nouveau prédicteur et initialiser avec les paramètres combinés
#     new_property_predictor = SupervisedPropertyPredictor(
#         vit_hidden_dim=128,
#         num_properties=1,
#         hidden_dim=64,
#         dropout=0.2
#     )
#     new_property_predictor.load_state_dict(combined_state_dict)
#     new_property_predictor.to(device)
    
#     # 7. Entraîner pour 1 époque
#     print("\n" + "="*80)
#     print("ONE-SHOT PREDICTION: Training combined predictor for 1 epoch")
#     print("="*80)
    
#     # Préparer l'entraînement (similaire à fine_tune_supervised_property_predictor)
#     for p in model.context_encoder.parameters():
#         p.requires_grad = False
#     for p in model.target_encoder.parameters():
#         p.requires_grad = False
#     for p in model.masked_column_predictor.parameters():
#         p.requires_grad = False
#     for p in new_property_predictor.parameters():
#         p.requires_grad = True
    
#     def get_tabular_embeddings(properties, text):
#         """Extract tabular embeddings from the model"""
#         with torch.no_grad():
#             model.eval()
#             forward_output = model(None, properties, text, apply_masking=False)
#             tabular_embeddings = forward_output['original_embeddings']['tabular']
#         return tabular_embeddings
    
#     optimizer = torch.optim.AdamW(
#         new_property_predictor.parameters(),
#         lr=1e-4,
#         weight_decay=1e-4
#     )
#     criterion = nn.MSELoss()
    
#     model.train()
#     new_property_predictor.train()
    
#     # Entraînement pour 1 époque
#     train_loss = 0.0
#     num_batches = 0
    
#     for batch_idx, batch in enumerate(train_dataloader):
#         optimizer.zero_grad()
        
#         images = batch['image'].to(device)
#         text = batch['text']
#         properties = batch['properties'].to(device)
#         targets = batch['target'].to(device).float()
        
#         targets_normalized = (targets - targets_mean) / targets_std
#         properties_normalized = (properties - properties_mean) / properties_std
        
#         tabular_embeddings = get_tabular_embeddings(properties_normalized, text)
#         predictions = new_property_predictor(tabular_embeddings)
        
#         loss = criterion(predictions.squeeze(-1), targets_normalized)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(new_property_predictor.parameters(), max_norm=1.0)
#         optimizer.step()
        
#         train_loss += loss.item()
#         num_batches += 1
    
#     avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
#     print(f"Training Loss (1 epoch): {avg_train_loss:.6f}")
    
#     # 8. Évaluer sur l'ensemble de test
#     new_property_predictor.eval()
#     test_predictions = []
#     test_targets = []
    
#     with torch.no_grad():
#         model.eval()
#         for batch in test_dataloader:
#             images = batch['image'].to(device)
#             text = batch['text']
#             properties = batch['properties'].to(device)
#             targets = batch['target'].to(device).float()
            
#             properties_normalized = (properties - properties_mean) / properties_std
#             targets_normalized = (targets - targets_mean) / targets_std
            
#             tabular_embeddings = get_tabular_embeddings(properties_normalized, text)
#             predictions_normalized = new_property_predictor(tabular_embeddings).squeeze(-1)
#             predictions = (predictions_normalized * targets_std) + targets_mean
            
#             test_predictions.append(predictions.cpu().numpy())
#             test_targets.append(targets.cpu().numpy())
    
#     test_predictions = np.concatenate(test_predictions).flatten()
#     test_targets = np.concatenate(test_targets).flatten()
    
#     test_r2 = r2_score(test_targets, test_predictions)
#     test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
#     test_mae = mean_absolute_error(test_targets, test_predictions)
    
#     print("\nOne-shot Prediction Results (after 1 epoch training):")
#     print(f"Test R²: {test_r2:.4f} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}")
    
#     return test_predictions, a

# def zero_shot_prediction_nonlinear(
#     model,
#     property_predictor,
#     train_dataloader,  # Ajouté pour l'entraînement du modèle non-linéaire
#     test_dataloader,
#     properties_mean,
#     properties_std,
#     targets_mean,
#     targets_std,
#     device,
#     method='mlp',  # Options: 'mlp', 'polynomial', 'rbf', 'gradient_boosting', 'random_forest'
#     lam=0.0001,
#     polynomial_degree=2,  # Pour méthode 'polynomial'
#     mlp_hidden_dim=64,  # Pour méthode 'mlp'
# ):
#     """
#     Zero-shot prediction avec régression non-linéaire pour combiner les prédictions.
    
#     Args:
#         method: 'mlp', 'polynomial', 'rbf', 'gradient_boosting', 'random_forest'
#     """
#     checkpoint_paths = [
#         "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_100.pth",
#         "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_10.pth",
#         "./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_50.pth",
#     ]

#     # 1. Collecter les targets d'entraînement et de test
#     train_targets = []
#     with torch.no_grad():
#         for batch in train_dataloader:
#             targets = batch["target"].to(device).float()
#             targets_normalized = (targets - targets_mean) / targets_std
#             train_targets.append(targets_normalized.cpu().numpy())
#     train_targets = np.concatenate(train_targets).flatten()
    
#     test_targets = []
#     with torch.no_grad():
#         for batch in test_dataloader:
#             targets = batch["target"].to(device).float()
#             targets_normalized = (targets - targets_mean) / targets_std
#             test_targets.append(targets_normalized.cpu().numpy())
#     test_targets = np.concatenate(test_targets).flatten()

#     # 2. Obtenir les prédictions de chaque checkpoint
#     train_predictions = []  # Pour l'entraînement du combiner
#     test_predictions = []   # Pour l'évaluation
    
#     for checkpoint_path in checkpoint_paths:
#         checkpoint = torch.load(checkpoint_path, map_location=device)
        
#         if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
#             state_dict = checkpoint["model_state_dict"]
#         else:
#             state_dict = checkpoint
        
#         property_predictor.load_state_dict(state_dict)
#         property_predictor.to(device)
#         property_predictor.eval()
        
#         # Prédictions sur train
#         train_preds = []
#         with torch.no_grad():
#             for batch in train_dataloader:
#                 images = batch['image'].to(device)
#                 text = batch['text']
#                 properties = batch['properties'].to(device)
#                 properties_normalized = (properties - properties_mean) / properties_std
                
#                 forward_output = model(None, properties_normalized, text, apply_masking=False)
#                 tabular_embeddings = forward_output['original_embeddings']['tabular']
#                 predictions_normalized = property_predictor(tabular_embeddings).squeeze(-1)
#                 train_preds.append(predictions_normalized.cpu().numpy())
#         train_predictions.append(np.concatenate(train_preds).flatten())
        
#         # Prédictions sur test
#         test_preds = evaluate_supervised_property_predictor(
#             model, property_predictor, test_dataloader,
#             properties_mean, properties_std, targets_mean, targets_std, device
#         )
#         test_predictions.append(np.array(test_preds).flatten())

#     # 3. Empiler les prédictions
#     Y_train = np.stack(train_predictions, axis=1)  # [N_train, K]
#     Y_test = np.stack(test_predictions, axis=1)   # [N_test, K]
#     K = Y_train.shape[1]
    
#     print(f"Train prediction matrix shape: {Y_train.shape}")
#     print(f"Test prediction matrix shape: {Y_test.shape}")

#     # 4. Appliquer la méthode non-linéaire choisie
#     if method == 'mlp':
#         # Méthode 1: Petit MLP pour combiner les prédictions
#         from sklearn.neural_network import MLPRegressor
        
#         mlp = MLPRegressor(
#             hidden_layer_sizes=(mlp_hidden_dim, mlp_hidden_dim // 2),
#             activation='relu',
#             solver='adam',
#             alpha=lam,  # L2 regularization
#             max_iter=500,
#             random_state=42,
#             early_stopping=True,
#             validation_fraction=0.1
#         )
#         mlp.fit(Y_train, train_targets)
#         y_pred = mlp.predict(Y_test)
        
#     elif method == 'polynomial':
#         # Méthode 2: Ridge avec features polynomiales
#         from sklearn.preprocessing import PolynomialFeatures
#         from sklearn.linear_model import Ridge
        
#         poly = PolynomialFeatures(degree=polynomial_degree, include_bias=True)
#         Y_train_poly = poly.fit_transform(Y_train)
#         Y_test_poly = poly.transform(Y_test)
        
#         ridge = Ridge(alpha=lam)
#         ridge.fit(Y_train_poly, train_targets)
#         y_pred = ridge.predict(Y_test_poly)
        
#     elif method == 'rbf':
#         # Méthode 3: Ridge avec kernel RBF (via KernelRidge)
#         from sklearn.kernel_ridge import KernelRidge
        
#         kr = KernelRidge(alpha=lam, kernel='rbf', gamma=0.1)
#         kr.fit(Y_train, train_targets)
#         y_pred = kr.predict(Y_test)
        
#     elif method == 'gradient_boosting':
#         # Méthode 4: Gradient Boosting
#         from sklearn.ensemble import GradientBoostingRegressor
        
#         gb = GradientBoostingRegressor(
#             n_estimators=100,
#             learning_rate=0.1,
#             max_depth=3,
#             random_state=42,
#             subsample=0.8
#         )
#         gb.fit(Y_train, train_targets)
#         y_pred = gb.predict(Y_test)
        
#     elif method == 'random_forest':
#         # Méthode 5: Random Forest
#         from sklearn.ensemble import RandomForestRegressor
        
#         rf = RandomForestRegressor(
#             n_estimators=100,
#             max_depth=5,
#             random_state=42,
#             n_jobs=-1
#         )
#         rf.fit(Y_train, train_targets)
#         y_pred = rf.predict(Y_test)
        
#     else:
#         raise ValueError(f"Unknown method: {method}. Choose from: 'mlp', 'polynomial', 'rbf', 'gradient_boosting', 'random_forest'")

#     # 5. Métriques
#     r2 = r2_score(test_targets, y_pred)
#     rmse = np.sqrt(mean_squared_error(test_targets, y_pred))
#     mae = mean_absolute_error(test_targets, y_pred)

#     print(f"\nZero-shot Non-linear ({method}) Combined Prediction Results:")
#     print(f"R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    
#     # Pour comparaison, calculer aussi avec ridge linéaire
#     YtY = Y_train.T @ Y_train
#     Yty = Y_train.T @ train_targets
#     A = YtY + lam * np.eye(K)
#     a_linear = np.linalg.solve(A, Yty)
#     y_pred_linear = Y_test @ a_linear
#     r2_linear = r2_score(test_targets, y_pred_linear)
#     print(f"\nComparison with Linear Ridge:")
#     print(f"Linear R²: {r2_linear:.4f} | Non-linear R²: {r2:.4f} | Improvement: {r2 - r2_linear:.4f}")

#     return y_pred, None  # Pas de poids pour méthodes non-linéaires

def plot_feature_attention(attn_matrix: torch.Tensor, feature_names=None, title="Attention Map"):
    """
    attn_matrix: [F, F] tensor
    feature_names: list of length F (optional)
    """
    #attn_np = attn_matrix.numpy()
    attn_np =  attn_matrix
    F = attn_np.shape[0]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(F)]

    plt.figure(figsize=(6, 5))
    plt.imshow(attn_np, interpolation="nearest")
    plt.colorbar()
    plt.xticks(np.arange(F), feature_names, rotation=90)
    plt.yticks(np.arange(F), feature_names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"./Results/attention_{title}.png")

class AttentionStore:
    def __init__(self):
        self.attn_per_layer = []

    def hook(self, module, input, output):
        """
        module: MultiheadAttention
        output: tuple (attn_output, attn_output_weights)
        attn_output_weights: [B, num_heads, F, F] when batch_first=True
        """
        attn_output, attn_weights = output
        self.attn_per_layer.append(attn_weights.detach().cpu())  # [B, H, F, F]

    def clear(self):
        self.attn_per_layer = []

def aggregate_attention(attn_store: AttentionStore):
    """
    Returns aggregated attention: [B, F, F] where B is the batch size
    Averages over layers and heads, keeps batch dimension
    """
    # attn_per_layer: list of [B, H, F, F] where B may vary per layer
    if len(attn_store.attn_per_layer) == 0:
        raise ValueError("No attention stored. Run a forward pass first.")

    # For each layer: average over heads to get [B, F, F]
    attn_per_layer_processed = []
    for attn_tensor in attn_store.attn_per_layer:
        # [B, H, F, F] -> [B, F, F] (average over heads)
        attn_mean_heads = attn_tensor.mean(dim=1)
        attn_per_layer_processed.append(attn_mean_heads)
    
    # Stack layers: [L, B, F, F] - but batch sizes may differ
    # Instead, average each layer separately then average over layers
    # Get batch size from first layer
    batch_size = attn_per_layer_processed[0].shape[0]
    
    # Average over layers: sum and divide by number of layers
    attn_sum = torch.zeros_like(attn_per_layer_processed[0])  # [B, F, F]
    for attn_layer in attn_per_layer_processed:
        # Handle different batch sizes by taking only the first batch_size samples
        if attn_layer.shape[0] >= batch_size:
            attn_sum += attn_layer[:batch_size]
        else:
            # If smaller, pad or repeat (shouldn't happen if store is cleared)
            attn_sum[:attn_layer.shape[0]] += attn_layer
    
    attn_mean = attn_sum / len(attn_per_layer_processed)  # [B, F, F]
    # attn_mean = attn_mean.mean(dim=0)
    return attn_mean  # [B, F, F]

def register_attention_hooks(feature_encoder: nn.Module):
    """
    feature_encoder: your FeatureWiseTransformerEncoder
                     with attribute .encoder (nn.TransformerEncoder)
    Returns: AttentionStore
    """
    store = AttentionStore()
    for layer in feature_encoder.encoder.layers:
        # layer.self_attn is nn.MultiheadAttention
        layer.self_attn.register_forward_hook(store.hook)
    return store

def compute_variable_weights_and_reweight_embedding(Ac, At, z_jepa, eps=1e-8):
 

    n_samples, n_vars = Ac.shape
    assert At.shape == Ac.shape, "Ac and At must have the same shape."

    # Compute mutual information for each variable i
    w = np.zeros(n_vars)
    for i in range(n_vars):
        # Convert continuous attentions → discretized bins for MI estimation
        Ac_col = np.digitize(Ac[:, i], np.histogram(Ac[:, i], bins=20)[1])
        At_col = np.digitize(At[:, i], np.histogram(At[:, i], bins=20)[1])
        w[i] = mutual_info_score(Ac_col, At_col)

    # Normalize weights to [0, 1]
    w_norm = (w - w.min()) / (w.max() - w.min() + eps)

    # Convert numpy arrays to torch tensors and move to same device as z_jepa
    device = z_jepa.device if hasattr(z_jepa, 'device') else 'cpu'
    w_norm_tensor = torch.from_numpy(w_norm).float().to(device)
    
    # Expand weights to embedding dimension if needed
    if z_jepa.ndim == 1:
        # Broadcasting if embedding dim = number of variables
        if len(z_jepa) == n_vars:
            z_weighted = w_norm_tensor * z_jepa
        else:
            # If embedding dimension != number of variables, project weights
            w_proj = np.interp(np.linspace(0,1,len(z_jepa)), 
                               np.linspace(0,1,len(w_norm)), 
                               w_norm)
            w_proj_tensor = torch.from_numpy(w_proj).float().to(device)
            z_weighted = w_proj_tensor * z_jepa
    else:
        # Batch mode (B, D)
        if z_jepa.shape[1] == n_vars:
            z_weighted = w_norm_tensor[None, :] * z_jepa
        else:
            # Project weights to embedding dimension
            w_proj = np.interp(np.linspace(0,1,z_jepa.shape[1]), 
                               np.linspace(0,1,len(w_norm)), 
                               w_norm)
            w_proj_tensor = torch.from_numpy(w_proj).float().to(device)
            z_weighted = w_proj_tensor[None, :] * z_jepa

    return w_norm, z_weighted

def attention_features(Ac, At):
    """
    Ac, At: attention matrices of shape (B, F, F) - torch tensors
    Returns:
      meanAtt: shape (B, F) - mean attention per variable for each sample
    """
    # Convert to numpy if torch tensors
    if isinstance(Ac, torch.Tensor):
        Ac_np = Ac.detach().cpu().numpy()
    else:
        Ac_np = Ac
    
    if isinstance(At, torch.Tensor):
        At_np = At.detach().cpu().numpy()
    else:
        At_np = At
    
    # Ac, At: [B, F, F]
    # Mean attention per variable (row-wise mean): [B, F]
    meanAtt = Ac_np.mean(axis=2)  # [B, F]
    
    # Convert back to torch tensor
    meanAtt = torch.from_numpy(meanAtt).float()

    align_list = []
    for batch_idx in range(Ac.shape[0]):
            a = Ac[batch_idx]
            b = At[batch_idx]
            sim = (a*b).sum(axis=1)
            norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            align= sim / (norm + 1e-8)
            align_list.append(align)
    align =np.stack(align_list, axis=0)
    align = align.mean(axis=1).reshape(-1, 1)
    align = torch.from_numpy(align).float()
    
    return meanAtt, align  # [B, F]

class ResidualMLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_properties, dropout=0.5, device="cuda:0"):
        super().__init__()
        self.device = device
        self.input_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, num_properties)

    def forward(self, x):
        x = self.input_norm(x)
        h = self.act(self.fc1(x))
        h = self.drop(h)
        h2 = self.act(self.fc2(h))
        h2 = self.drop(h2)
        h = h + h2                      # residual
        return self.out(h)

# class FeatureGraphPredictor(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.1, device="cuda:0"):
#         """
#         Feature Graph Predictor with configurable number of GNN layers.
        
#         Args:
#             A_prior: Prior adjacency matrix [F, F] - fixed graph structure
#             in_dim: Number of features (F)
#             hidden_dim: Hidden dimension for GNN layers
#             out_dim: Output dimension
#             num_layers: Number of GNN layers (default: 3)
#             dropout: Dropout rate (default: 0.1)
#             device: Device to place model on
#             use_attention_adjacency: If True, use per-batch attention matrix A; 
#                                      If False, use fixed A_prior only
#         """
#         super().__init__()
#         self.num_layers = num_layers
#         self.in_dim = in_dim  # Store in_dim for use in forward
#         self.device = device
        
#         # First layer: accepts projected_dim (128) from context encoder embeddings
#         # Context encoder outputs [B, F * projected_dim] which we reshape to [B, F, projected_dim]
#         projected_dim = 128  # From context encoder
#         self.gnn_layers = nn.ModuleList()
#         self.gnn_layers.append(nn.Linear(projected_dim, hidden_dim).to(device))
        
#         # Intermediate layers: hidden_dim -> hidden_dim
#         for _ in range(num_layers - 1):
#             self.gnn_layers.append(nn.Linear(hidden_dim, hidden_dim).to(device))
        
#         # Dropout layers
#         self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        
#         # Output layer
#         self.out = nn.Linear(hidden_dim, out_dim).to(device)

#     def forward(self, X, A=None):
#         """
#         Forward pass.
        
#         Args:
#             X: [B, F] or [B, F, D] - input features
#             A: [B, F, F] or [F, F] - adjacency matrix (attention). 
#                If None and use_attention_adjacency=False, uses A_prior.
#                If provided, can be per-batch attention (dynamic) or fixed.
        
#         Returns:
#             [B, out_dim] - predictions
#         """
#         # X should be [B, F, projected_dim] from context encoder embeddings
#         if X.dim() == 2:
#             # If 2D, assume it's [B, F * projected_dim] and reshape
#             B = X.size(0)
#             # We know projected_dim = 128 from context encoder and in_dim = 7
#             projected_dim = 128
#             F = self.in_dim  # Number of features (7)
#             X = X.view(B, F, projected_dim)  # [B, F, projected_dim]
        
#         B, F, D = X.shape
        
#         # Handle adjacency matrix A
#         if A is None:
#             # Use fixed A_prior
#             raise ValueError("A_prior is not provided. Please provide A_prior.")
#         else:
#             # Use provided A (can be per-batch attention or fixed)
#             if A.dim() == 2:
#                 A = A.unsqueeze(0).expand(B, -1, -1)  # [F, F] -> [B, F, F]
#             elif A.dim() == 3:
#                 if A.size(0) != B:
#                     if A.size(0) == 1:
#                         A = A.expand(B, -1, -1)
#                     else:
#                         A = A[:B]
            
#             # Normalize adjacency matrix for graph convolution (row normalization)
#             # Add small epsilon to avoid division by zero
#             row_sums = A.sum(dim=-1, keepdim=True) + 1e-8  # [B, F, 1]
#             A = A / row_sums  # Row-normalized adjacency matrix
        
#         # Apply GNN layers
#         for i, (gnn_layer, dropout_layer) in enumerate(zip(self.gnn_layers, self.dropout_layers)):
#             # Graph convolution: [B, F, F] @ [B, F, D] -> [B, F, D]
#             X = torch.matmul(A, X)  # [B, F, F] @ [B, F, D] -> [B, F, D]
            
#             # Apply GNN layer: [B, F, D] -> [B, F, hidden_dim]
#             B, F, D = X.shape
#             X_flat = X.view(B * F, D)  # [B*F, D]
#             X_flat = gnn_layer(X_flat)  # [B*F, hidden_dim]
#             X_flat = torch.relu(X_flat)
#             X_flat = dropout_layer(X_flat)
#             X = X_flat.view(B, F, -1)  # [B, F, hidden_dim]
        
#         # Pool over features: [B, F, hidden_dim] -> [B, hidden_dim]
#         X = X.mean(dim=1)  # [B, hidden_dim]
        
#         # Final output: [B, hidden_dim] -> [B, out_dim]
#         return self.out(X)  # [B, out_dim]



class FeatureGraphPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.1, device="cuda:0"):
        
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim  # Store in_dim for use in forward
        self.device = device
        
        # First layer: accepts dimension 1 (raw features after unsqueeze)
        # Input is [B, F] -> unsqueeze -> [B, F, 1]
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(nn.Linear(1, hidden_dim).to(device))
        
        # Intermediate layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.gnn_layers.append(nn.Linear(hidden_dim, hidden_dim).to(device))
        
        # Dropout layers
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        
        self.prop_head = nn.Sequential(
            nn.Linear(7*hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        ).to(device)
        # # Output layer
        # self.out = nn.Linear(hidden_dim, out_dim).to(device)

    def forward(self, X, A=None):
       
        # X should be [B, F, projected_dim] from context encoder embeddings
        if X.dim() == 2:
            X = X.unsqueeze(-1)
            # # If 2D, assume it's [B, F * projected_dim] and reshape
            # B = X.size(0)
            # # We know projected_dim = 128 from context encoder and in_dim = 7
            # projected_dim = 128
            # F = self.in_dim  # Number of features (7)
            # X = X.view(B, F, projected_dim)  # [B, F, projected_dim]
        
        B, F, D = X.shape
        
        # Handle adjacency matrix A
        if A is None:
            raise ValueError("A_prior is not provided. Please provide A_prior.")
        else:
            # Use provided A (can be per-batch attention or fixed)
            if A.dim() == 2:
                A = A.unsqueeze(0).expand(B, -1, -1)  # [F, F] -> [B, F, F]
            elif A.dim() == 3:
                if A.size(0) != B:
                    if A.size(0) == 1:
                        A = A.expand(B, -1, -1)
                    else:
                        A = A[:B]
            
            # Normalize adjacency matrix for graph convolution (row normalization)
            # Add small epsilon to avoid division by zero
            row_sums = A.sum(dim=-1, keepdim=True) + 1e-8  # [B, F, 1]
            A = A / row_sums  # Row-normalized adjacency matrix
        
        # Apply GNN layers
        for i, (gnn_layer, dropout_layer) in enumerate(zip(self.gnn_layers, self.dropout_layers)):
            # Graph convolution: [B, F, F] @ [B, F, D] -> [B, F, D]
            X = torch.matmul(A, X)  # [B, F, F] @ [B, F, D] -> [B, F, D]
            
            # Apply GNN layer: [B, F, D] -> [B, F, hidden_dim]
            B, F, D = X.shape
            X_flat = X.view(B * F, D)  # [B*F, D]
            X_flat = gnn_layer(X_flat)  # [B*F, hidden_dim]
            X_flat = torch.relu(X_flat)
            X_flat = dropout_layer(X_flat)
            X = X_flat.view(B, F, -1)  # [B, F, hidden_dim]
        X = X.reshape(B, X.shape[1]*X.shape[2])
        X = self.prop_head(X)
        # Pool over features: [B, F, hidden_dim] -> [B, hidden_dim]
        #X = X.mean(dim=1)  # [B, hidden_dim]
        
        # Final output: [B, hidden_dim] -> [B, out_dim]
        return X  # [B, out_dim]

        
def topk_sparsify(A, k=3):
    # A: [B,F,F]
    vals, idx = torch.topk(A, k=k, dim=-1)
    A_sparse = torch.zeros_like(A)
    A_sparse.scatter_(-1, idx, vals)
    return A_sparse

def clr_transform(x, eps=1e-6):
   
    x = x.clamp_min(eps)
    x = x / x.sum(dim=-1, keepdim=True)
    log_x = torch.log(x)
    return log_x - log_x.mean(dim=-1, keepdim=True)
    

def get_properties_mean_std(data ,device):
    properties_list = []
    targets_list = []
    for b in data:
        properties_list.append(b['properties'].float())
        targets_list.append(b['target'].float())
    properties_list = np.concatenate(properties_list)
    targets_list = np.concatenate(targets_list)
    properties_mean = torch.tensor(properties_list.mean(axis=0), dtype=torch.float32)  # ✅ [7] - mean per column
    properties_std = torch.tensor(properties_list.std(axis=0), dtype=torch.float32)    # ✅ [7] - std per column
    properties_std = torch.clamp(properties_std, min=1e-8)
    properties_mean = properties_mean.to(device)
    properties_std = properties_std.to(device)
    targets_mean = targets_list.mean()
    targets_std = max(targets_list.std(), 1e-8)  # Clamp to avoid division by zero
    return properties_mean, properties_std, targets_mean, targets_std

def get_loaders(batch_size=128, QH_flag=False, QH_random_1_flag=False):
    if QH_flag:
        csv_file = "/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_QH.csv"
        image_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure_images_QH" 
        # image_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure"
        text_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports_QH"
        # text_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/text"
        target_name = 'elongation' #, 'yield', 'elongation'
        full_dataset = CompositeImageTextDataset(
            csv_file, image_folder, text_folder, target_name,
            num_augmentations=5, mode='QH_random',
            add_gaussian_noise=False, noise_std=0.01
        )
        full_train_dataset, test_dataset = split_data(full_dataset, train_ratio=0.80, random_seed=1234)
        train_dataset, val_dataset = split_data(full_train_dataset, train_ratio=0.80, random_seed=1234)
        generator = torch.Generator().manual_seed(1234)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
        return train_dataloader, val_dataloader, test_dataloader
    elif QH_random_1_flag:
        csv_file = "/home/abhibhatt/Pycharm_Projects/Data_composites/Composite_Data_QH1.csv"
        image_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure"
        text_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/text"
        target_name = 'Young_modulus' #, Young's modulus (MPa)	Bulk modulus (MPa)

        full_dataset = CompositeImageTextDataset(
            csv_file, image_folder, text_folder, target_name,
            num_augmentations=5, mode='QH_random_1',
            add_gaussian_noise=False, noise_std=0.01
        )
        full_train_dataset, test_dataset = split_data(full_dataset, train_ratio=0.80, random_seed=1234)
        train_dataset, val_dataset = split_data(full_train_dataset, train_ratio=0.80, random_seed=1234)
        print(f"train_dataset: {len(train_dataset)}, val_dataset: {len(val_dataset)}, test_dataset: {len(test_dataset)}")
        generator = torch.Generator().manual_seed(1234)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
        return train_dataloader, val_dataloader, test_dataloader
    else:
        csv_file_train = "/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_train.csv"
        csv_file_test = "/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_test.csv"
        image_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure_images"  
        text_folder_train = "/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports"
        text_folder_test = "/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports_test"
        target_name = 'elongation' #, 'yield', 'elastic modulus', 'elongation', 'tangent modulus'
        full_train_dataset = CompositeImageTextDataset(
            csv_file_train, image_folder, text_folder_train, target_name,
            num_augmentations=5, mode='random',
            add_gaussian_noise=False, noise_std=0.01
        )
        test_dataset = CompositeImageTextDataset(
        csv_file_test, image_folder, text_folder_test, target_name,
        num_augmentations=5, mode='random',
        add_gaussian_noise=False, noise_std=0.01
        )
        train_dataset, val_dataset = split_data(full_train_dataset, train_ratio=0.80, random_seed=1234)
        generator = torch.Generator().manual_seed(1234)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
        return train_dataloader, val_dataloader, test_dataloader

def split_data_by_conditions(dataset, test_conditions, radius_col='r', test_filter_type='equal'):   
    """
    Split dataset into train and test based on specific conditions.
    
    Args:
        dataset: CompositeImageTextDataset instance
        test_conditions: List of values for filtering (used with 'equal' type)
        radius_col: Column name for radius filtering
        test_filter_type: 'quantile' or 'equal'
            - 'quantile': Selects samples in extreme quantiles (upper 10% or lower 10%) for both 't' and 'w'
            - 'equal': Selects samples where radius_col matches values in test_conditions
    
    Returns:
        train_dataset, test_dataset: PyTorch Subset datasets
    """
    from torch.utils.data import Subset        
   
    df = dataset.data        
    available_cols = df.columns.tolist()    
    print(f"Available columns: {available_cols}")      
    
    # Initialize the mask
    test_mask = pd.Series([False] * len(df))
    
    if test_filter_type == 'quantile':
        upper_test_quantile = 0.85
        lower_test_quantile = 0.20
        upper_train_quantile = 0.8
        lower_train_quantile = 0.2
        
        # Calculate quantiles once
        t_upper = df['t'].quantile(upper_test_quantile)
        t_lower = df['t'].quantile(lower_test_quantile)
        w_upper = df['w'].quantile(upper_test_quantile)
        w_lower = df['w'].quantile(lower_test_quantile)

        t_train_upper = df['t'].quantile(upper_train_quantile)
        w_train_upper = df['w'].quantile(upper_train_quantile)
        t_train_lower = df['t'].quantile(lower_train_quantile)
        w_train_lower = df['w'].quantile(lower_train_quantile)
        
        # Select values in extreme tails of distribution (upper OR lower)
        # Use parentheses to ensure correct operator precedence
        t_extreme = (df['t'] > t_upper) 
        w_extreme = (df['w'] > w_upper)
        
        # Combine: samples with both t AND w in extreme tails
        test_mask = t_extreme
        
        t_id = (df['t'] > t_train_lower) & (df['t'] < t_train_upper)
        w_id = (df['w'] > w_train_lower) & (df['w'] < w_train_upper)

        train_mask = t_id 

        print(f"t quantiles: lower={t_lower:.4f}, upper={t_upper:.4f}")
        print(f"w quantiles: lower={w_lower:.4f}, upper={w_upper:.4f}")
        print(f"Test samples (extreme quantiles): {test_mask.sum()}")
        
    elif test_filter_type == 'equal':
        # Filter by exact values of radius
        for radius_val in test_conditions:
            condition_mask = (df[radius_col] == radius_val)
            test_mask = test_mask | condition_mask
            print(f"Found {condition_mask.sum()} samples for condition (radius={radius_val})")
    
    else:
        raise ValueError(f"Invalid test filter type: {test_filter_type}")
    
    # Convert mask to indices    
    test_indices = df[test_mask].index.tolist()  
    train_indices = df[~test_mask].index.tolist()       
    print(f"Total test samples: {len(test_indices)}")    
    print(f"Total train samples: {len(train_indices)}")
    # assert len(test_indices)> 100 and len(train_indices)> 100 , "Not enough samples for training and testing"	
    # Create subsets    
    train_dataset = Subset(dataset, train_indices)    
    test_dataset = Subset(dataset, test_indices)        
    
    return train_dataset, test_dataset

def generate_ood_data(batch_size=128, target_name='tangent modulus'):
    master_CSV = "/home/abhibhatt/Pycharm_Projects/Data_composites/OOD/composite_data_ood.csv"
    image_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure_images"  
    text_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/OOD/text_reports" #, 'yield', 'elastic modulus', 'elongation', 'tangent modulus'
    master_dataset = CompositeImageTextDataset(
        master_CSV, image_folder, text_folder, target_name,
        num_augmentations=10, mode='random',
        add_gaussian_noise=True, noise_std=0.01)

    # test_conditions = [(0.2, 16, 16,0),(0.2, 16, 16,1600),
    # (0.2, 16, 22,0),(0.2, 16, 22,1600),
    # (0.2, 22, 16,0),(0.2, 22, 16,1600),
    # (0.2, 22, 22,0),(0.2, 22, 22,1600),
    # (0.6, 16, 16,0),(0.6, 16, 16,1600),
    # (0.6, 16, 22,0),(0.6, 16, 22,1600),
    # (0.6, 22, 16,0),(0.6, 22, 16,1600),
    # (0.6, 22, 22,0),(0.6, 22, 22,1600)]
    test_conditions =[1200,1600]
    test_filter_type = 'equal'
    full_train_dataset, test_dataset = split_data_by_conditions(master_dataset, test_conditions, radius_col ='r', test_filter_type=test_filter_type)
    train_dataset, val_dataset = split_data(full_train_dataset, train_ratio=0.80, random_seed=1234)
    print(f"train_dataset: {len(train_dataset)}, val_dataset: {len(val_dataset)}, test_dataset: {len(test_dataset)}")
    generator = torch.Generator().manual_seed(1234)
    train_dataloader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
    return train_dataloader, val_dataloader, test_dataloader

def split_qh_data_by_conditions(dataset, test_filter_type='quantile'):   
    
    from torch.utils.data import Subset        
   
    df = dataset.data        
    available_cols = df.columns.tolist()    
    print(f"Available columns: {available_cols}")      
    
    # Initialize the mask
    test_mask = pd.Series([False] * len(df))
    train_mask = pd.Series([False] * len(df))
    
    if test_filter_type == 'quantile':
        upper_test_quantile = 0.85
        lower_test_quantile = 0.15
        train_upper_quantile = 0.6
        train_lower_quantile = 0.4

        var1_train_upper = df['NumFibers'].quantile(train_upper_quantile)
        var1_train_lower = df['NumFibers'].quantile(train_lower_quantile)
        
        # # Calculate quantiles once
        # var1_upper = df['NumFibers'].quantile(upper_test_quantile)
        # var1_lower = df['NumFibers'].quantile(lower_test_quantile)
        # var2_upper = df['NumFibers'].quantile(upper_test_quantile)
        # var2_lower = df['NumFibers'].quantile(lower_test_quantile)

    
        # Select values in extreme tails of distribution (upper OR lower)
        # # Use parentheses to ensure correct operator precedence
        var1_extreme = (df['NumFibers'] > var1_upper) | (df['NumFibers'] < var1_lower)
     
        
        # Combine: samples with both vf in extreme tails
        test_mask = var1_extreme 
        train_mask = (df['NumFibers'] > var1_train_lower) & (df['NumFibers'] < var1_train_upper) 
    

        # print(f"Variable1 quantiles: lower={var1_lower:.4f}, upper={var1_upper:.4f}")
        # print(f"Variable2 quantiles: lower={var2_lower:.4f}, upper={var2_upper:.4f}")
        print(f"Test samples (extreme quantiles): {test_mask.sum()}")
        
    else:
        raise ValueError(f"Invalid test filter type: {test_filter_type}")
    
    # Convert mask to indices    
    test_indices = df[test_mask].index.tolist()  
    train_indices = df[train_mask].index.tolist()       
    print(f"Total test samples: {len(test_indices)}")    
    print(f"Total train samples: {len(train_indices)}")
    # assert len(test_indices)> 100 and len(train_indices)> 100 , "Not enough samples for training and testing"	
    # Create subsets    
    train_dataset = Subset(dataset, train_indices)    
    test_dataset = Subset(dataset, test_indices)        
    
    return train_dataset, test_dataset

def generate_ood_qh_data(batch_size=128, target_name='elongation'):
    master_CSV = "/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_QH.csv"
    image_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure_images_QH"  
    text_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports_QH" #, 'yield', 'elastic modulus', 'elongation', 'tangent modulus'
    master_dataset = CompositeImageTextDataset(
        master_CSV, image_folder, text_folder, target_name,
        num_augmentations=5, mode='QH_random',
        add_gaussian_noise=False, noise_std=0.01)

    test_filter_type = 'quantile'
    full_train_dataset, test_dataset = split_qh_data_by_conditions(master_dataset, test_filter_type=test_filter_type)
    train_dataset, val_dataset = split_data(full_train_dataset, train_ratio=0.80, random_seed=1234)
    print(f"train_dataset: {len(train_dataset)}, val_dataset: {len(val_dataset)}, test_dataset: {len(test_dataset)}")
    generator = torch.Generator().manual_seed(1234)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
    return train_dataloader, val_dataloader, test_dataloader


def split_qh1_data_by_conditions(dataset, test_filter_type='quantile'):   
    
    from torch.utils.data import Subset        
   
    df = dataset.data        
    available_cols = df.columns.tolist()    
    print(f"Available columns: {available_cols}")      
    
    # Initialize the mask
    test_mask = pd.Series([False] * len(df))
    train_mask = pd.Series([False] * len(df))
   
    if test_filter_type == 'quantile':
        upper_test_quantile = 0.9
        lower_test_quantile = 0.1

        train_upper_quantile = 0.7
        train_lower_quantile = 0.3

        var1_train_upper = df['f_gra'].quantile(train_upper_quantile)
        var1_train_lower = df['f_gra'].quantile(train_lower_quantile)

        # # Calculate quantiles once
        var1_upper = df['f_gra'].quantile(upper_test_quantile)
        var1_lower = df['f_gra'].quantile(lower_test_quantile)
        var1_extreme = (df['f_gra'] > var1_upper)|(df['f_gra'] < var1_lower) #
        test_mask = var1_extreme 
        train_mask = (df['f_gra'] > var1_train_lower) & (df['f_gra'] < var1_train_upper)
        print(f"Test samples (extreme quantiles): {test_mask.sum()}")
    else:
        raise ValueError(f"Invalid test filter type: {test_filter_type}")
    
    # Convert mask to indices    
    test_indices = df[test_mask].index.tolist()  
    train_indices = df[train_mask].index.tolist()    
    print(f"Total test samples: {len(test_indices)}")    
    print(f"Total train samples: {len(train_indices)}")
    # assert len(test_indices)> 100 and len(train_indices)> 100 , "Not enough samples for training and testing"	
    # Create subsets    
    train_dataset = Subset(dataset, train_indices)    
    test_dataset = Subset(dataset, test_indices)        
    return train_dataset, test_dataset

def generate_ood_qh1_data(batch_size=128, target_name='elongation'):
    master_CSV = "/home/abhibhatt/Pycharm_Projects/Data_composites/Comp1_data.csv"
    image_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure"  
    text_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/text" #, 'yield', 'elastic modulus', 'elongation', 'tangent modulus'
    master_dataset = CompositeImageTextDataset(
        master_CSV, image_folder, text_folder, target_name,
        num_augmentations=5, mode='Comp1',
        add_gaussian_noise=False, noise_std=0.01)

    test_filter_type = 'quantile'
    full_train_dataset, test_dataset = split_qh1_data_by_conditions(master_dataset, test_filter_type=test_filter_type)
    train_dataset, val_dataset = split_data(full_train_dataset, train_ratio=0.80, random_seed=1234)
    print(f"train_dataset: {len(train_dataset)}, val_dataset: {len(val_dataset)}, test_dataset: {len(test_dataset)}")
    generator = torch.Generator().manual_seed(1234)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
    return train_dataloader, val_dataloader, test_dataloader


def split_qh2_data_by_conditions(dataset, feat, test_filter_type='quantile'):   
    
    from torch.utils.data import Subset        
   
    df = dataset.data        
    available_cols = df.columns.tolist()    
    print(f"Available columns: {available_cols}")      
    
    # Initialize the mask
    test_mask = pd.Series([False] * len(df))
    train_mask = pd.Series([False] * len(df))
    
    if test_filter_type == 'quantile':

        test_mask = (df[feat] == 0.0)
        train_mask = (df[feat] != 0.0)
        # print(f"Vf quantiles: lower={var1_lower:.4f}, upper={var1_upper:.4f}")
        # print(f"Vf quantiles: lower={var2_lower:.4f}, upper={var2_upper:.4f}")
        print(f"Test samples (extreme quantiles): {test_mask.sum()}")
        
    else:
        raise ValueError(f"Invalid test filter type: {test_filter_type}")
    
    # Convert mask to indices    
    test_indices = df[test_mask].index.tolist()  
    train_indices = df[train_mask].index.tolist()    
    print(f"Total test samples: {len(test_indices)}")    
    print(f"Total train samples: {len(train_indices)}")
    assert len(test_indices)> 100 and len(train_indices)> 100 , "Not enough samples for training and testing"	
    # Create subsets    
    train_dataset = Subset(dataset, train_indices)    
    test_dataset = Subset(dataset, test_indices)        
    return train_dataset, test_dataset

def generate_ood_qh2_data(batch_size=128, target_name='elongation'):
    master_CSV = "/home/abhibhatt/Pycharm_Projects/Data_composites/DS2.csv"
    image_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure"  
    text_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/text" #, 'yield', 'elastic modulus', 'elongation', 'tangent modulus'
    master_dataset = CompositeImageTextDataset(
        master_CSV, image_folder, text_folder, target_name,
        num_augmentations=5, mode='Comp2',
        add_gaussian_noise=False, noise_std=0.01)

    test_filter_type = 'quantile'
    full_train_dataset, test_dataset = split_qh2_data_by_conditions(master_dataset, feat="f1_T800", test_filter_type=test_filter_type)
    train_dataset, val_dataset = split_data(full_train_dataset, train_ratio=0.80, random_seed=1234)
    print(f"train_dataset: {len(train_dataset)}, val_dataset: {len(val_dataset)}, test_dataset: {len(test_dataset)}")
    generator = torch.Generator().manual_seed(1234)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
    return train_dataloader, val_dataloader, test_dataloader
    # return full_train_dataset, test_dataset







def plot_histogram(ID_data, OOD_data, target_name='elongation'):
    plt.hist(ID_data, bins=50, alpha=0.5, label='ID')
    plt.hist(OOD_data, bins=50, alpha=0.5, label='OOD')
    plt.legend()
    plt.savefig(f'./Results/histogram_thickness_{target_name}.png')
    plt.close()

# def compare_best_d_histograms(best_d_id, best_d_ood, bins=50, title="best_d: ID vs OOD", log_scale=False):
#     """
#     Plot overlapping histograms for best_d (distance to closest anchor) for ID and OOD.
#     best_d_id / best_d_ood can contain inf/nan; we filter to finite.
#     """
#     best_d_id = np.asarray(best_d_id, dtype=np.float64)
#     best_d_ood = np.asarray(best_d_ood, dtype=np.float64)

#     id_vals = best_d_id[np.isfinite(best_d_id)]
#     ood_vals = best_d_ood[np.isfinite(best_d_ood)]

#     if len(id_vals) == 0 and len(ood_vals) == 0:
#         raise ValueError("No finite values in either best_d_id or best_d_ood.")

#     all_vals = np.concatenate([id_vals, ood_vals]) if (len(id_vals) and len(ood_vals)) else (id_vals if len(id_vals) else ood_vals)

#     # Shared binning for fair comparison
#     if log_scale:
#         eps = 1e-12
#         minv = max(all_vals.min(), eps)
#         maxv = all_vals.max()
#         edges = np.logspace(np.log10(minv), np.log10(maxv), bins + 1)
#     else:
#         edges = np.histogram_bin_edges(all_vals, bins=bins)
#     q0, q25, q50, q75, q95, q100 = np.quantile(finite, [0, 0.25, 0.5, 0.75, 0.95, 1.0])
#     plt.figure(figsize=(7, 5))
#     plt.hist(id_vals, bins=edges, alpha=0.5, density=True, label=f"ID (n={len(id_vals)})")
#     plt.hist(ood_vals, bins=edges, alpha=0.5, density=True, label=f"OOD (n={len(ood_vals)})")
#     plt.title(title)
#     plt.xlabel("best_d (geodesic distance to closest anchor)")
#     plt.ylabel("Density")
#     plt.legend()
#     if log_scale:
#         plt.xscale("log")
#     plt.grid(True, linestyle="--", linewidth=0.5)
#     plt.savefig(f'./Results/{title}.png')
#     plt.close()

def summarize_best_d(best_d):
    """Quick stats (finite only) to print alongside the plots."""
    best_d = np.asarray(best_d, dtype=np.float64)
    finite = best_d[np.isfinite(best_d)]
    if len(finite) == 0:
        return {"n_total": int(len(best_d)), "n_finite": 0}
    q0, q25, q50, q75, q95, q100 = np.quantile(finite, [0, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "n_total": int(len(best_d)),
        "n_finite": int(len(finite)),
        "min": float(q0),
        "p25": float(q25),
        "median": float(q50),
        "p75": float(q75),
        "p95": float(q95),
        "max": float(q100),
        "mean": float(finite.mean()),
    }

def compare_best_d_histograms(
    best_d_id,
    best_d_ood,
    bins=50,
    title="best_d: ID vs OOD",
    log_scale=False,
    percentiles=( 25, 75),
    show_percentile_labels=True,
    percentile_label_loc="upper right",
    save_dir="./Results",
):
    """
    Plot overlapping histograms for best_d (distance to closest anchor) for ID and OOD,
    and overlay percentile distributions as vertical lines.

    best_d_id / best_d_ood can contain inf/nan; we filter to finite.
    """
    best_d_id = np.asarray(best_d_id, dtype=np.float64)
    best_d_ood = np.asarray(best_d_ood, dtype=np.float64)

    id_vals = best_d_id[np.isfinite(best_d_id)]
    ood_vals = best_d_ood[np.isfinite(best_d_ood)]

    if len(id_vals) == 0 and len(ood_vals) == 0:
        raise ValueError("No finite values in either best_d_id or best_d_ood.")

    all_vals = (
        np.concatenate([id_vals, ood_vals])
        if (len(id_vals) and len(ood_vals))
        else (id_vals if len(id_vals) else ood_vals)
    )

    # Shared binning for fair comparison
    if log_scale:
        eps = 1e-12
        minv = max(all_vals.min(), eps)
        maxv = all_vals.max()
        edges = np.logspace(np.log10(minv), np.log10(maxv), bins + 1)
    else:
        edges = np.histogram_bin_edges(all_vals, bins=bins)

    # Percentiles (computed separately for ID and OOD)
    p = np.asarray(percentiles, dtype=float) / 100.0
    id_q = np.quantile(id_vals, p) if len(id_vals) else None
    ood_q = np.quantile(ood_vals, p) if len(ood_vals) else None

    plt.figure(figsize=(7, 5))
    plt.hist(id_vals, bins=edges, alpha=0.5, density=True, label=f"ID (n={len(id_vals)})")
    plt.hist(ood_vals, bins=edges, alpha=0.5, density=True, label=f"OOD (n={len(ood_vals)})")

    # Draw percentile lines
    # Use different linestyles for ID vs OOD to keep colors unchanged.
    # (Matplotlib will auto-assign colors for vlines if we don't specify.)
    if id_q is not None:
        for x in id_q:
            plt.axvline(x, linestyle="--", linewidth=1, alpha=0.9)
    if ood_q is not None:
        for x in ood_q:
            plt.axvline(x, linestyle=":", linewidth=1, alpha=0.9)

    # Optional: add a compact label box with values
    if show_percentile_labels:
        lines = []
        if id_q is not None:
            id_txt = ", ".join([f"P{int(pp)}={vv:.4g}" for pp, vv in zip(percentiles, id_q)])
            lines.append("ID  (--)  " + id_txt)
        if ood_q is not None:
            ood_txt = ", ".join([f"P{int(pp)}={vv:.4g}" for pp, vv in zip(percentiles, ood_q)])
            lines.append("OOD (:)   " + ood_txt)

        plt.gca().text(
            0.99 if "right" in percentile_label_loc else 0.01,
            0.99 if "upper" in percentile_label_loc else 0.01,
            "\n".join(lines),
            transform=plt.gca().transAxes,
            ha="right" if "right" in percentile_label_loc else "left",
            va="top" if "upper" in percentile_label_loc else "bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, linewidth=0.5),
        )

    plt.title(title)
    plt.xlabel("best_d (geodesic distance to closest anchor)")
    plt.ylabel("Density")
    plt.legend(loc='upper left')

    if log_scale:
        plt.xscale("log")

    plt.grid(True, linestyle="--", linewidth=0.5)

    # Save
    import os
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{title}.png"), dpi=300, bbox_inches="tight")
    plt.close()