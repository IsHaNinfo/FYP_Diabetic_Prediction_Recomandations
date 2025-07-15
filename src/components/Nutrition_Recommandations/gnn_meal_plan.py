# NRKG - Diabetes Friendly Food Recommender (UPDATED)
import os, random, warnings
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, Linear
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ================= CONFIG ======================
BASE_DIR = 'notebook/data/RecommandationDatasets/NutritionDatasets'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
HIDDEN_DIM = 64
N_HEADS = 4
SIM_THR = 0.95
N_EPOCHS = 25
BATCH_SIZE = 1024
LR = 1e-4
INTERACT_REL = 'interacts'

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Features
USER_NUTR_COLS = ['Carbohydrate_Consumption', 'Protein_Intake', 'Fat_Intake', 'Caloric_Balance', 'Sugar_Consumption']
FOOD_NUTR_COLS = ['calories', 'carbs', 'protein', 'fat', 'glycemic_index']

# ================= DATA LOADING ===============
print("Loading CSVs …")
user_df = pd.read_csv(os.path.join(BASE_DIR, 'Updated_User_Nutrition_Parameters.csv'))
food_df = pd.read_csv(os.path.join(BASE_DIR, 'Foods_Datasets.csv'))
nutrient_df = pd.read_csv(os.path.join(BASE_DIR, 'nutrients.csv'))
disease_df = pd.read_csv(os.path.join(BASE_DIR, 'diseases.csv'))
edge_df = pd.read_csv(os.path.join(BASE_DIR, 'Updated_Edges_Dataset.csv'))

for df in [user_df, food_df, edge_df]:
    for col in ['user_id', 'food_id', 'source', 'target']:
        if col in df.columns:
            df[col] = df[col].astype(str)

# User preference vectors
liked = food_df.merge(edge_df[edge_df['relation'] == INTERACT_REL], left_on='food_id', right_on='target')
pref = liked.groupby('source')[FOOD_NUTR_COLS].mean().rename_axis('user_id').reset_index()
user_df = user_df.merge(pref, how='left', on='user_id')
user_df[USER_NUTR_COLS] = user_df[USER_NUTR_COLS].fillna(user_df[USER_NUTR_COLS].mean())

# ================= GRAPH ======================
scaler = StandardScaler()
nutr_mat = scaler.fit_transform(food_df[FOOD_NUTR_COLS])
sim = cosine_similarity(nutr_mat)
src, dst = np.where((sim > SIM_THR) & (sim < 0.999))
food_sim_edges = torch.from_numpy(np.vstack((src, dst))).long()

data = HeteroData()
uid_map = {u: i for i, u in enumerate(user_df['user_id'])}
fid_map = {f: i for i, f in enumerate(food_df['food_id'])}
nid_map = {n: i for i, n in enumerate(nutrient_df['nutrient_id'])}
did_map = {d: i for i, d in enumerate(disease_df['disease_id'])}

user_features = ['Age','Gender','Height','Weight','BMI','DiabetesRisk','NutritionRisk'] + USER_NUTR_COLS
data['user'].x = torch.tensor(user_df[user_features].values, dtype=torch.float)
data['food'].x = torch.tensor(food_df[FOOD_NUTR_COLS].values, dtype=torch.float)
data['nutrient'].x = torch.randn(len(nutrient_df), HIDDEN_DIM)
data['disease'].x = torch.randn(len(disease_df), HIDDEN_DIM)

# Edges
print("Populating KG edges …")
edge_lists = defaultdict(list)
for _, row in edge_df.iterrows():
    s, rel, t = row['source'], row['relation'], row['target']
    if rel == INTERACT_REL and s in uid_map and t in fid_map:
        edge_lists[('user','interacts','food')].append([uid_map[s], fid_map[t]])
    elif rel == 'contains' and s in fid_map and t in nid_map:
        edge_lists[('food','contains','nutrient')].append([fid_map[s], nid_map[t]])
    elif rel == 'hasRisk' and s in uid_map and t in did_map:
        edge_lists[('user','hasRisk','disease')].append([uid_map[s], did_map[t]])

for e_type, lst in edge_lists.items():
    data[e_type].edge_index = torch.tensor(np.array(lst).T, dtype=torch.long)

data['food','similar','food'].edge_index = food_sim_edges
data['food','rev_similar','food'].edge_index = food_sim_edges.flip(0)
for (src, rel, dst) in list(data.edge_types):
    rev = 'rev_' + rel
    if (dst, rev, src) not in data.edge_types:
        ei = data[(src, rel, dst)].edge_index
        data[(dst, rev, src)].edge_index = ei.flip(0)

# ================= MODEL ======================
class NutrientAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.proj = nn.Linear(len(FOOD_NUTR_COLS), d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, food_x, nutrient_emb):
        nutrient_emb = F.relu(self.proj(nutrient_emb))
        q = food_x.unsqueeze(1)
        k = v = nutrient_emb.unsqueeze(1)
        attn_out, _ = self.attn(q, k, v)
        return attn_out.squeeze(1)

class DietGNN(nn.Module):
    def __init__(self, hidden_dim, heads):
        super().__init__()
        self.food_enc = Linear(data['food'].num_features, hidden_dim)
        self.user_enc = Linear(data['user'].num_features, hidden_dim)
        self.nutr_attn = NutrientAttention(hidden_dim, heads)

        convs = nn.ModuleDict({
            'ui': GATConv((-1,-1), hidden_dim, heads=1, add_self_loops=False),
            'fn': GATConv((-1,-1), hidden_dim, heads=1, add_self_loops=False),
            'ff': GATConv((-1,-1), hidden_dim, heads=1, add_self_loops=False),
            'du': GATConv((-1,-1), hidden_dim, heads=1, add_self_loops=False),
        })
        self.conv = HeteroConv({
            ('user','interacts','food'):        convs['ui'],
            ('food','rev_interacts','user'):    convs['ui'],
            ('food','contains','nutrient'):     convs['fn'],
            ('nutrient','rev_contains','food'): convs['fn'],
            ('food','similar','food'):          convs['ff'],
            ('food','rev_similar','food'):      convs['ff'],
            ('user','hasRisk','disease'):       convs['du'],
            ('disease','rev_hasRisk','user'):   convs['du'],
        }, aggr='mean')
        self.out_lin = Linear(hidden_dim, hidden_dim)

    def forward(self, x_dict, edge_index_dict, nutrient_ctx):
        x_dict = {
            'user': F.relu(self.user_enc(x_dict['user'])),
            'food': F.relu(self.food_enc(x_dict['food'])),
            'nutrient': x_dict['nutrient'],
            'disease': x_dict['disease']
        }
        x_dict = self.conv(x_dict, edge_index_dict)
        x_dict['food'] = self.nutr_attn(x_dict['food'], nutrient_ctx)
        return {k: self.out_lin(F.relu(v)) for k,v in x_dict.items()}

class SuitabilityMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim, 1)
        )
    def forward(self, user_e, food_e):
        return self.mlp(torch.cat([user_e, food_e], dim=-1)).squeeze(-1)

class NRKGSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = DietGNN(HIDDEN_DIM, N_HEADS)
        self.mlp = SuitabilityMLP(HIDDEN_DIM)
    def forward(self, data, nutr_ctx):
        embeds = self.gnn(data.x_dict, data.edge_index_dict, nutr_ctx)
        user_e, food_e = embeds['user'], embeds['food']
        dot_scores = torch.matmul(user_e, food_e.T).view(-1)
        mlp_scores = self.mlp(user_e.repeat_interleave(food_e.size(0), 0),
                              food_e.repeat(user_e.size(0), 1))
        return dot_scores + mlp_scores

# ================= TRAINING ===================
print("Preparing training pairs …")
pos_pairs = [(uid_map[u], fid_map[f]) for u,f in zip(edge_df[edge_df['relation'] == INTERACT_REL]['source'], edge_df[edge_df['relation'] == INTERACT_REL]['target']) if u in uid_map and f in fid_map]

if len(pos_pairs) == 0:
    print("⚠️ No positive pairs found — creating dummy ones for testing")
    pos_pairs = [(random.choice(list(uid_map.values())), random.choice(list(fid_map.values()))) for _ in range(20)]

pos_set = set(pos_pairs)
neg_pairs = []
while len(neg_pairs) < len(pos_pairs):
    u, f = random.choice(list(uid_map.values())), random.choice(list(fid_map.values()))
    if (u,f) not in pos_set:
        neg_pairs.append((u,f))

pairs = pos_pairs + neg_pairs
labels = torch.cat([torch.ones(len(pos_pairs)), torch.zeros(len(neg_pairs))])
train_idx, val_idx = train_test_split(np.arange(len(pairs)), test_size=0.2, random_state=SEED)

model = NRKGSystem().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
bce = nn.BCEWithLogitsLoss()
food_nutr_ctx = torch.tensor(scaler.transform(food_df[FOOD_NUTR_COLS]), dtype=torch.float).to(DEVICE)
pairs_t = torch.tensor(pairs, dtype=torch.long)

# ================= TRAINING & MODEL LOADING ===================
MODEL_PATH = "G:/FYP_Diabetic_Prediction_Recomandations/artifact/nutrition_recommendations/model_checkpoint.pth"

def train_model():
    print("Preparing training pairs …")
    pos_pairs = [(uid_map[u], fid_map[f]) for u, f in zip(
        edge_df[edge_df['relation'] == INTERACT_REL]['source'],
        edge_df[edge_df['relation'] == INTERACT_REL]['target']
    ) if u in uid_map and f in fid_map]

    if len(pos_pairs) == 0:
        print("⚠️ No positive pairs found — creating dummy ones for testing")
        pos_pairs = [(random.choice(list(uid_map.values())), random.choice(list(fid_map.values()))) for _ in range(20)]

    pos_set = set(pos_pairs)
    neg_pairs = []
    while len(neg_pairs) < len(pos_pairs):
        u, f = random.choice(list(uid_map.values())), random.choice(list(fid_map.values()))
        if (u, f) not in pos_set:
            neg_pairs.append((u, f))

    pairs = pos_pairs + neg_pairs
    labels = torch.cat([torch.ones(len(pos_pairs)), torch.zeros(len(neg_pairs))])
    train_idx, _ = train_test_split(np.arange(len(pairs)), test_size=0.2, random_state=SEED)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    pairs_t = torch.tensor(pairs, dtype=torch.long)

    print("Training …")
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(train_idx))
        losses = []
        for start in range(0, len(train_idx), BATCH_SIZE):
            batch_ids = train_idx[perm[start:start + BATCH_SIZE]]
            u_idx, f_idx = pairs_t[batch_ids][:, 0], pairs_t[batch_ids][:, 1]
            opt.zero_grad()
            logits = model(data.to(DEVICE), food_nutr_ctx)
            preds = logits[u_idx * len(fid_map) + f_idx]
            loss = bce(preds, labels[batch_ids].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.item())
        print(f"Epoch {epoch:02d}  train loss = {np.mean(losses):.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ Model saved to:", MODEL_PATH)
    return model

# Initialize model and either load or train
model = NRKGSystem().to(DEVICE)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()
    print("✅ Loaded model from checkpoint.")
else:
    model = train_model()

# ================= INFERENCE ==================

def generate_meal_plan_for_user_from_data(user_data, days=7):
    user_culture = user_data.get('preferences', {}).get('culture', 'Sri Lankan')  # Default to 'Sri Lankan'
    diabetes_risk = user_data['diabetes_risk']
    nutrition_risk = user_data['nutrition_risk']

    with torch.no_grad():
        model.eval()
        embeds = model.gnn(data.x_dict, data.edge_index_dict, food_nutr_ctx)
        scores = model.mlp(embeds['user'].mean(dim=0).repeat(embeds['food'].shape[0], 1), embeds['food']).sigmoid()
        top_indices = scores.topk(300).indices.cpu().numpy()

    topk_df = food_df.iloc[top_indices].copy()

    if 'culture' in topk_df.columns:
        topk_df = topk_df[topk_df['culture'] == user_culture]

    if 'sugar' in topk_df.columns and diabetes_risk > 50:
        topk_df = topk_df[topk_df['sugar'] < 30]

    def estimate_portion(base_weight):
        if nutrition_risk > 70:
            return round(base_weight * 0.6, 1)
        elif nutrition_risk > 40:
            return round(base_weight * 0.8, 1)
        else:
            return base_weight

    plan = []
    meals = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
    for day in range(days):
        total_grams = 1000 if nutrition_risk > 70 else 1400 if nutrition_risk > 40 else 1800
        per_meal_target = total_grams / 4
        daily_meals = topk_df.sample(n=min(4, len(topk_df)))
        for meal_time, (_, row) in zip(meals, daily_meals.iterrows()):
            portion = estimate_portion(row['estimated_weight_g']) if 'estimated_weight_g' in row else per_meal_target
            nutrition = {col: row[col] for col in FOOD_NUTR_COLS if col in row}
            plan.append({
                'day': f"Day {day+1}",
                'meal': meal_time,
                'food_id': row['food_id'],
                'food_item': row['food_item'],
                'portion_g': portion,
                'nutrients': nutrition
            })

    return plan   