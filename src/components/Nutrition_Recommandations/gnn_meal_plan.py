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
pos_pairs = [(uid_map[u], fid_map[f]) for u,f in zip(edge_df[edge_df['relation'] == INTERACT_REL]['source'], edge_df[edge_df['relation'] == INTERACT_REL]['target']) if u in uid_map and f in fid_map]

if len(pos_pairs) == 0:
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
    return model

# Initialize model and either load or train
model = NRKGSystem().to(DEVICE)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()
else:
    model = train_model()

# ================= INFERENCE ==================
def is_food_risky(row, user_diseases):
    """
    Checks if a food row is risky for a user given their diseases.
    Uses both explicit food features and common-sense heuristics.
    """
    risky = False
    reasons = []

    # Metabolic/Endocrine Diseases
    if 'd1' in user_diseases:  # Diabetes
        if row.get('carbs', 0) > 50:
            risky = True
            reasons.append('High carbohydrate for diabetes')
        if row.get('glycemic_index', 0) > 60:
            risky = True
            reasons.append('High glycemic index for diabetes')
        if row.get('sugar', 0) > 20:
            risky = True
            reasons.append('High sugar for diabetes')

    # Obesity
    if 'd2' in user_diseases:  # Obesity
        if row.get('calories', 0) > 600:
            risky = True
            reasons.append('High calorie for obesity')
        if row.get('fat', 0) > 25:
            risky = True
            reasons.append('High fat for obesity')

    # Cardiovascular Conditions
    if any(d in user_diseases for d in ['d3', 'd4', 'd5', 'd6']):  # Hypertension, Heart Disease, CAD, Stroke
        if row.get('sodium', 0) > 400:
            risky = True
            reasons.append('High sodium for cardiovascular conditions')
        if row.get('fat', 0) > 25:
            risky = True
            reasons.append('High fat for cardiovascular disease')
        if row.get('sat_fat', 0) and row['sat_fat'] > 8:
            risky = True
            reasons.append('High saturated fat for cardiovascular disease')

    # Kidney Disease
    if 'd7' in user_diseases:  # Chronic Kidney Disease
        if row.get('protein', 0) > 30:
            risky = True
            reasons.append('High protein for kidney disease')
        if row.get('potassium', 0) and row['potassium'] > 700:
            risky = True
            reasons.append('High potassium for kidney disease')
        if row.get('phosphorus', 0) and row['phosphorus'] > 350:
            risky = True
            reasons.append('High phosphorus for kidney disease')

    # Liver Conditions
    if 'd8' in user_diseases:  # Non-Alcoholic Fatty Liver Disease
        if row.get('fat', 0) > 25:
            risky = True
            reasons.append('High fat for fatty liver disease')

    # PCOS and Related
    if any(d in user_diseases for d in ['d9', 'd11', 'd12']):  # PCOS, Insulin Resistance, Metabolic Syndrome
        if row.get('carbs', 0) > 45:
            risky = True
            reasons.append('High carbohydrate for metabolic conditions')

    # Lipid Disorders
    if 'd10' in user_diseases:  # Hyperlipidemia
        if row.get('fat', 0) > 25:
            risky = True
            reasons.append('High fat for hyperlipidemia')
        if row.get('sat_fat', 0) and row['sat_fat'] > 8:
            risky = True
            reasons.append('High saturated fat for hyperlipidemia')

    # Gout
    if 'd14' in user_diseases:  # Gout
        if any(food in row.get('food_item', '').lower() for food in ['beef', 'lamb', 'liver']):
            risky = True
            reasons.append('Purine-rich food for gout')

    # Pancreatitis
    if 'd16' in user_diseases:  # Pancreatitis
        if row.get('fat', 0) > 10:
            risky = True
            reasons.append('High fat for pancreatitis')

    # GERD
    if 'd37' in user_diseases:  # GERD
        if any(food in row.get('food_item', '').lower() for food in ['citrus', 'tomato', 'spicy']):
            risky = True
            reasons.append('May trigger GERD symptoms')

    return risky, reasons

def get_risky_food_ids_for_diseases(edge_df, food_df, disease_ids):
    """
    Find foods that are risky for given diseases based on the knowledge graph.
    """
    risky_food_ids = set()
    
    # Only check valid disease IDs that exist in our updated disease list
    valid_disease_ids = [d for d in disease_ids if d in [f'd{i}' for i in range(1, 59)]]
    
    for rel in ['notRecommended', 'riskFor']:
        mask = (edge_df['relation'] == rel) & (edge_df['target'].isin(valid_disease_ids))
        risky_food_ids.update(edge_df[mask]['source'])
    
    return {fid for fid in risky_food_ids if fid in set(food_df['food_id'])}

def adjust_portion(base_portion, diabetes_risk, nutrition_risk, user_diseases):
    portion = base_portion

    # 1. Risk-based portion control (more risk = smaller portions)
    if diabetes_risk > 70:
        portion *= 0.6
    elif diabetes_risk > 40:
        portion *= 0.8

    if nutrition_risk > 70:
        portion *= 0.8
    elif nutrition_risk > 40:
        portion *= 0.9

    # 2. Disease-based portion adjustment
    disease_factor = 1.0

    # Diabetes (d1): reduce post-meal glucose spike risk
    if diabetes_risk > 70:
        portion *= 0.6   # High diabetes risk: strong reduction
    elif diabetes_risk > 40:
        portion *= 0.8   # Moderate diabetes risk: moderate reduction

    if nutrition_risk > 70:
        portion *= 0.8   # High nutrition risk: additional reduction
    elif nutrition_risk > 40:
        portion *= 0.9

    # 2. Disease-based portion adjustment
    disease_factor = 1.0

    # Diabetes (d1): reduce post-meal glucose spike risk
    if 'd1' in user_diseases:
        disease_factor *= 0.8

    # Obesity (d2, d73): total calorie restriction
    if 'd2' in user_diseases or 'd73' in user_diseases:
        disease_factor *= 0.85

    # Hypertension (d3, d75): reduce sodium and volume
    if 'd3' in user_diseases or 'd75' in user_diseases:
        disease_factor *= 0.9

    # Heart & Vascular Disease
    if 'd4' in user_diseases:   # Heart Disease
        disease_factor *= 0.9
    if 'd5' in user_diseases:   # Coronary Artery Disease
        disease_factor *= 0.9
    if 'd6' in user_diseases:   # Stroke
        disease_factor *= 0.9
    if 'd76' in user_diseases:  # Cardiovascular Disease
        disease_factor *= 0.9

    # Chronic Kidney Disease (d7, d23): protein/potassium/phos restriction
    if 'd7' in user_diseases or 'd23' in user_diseases:
        disease_factor *= 0.7

    # Non-Alcoholic Fatty Liver Disease (d8)
    if 'd8' in user_diseases:
        disease_factor *= 0.9

    # PCOS & related (d9, d33): reduce carbs and calories
    if 'd9' in user_diseases or 'd33' in user_diseases:
        disease_factor *= 0.95

    # Hyperlipidemia (d10), High Cholesterol (d61): limit saturated fat, total energy
    if 'd10' in user_diseases or 'd61' in user_diseases:
        disease_factor *= 0.9

    # Insulin Resistance (d11)
    if 'd11' in user_diseases:
        disease_factor *= 0.9

    # Metabolic Syndrome (d12)
    if 'd12' in user_diseases:
        disease_factor *= 0.9

    # Gout (d14, d74): portion down to reduce purines/weight
    if 'd14' in user_diseases or 'd74' in user_diseases:
        disease_factor *= 0.9

    # Pancreatitis (d16), Liver Cirrhosis (d54): restrict fat/energy
    if 'd16' in user_diseases or 'd54' in user_diseases:
        disease_factor *= 0.8

    # Sleep Apnea (d13): reduce calorie intake for weight management
    if 'd13' in user_diseases:
        disease_factor *= 0.9

    # Asthma, COPD (d59, d60): generally no restriction unless severe
    if 'd59' in user_diseases or 'd60' in user_diseases:
        disease_factor *= 1.0

    # Chronic Constipation (d64): increase fiber but not portion
    if 'd64' in user_diseases:
        disease_factor *= 1.0

    # Thyroid Disorders (d15), Vitamin D Deficiency (d29), Anemia (d58), Osteoporosis (d28): no portion reduction
    for d in ['d15', 'd29', 'd58', 'd28']:
        if d in user_diseases:
            disease_factor *= 1.0

    # Allergies, Celiac, IBS, Food intolerances (d62–d73): handle by excluding foods, not portion
    for d in ['d62','d63','d64','d65','d66','d67','d68','d69','d70','d71','d72','d73']:
        if d in user_diseases:
            disease_factor *= 1.0

    # Other diseases (mental health, neuropathies, infections): no impact on portions by default
    for d in ['d17','d18','d19','d20','d21','d22','d24','d25','d26','d27','d30','d31','d32','d34','d35','d36','d37','d38','d39','d40','d41','d42','d43','d44','d45','d46','d47','d48','d49','d50','d51','d52','d53','d55','d56','d57']:
        if d in user_diseases:
            disease_factor *= 1.0


    # Allergies, Thyroid, Constipation, Anemia, etc.: usually no portion cut
    # ...already handled above, or excluded via food filtering

    portion *= disease_factor

    # Set sensible min/max to avoid extreme portions
    portion = max(80, min(portion, base_portion))

    return round(portion, 1)


# Updated Meal Plan Generator
def generate_meal_plan_for_user_from_data(user_data, days=7):
    user_culture = user_data.get('preferences', {}).get('culture', 'Sri Lankan')
    diabetes_risk = user_data['diabetes_risk']
    nutrition_risk = user_data['nutrition_risk']
    user_diseases = user_data.get('diseases', [])

    # KG-based risky food exclusion
    risky_food_ids = get_risky_food_ids_for_diseases(edge_df, food_df, user_diseases)

    with torch.no_grad():
        model.eval()
        embeds = model.gnn(data.x_dict, data.edge_index_dict, food_nutr_ctx)
        scores = model.mlp(embeds['user'].mean(dim=0).repeat(embeds['food'].shape[0], 1), embeds['food']).sigmoid()
        top_indices = scores.topk(500).indices.cpu().numpy()  # larger pool for better filtering

    topk_df = food_df.iloc[top_indices].copy()

    # Filter by user culture (if provided)
    if 'culture' in topk_df.columns:
        topk_df = topk_df[topk_df['culture'] == user_culture]

    # Optional: Filter by sugar for high diabetes risk
    if 'sugar' in topk_df.columns and diabetes_risk > 50:
        topk_df = topk_df[topk_df['sugar'] < 30]

    # Apply filtering for KG and nutrient-based risks
    safe_rows = []
    excluded_rows = []
    for idx, row in topk_df.iterrows():
        risky, reasons = is_food_risky(row, user_diseases)
        if (row['food_id'] not in risky_food_ids) and not risky:
            safe_rows.append(row)
        else:
            # Optionally collect or log excluded meals and reasons
            excluded_rows.append({"food_id": row['food_id'], "food_item": row['food_item'], "reasons": reasons})

    

    filtered_df = pd.DataFrame(safe_rows)
    if filtered_df.empty:
        raise Exception("No safe meals found for this user! Please relax constraints or check data.")

    # Now, create the meal plan from filtered_df
    plan = []
    meals = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
    for day in range(days):
        daily_meals = filtered_df.sample(n=min(4, len(filtered_df)))  # 4 meals per day
        for meal_time, (_, row) in zip(meals, daily_meals.iterrows()):
            base_portion = row['estimated_weight_g'] if 'estimated_weight_g' in row else 300
            portion = adjust_portion(
                base_portion,
                diabetes_risk,
                nutrition_risk,
                user_diseases
            )
            nutrition = {col: row[col] for col in FOOD_NUTR_COLS if col in row}
            plan.append({
                'day': f"Day {day+1}",
                'meal': meal_time,
                'food_id': row['food_id'],
                'food_item': row['food_item'],
                'portion_g': portion,
                'nutrients': nutrition
            })

    # Optional: return excluded_rows for review
    return plan   


