import torch
import torch.nn as nn
import sys
import numpy as np
import torch
import torch.nn.functional as F

import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

def kge_loss(pred, target):
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)    
    pred_std = torch.std(pred)
    target_std = torch.std(target)
    correlation = torch.corrcoef(torch.stack([pred.flatten(), target.flatten()]))[0, 1]
    beta = pred_mean / (target_mean + 1e-6)  # Avoid division by zero
    gamma = pred_std / (target_std + 1e-6)  # Avoid division by zer
    kge = 1 - torch.sqrt((correlation - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    kge_ss=1-((1-kge)/torch.sqrt(2))
    return -kge_ss  # Minimize negative KGE

def kge(pred, target):
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)    
    pred_std = torch.std(pred)
    target_std = torch.std(target)
    correlation = torch.corrcoef(torch.stack([pred.flatten(), target.flatten()]))[0, 1]
    beta = pred_mean / (target_mean + 1e-6)  # Avoid division by zero
    gamma = pred_std / (target_std + 1e-6)  # Avoid division by zer
    kge = 1 - torch.sqrt((correlation - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    kge_ss=1-((1-kge)/torch.sqrt(2))
    return kge_ss, beta, gamma,correlation   # Minimize negative KGE





import torch

def gkge_static_median_rearranged(Qo, days_per_year=365):
    """
    Prepare components for GKGE based on median-split within each year, rearranged per year.

    Args:
        Qo: Tensor [N] observed discharge
        days_per_year: int, typically 365

    Returns:
        Qo_transformed: Tensor [N] transformed and rearranged Qo
        Qs_rearrangement_order: Tensor [N] rearrangement indices for Qs
        group_indices: Tensor mapping each day to a group (year + subgroup)
    """
    n = Qo.shape[0]
    device = Qo.device
    inverse_indices = (torch.arange(n, device=device) // days_per_year).long()
    n_years = inverse_indices.max().item() + 1

    Qo_transformed = []
    Qs_rearrangement_order = []
    group_indices = []

    current_index = 0

    for year in range(n_years):
        year_mask = (inverse_indices == year)
        Qo_year = Qo[year_mask]

        if Qo_year.numel() == 0:
            continue  # Skip empty years

        median_value = (Qo_year.median())

        # Get indices within the year
        year_indices = year_mask.nonzero(as_tuple=True)[0]

        # Split into subgroups
        Qo1_mask = Qo_year < median_value
        Qo2_mask = ~Qo1_mask

        # Transform Qo1 (power 1/3), keep Qo2 as is
        Qo1_transformed = Qo_year[Qo1_mask] 
        Qo2_transformed = Qo_year[Qo2_mask]

        # Stack Qo1 and Qo2
        Qo_transformed.append(Qo1_transformed)
        Qo_transformed.append(Qo2_transformed)

        # Create rearrangement order for Qs
        Qs_rearrangement_order.append(year_indices[Qo1_mask])
        Qs_rearrangement_order.append(year_indices[Qo2_mask])

        # Assign group indices (unique for each subgroup)
        group_indices.append(torch.full((Qo1_transformed.shape[0],), current_index, device=device))
        current_index += 1
        group_indices.append(torch.full((Qo2_transformed.shape[0],), current_index, device=device))
        current_index += 1

    # Concatenate all groups
    Qo_transformed = torch.cat(Qo_transformed)
    Qs_rearrangement_order = torch.cat(Qs_rearrangement_order)
    group_indices = torch.cat(group_indices)

    return Qo_transformed, Qs_rearrangement_order, group_indices

def compute_gkge_median_rearranged(Qs, Qo_transformed, Qs_rearrangement_order, group_indices):
    """
    Compute GKGE on rearranged and transformed Qs and Qo using median-based groups.
    
    Args:
        Qs: Tensor [N] simulated discharge (original)
        Qo_transformed: Tensor [N] transformed and rearranged observed discharge
        Qs_rearrangement_order: Tensor [N] indices to rearrange Qs to match Qo
        group_indices: Tensor [N] mapping each day to a group (subgroup within each year)

    Returns:
        GKGE loss (negative of GKGE)
    """
    device = Qs.device

    # Rearrange Qs to match Qo rearrangement
    Qs_rearranged = Qs[Qs_rearrangement_order]

    # Get unique groups
    unique_groups = torch.unique(group_indices)
    n_groups = unique_groups.numel()

    
    Qs_rearranged[group_indices % 2 == 0] = Qs_rearranged[group_indices % 2 == 0] 

    # Compute group-wise counts and sums
    group_counts = torch.zeros(n_groups, device=device).scatter_add(0, group_indices, torch.ones_like(Qo_transformed))
    Qs_sum = torch.zeros(n_groups, device=device).scatter_add(0, group_indices, Qs_rearranged)
    Qo_sum = torch.zeros(n_groups, device=device).scatter_add(0, group_indices, Qo_transformed)

    Qs_mean_per_group = Qs_sum / (group_counts + 1e-8)
    Qo_mean_per_group = Qo_sum / (group_counts + 1e-8)

    # Broadcast means to each time step
    Zs = Qs_mean_per_group[group_indices]
    Zo = Qo_mean_per_group[group_indices]

    # GKGE components
    Bg = torch.mean((1 - Zs / (Zo + 1e-8)) ** 2)
    psi_s = torch.sqrt(torch.mean((Qs_rearranged - Zs) ** 2))
    psi_o = torch.sqrt(torch.mean((Qo_transformed - Zo) ** 2))
    alpha_g = psi_s / (psi_o + 1e-8)
    Ag = (1 - alpha_g) ** 2
    Rg = torch.mean(((Qs_rearranged - Zs) / (psi_s + 1e-8)) * ((Qo_transformed - Zo) / (psi_o + 1e-8)))
    Rg_sq = (1 - Rg) ** 2

    GKGE = 1 - torch.sqrt((Bg + Ag + Rg_sq) / 2)
    return -GKGE


def gkge_static(Qo, splits_per_year=1, days_per_year=365, verbose=True):
    """
    Prepare static components for GKGE: group indices and Zo (observed benchmark).
    Truncates extra data that doesn't perfectly fit the split groups.

    Args:
        Qo: Tensor [N] observed discharge
        splits_per_year: int, number of parts to divide each year into (e.g., 4 for quarterly)
        days_per_year: int, typically 365
        verbose: bool, whether to print truncation warning

    Returns:
        inverse_indices: Tensor [N_truncated] mapping each day to a group (split-year)
        Zo: Tensor [N_truncated] mean Qo for each group (split-year)
        group_counts: Tensor [num_groups] number of samples in each group
    """
    n = Qo.shape[0]
    device = Qo.device

    # Calculate number of days per split
    days_per_split = days_per_year / splits_per_year

    # Calculate maximum valid length that fits full groups
    max_valid_length = int((n // days_per_split) * days_per_split)

    # Optionally print truncation info
    if verbose and max_valid_length < n:
        print(f"Data truncated from {n} to {max_valid_length} to fit complete splits.")

    # Truncate Qo
    Qo = Qo[:max_valid_length]

    # Recalculate n after truncation
    n = Qo.shape[0]

    # Assign each time step to a split group
    inverse_indices = (torch.arange(n, device=device) / days_per_split).floor().long()
    n_groups = inverse_indices.max().item() + 1

    # Compute mean Qo per group
    Qo_sum = torch.zeros(n_groups, device=device).scatter_add(0, inverse_indices, Qo)
    group_counts = torch.zeros(n_groups, device=device).scatter_add(0, inverse_indices, torch.ones_like(Qo))
    Qo_mean_per_group = Qo_sum / (group_counts + 1e-8)

    # Broadcast mean to all time steps
    Zo = Qo_mean_per_group[inverse_indices]

    return inverse_indices, Zo, group_counts

def compute_gkge(Qs, Qo, inverse_indices, Zo, group_counts):
    """
    Compute GKGE using predicted Qs and precomputed inverse_indices and Zo.
    Automatically truncates Qs and Qo to match the length of the provided Zo.

    Parameters:
        Qs : torch.Tensor, simulated discharge [N or longer]
        Qo : torch.Tensor, observed discharge [N or longer]
        inverse_indices : torch.Tensor, group index for each sample [N_truncated]
        Zo : torch.Tensor, observed mean per group broadcasted to [N_truncated]
        group_counts : torch.Tensor, number of samples per group

    Returns:
        GKGE : torch.Tensor (scalar)
    """
    device = Qs.device
    n_groups = group_counts.size(0)
    n_target = Zo.shape[0]  # Truncated length

    # Automatically truncate Qs and Qo to match Zo length
    Qs = Qs[:n_target]
    Qo = Qo[:n_target]

    # Compute mean Qs per group
    Qs_sum = torch.zeros(n_groups, device=device).scatter_add(0, inverse_indices, Qs)
    Qs_mean_per_group = Qs_sum / (group_counts + 1e-8)
    Zs = Qs_mean_per_group[inverse_indices]  # Broadcast mean to all days in that group

    # GKGE components
    Bg = torch.mean((1 - Zs / (Zo + 1e-8)) ** 2)
    psi_s = torch.sqrt(torch.mean((Qs - Zs) ** 2))
    psi_o = torch.sqrt(torch.mean((Qo - Zo) ** 2))
    alpha_g = psi_s / (psi_o + 1e-8)
    Ag = (1 - alpha_g) ** 2
    Rg = torch.mean(((Qs - Zs) / (psi_s + 1e-8)) * ((Qo - Zo) / (psi_o + 1e-8)))
    Rg_sq = (1 - Rg) ** 2

    GKGE = 1 - torch.sqrt((Bg + Ag + Rg_sq) / 2)
    return -GKGE



'''
def gkge_static(Qo):
    """
    Prepare static components for GKGE: year indices and Zo (observed benchmark).
    
    Returns:
        inverse_indices: Tensor [N] mapping each day to a year group.
        Zo: Tensor [N] yearly mean for each day (based on Qo).
    """
    days_per_year=365
    n = Qo.shape[0]
    device = Qo.device

    # Assign each time step to a year
    inverse_indices = (torch.arange(n, device=device) // days_per_year).long()
    n_years = inverse_indices.max().item() + 1

    # Compute yearly mean for Qo
    Qo_sum = torch.zeros(n_years, device=device).scatter_add(0, inverse_indices, Qo)
    year_counts = torch.zeros(n_years, device=device).scatter_add(0, inverse_indices, torch.ones_like(Qo))
    Qo_mean_per_year = Qo_sum / (year_counts + 1e-8)
    Zo = Qo_mean_per_year[inverse_indices]  # full-length observed benchmark

    return inverse_indices, Zo, year_counts

def compute_gkge(Qs, Qo, inverse_indices, Zo, year_counts):
    """
    Compute GKGE using predicted Qs and precomputed inverse_indices and Zo.
    
    Parameters:
        Qs : torch.Tensor, simulated discharge [N]
        Qo : torch.Tensor, observed discharge [N]
        inverse_indices : torch.Tensor, year index for each sample [N]
        Zo : torch.Tensor, observed yearly mean broadcasted to [N]
    
    Returns:
        GKGE : torch.Tensor (scalar)
    """
    device = Qs.device
    n_years = year_counts.size(0)

    # Compute yearly mean for Qs
    Qs_sum = torch.zeros(n_years, device=device).scatter_add(0, inverse_indices, Qs)
    Qs_mean_per_year = Qs_sum / (year_counts + 1e-8)
    Zs = Qs_mean_per_year[inverse_indices]  # full-length simulated benchmark

    # GKGE components
    Bg = torch.mean((1 - Zs / (Zo + 1e-8)) ** 2)
    psi_s = torch.sqrt(torch.mean((Qs - Zs) ** 2))
    psi_o = torch.sqrt(torch.mean((Qo - Zo) ** 2))
    alpha_g = psi_s / (psi_o + 1e-8)
    Ag = (1 - alpha_g) ** 2
    Rg = torch.mean(((Qs - Zs) / (psi_s + 1e-8)) * ((Qo - Zo) / (psi_o + 1e-8)))
    Rg_sq = (1 - Rg) ** 2

    GKGE = 1 - torch.sqrt((Bg + Ag + Rg_sq) / 2)
    return -GKGE

'''
def get_direct_arrays(forcing_data):

    #areaa = forcing_data[0, 13].item()
    dt = 86400  # Daily Time step
    zm = 200; k=0.41; Z0_bare=0.0002
    SMo, alpha_T, cp, density, k, KD1, KD2, KR, T0, Th, Tl, boltzman, Z0_bare, e_surface = 5000, 1, 1013, 1.23, 0.41, -0.307, 0.019, 200, 300.5, 328, 273, 5.67e-08, 0.0002, 0.95
    # Wind speed calculations (ensure columns are correct, convert to tensors)
    wind_u = forcing_data[:, 11]  # U component of wind
    wind_v = forcing_data[:, 12]  # V component of wind
    um_10m = torch.sqrt(wind_u ** 2 + wind_v ** 2)  # Wind speed at 10m
    um = um_10m * (zm / 10) ** 0.14  # Wind speed at 30m
    um2 = um_10m * (1 / 10) ** 0.14  # Wind speed at 1m
    # Temperature and other data (converted to tensors)
    Temp_2m = forcing_data[:, 1]-273.15   # Convert from Kelvin to Celsius
    Temp_30m = Temp_2m   # Temperature at 30m
    ShortW_Ground = forcing_data[:, 5]  # Net Shortwave radiation
    LongW_down = forcing_data[:, 4]  # Longwave down radiation
    Precip = forcing_data[:, 3]  # Precipitation in mm
    P = forcing_data[:, 8] / 1000  # Air Pressure in kPa
    LAI = forcing_data[:, 7]*8  # Leaf Area Index (LAI)
    albedo = forcing_data[:, 2]  # Albedo
    sk_temp=forcing_data[:, 9]  # Surface Temp
    SPFH=forcing_data[:, 10]  # Surface Temp
    # Vapor Pressure (sat/actual), Deficit, RH
    e_sat = 0.6108 * torch.exp((17.27 * Temp_30m) / (237.3 + Temp_30m))
    e = (SPFH * P) / 0.622
    Deficit = e_sat - e 
    T_kelvin = Temp_30m + 273.17
    Rn = ShortW_Ground* (1 - albedo) + LongW_down - 0.97 * (5.67e-08) * (sk_temp ** 4)
    lamda = (2.501 - 0.002361 * Temp_30m) * 1000000
    delt = (4098 * e_sat) / ((237.3 + Temp_30m) ** 2)
    gama = (1013 * P) / (0.622 * lamda)

    return Rn, LAI, Deficit, lamda, delt, gama, um

def get_direct_arrays_scaled(forcing_data):

   # areaa = forcing_data[0, 13].item()
    dt = 86400  # Daily Time step
    zm = 200; k=0.41; Z0_bare=0.0002
    SMo, alpha_T, cp, density, k, KD1, KD2, KR, T0, Th, Tl, boltzman, Z0_bare, e_surface = 5000, 1, 1013, 1.23, 0.41, -0.307, 0.019, 200, 300.5, 328, 273, 5.67e-08, 0.0002, 0.95
    # Wind speed calculations (ensure columns are correct, convert to tensors)
    wind_u = forcing_data[:, 11]  # U component of wind
    wind_v = forcing_data[:, 12]  # V component of wind
    um_10m = torch.sqrt(wind_u ** 2 + wind_v ** 2)  # Wind speed at 10m
    um = um_10m * (zm / 10) ** 0.14  # Wind speed at 30m
    um2 = um_10m * (1 / 10) ** 0.14  # Wind speed at 1m
    # Temperature and other data (converted to tensors)
    Temp_2m = forcing_data[:, 1]-273.15   # Convert from Kelvin to Celsius
    Temp_30m = Temp_2m   # Temperature at 30m
    ShortW_Ground = forcing_data[:, 5]  # Net Shortwave radiation
    LongW_down = forcing_data[:, 4]  # Longwave down radiation
    Precip = forcing_data[:, 3]  # Precipitation in mm
    P = forcing_data[:, 8] / 1000  # Air Pressure in kPa
    LAI = forcing_data[:, 7]*8  # Leaf Area Index (LAI)
    albedo = forcing_data[:, 2]  # Albedo
    sk_temp=forcing_data[:, 9]  # Surface Temp
    SPFH=forcing_data[:, 10]  # Surface Temp
    # Vapor Pressure (sat/actual), Deficit, RH
    e_sat = 0.6108 * torch.exp((17.27 * Temp_30m) / (237.3 + Temp_30m))
    e = (SPFH * P) / 0.622
    Deficit = e_sat - e 
    T_kelvin = Temp_30m + 273.17
    Rn = ShortW_Ground* (1 - albedo) + LongW_down - 0.97 * (5.67e-08) * (sk_temp ** 4)
    lamda = (2.501 - 0.002361 * Temp_30m) * 1000000
    delt = (4098 * e_sat) / ((237.3 + Temp_30m) ** 2)
    gama = (1013 * P) / (0.622 * lamda)
        # Stack tensors to form a 2D array
    data = torch.stack([Rn, LAI, Deficit, lamda, delt, gama, um], dim=1).numpy()
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    # Fit and transform
    scaled_data = scaler.fit_transform(data)
    # Convert back to PyTorch tensors
    scaled_tensors = torch.tensor(scaled_data, dtype=torch.float32)
    # Split back into individual tensors
    Rn_scaled, LAI_scaled, Deficit_scaled, lamda_scaled, delt_scaled, gama_scaled, um_scaled = scaled_tensors.T

    return Rn_scaled, LAI_scaled, Deficit_scaled, lamda_scaled, delt_scaled, gama_scaled, um_scaled




import torch
import torch.nn as nn
import torch.nn.functional as F

class ET_surrogate(nn.Module):
    def __init__(self):
        super(ET_surrogate, self).__init__()
        self.fc1 = nn.Linear(6, 10)   # Input → Hidden (15 nodes)
        self.out = nn.Linear(10, 1)   # Hidden → Output

        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        
        x = torch.log1p(x)        # Log-transform input
        x = F.relu(self.fc1(x))   # Hidden layer with ReLU
        x = self.out(x)           # Output layer
        return F.softplus(x)      # Ensure ET ≥ 0


class SimpleMonotonicETSurrogate(nn.Module):
    def __init__(self):
        super(SimpleMonotonicETSurrogate, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 1)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        w1 = F.softplus(self.fc1.weight)  # enforce positive weights
        w2 = F.softplus(self.fc2.weight)

        x = F.softplus(F.linear(x, w1, self.fc1.bias))
        x = F.softplus(F.linear(x, w2, self.fc2.bias))
        return x.squeeze(-1)



class canopy_NN(nn.Module):
    def __init__(self):
        super(canopy_NN, self).__init__()
        self.fc1 = nn.Linear(4, 14)  # Input: 4 → Hidden: 4
        self.fc2 = nn.Linear(14, 1)  # Hidden: 4 → Output: 1

        # Xavier initialization for weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))        # Hidden layer with ReLU
        x = self.fc2(x)                # Raw output
        x = F.softplus(x)              # Enforce positivity
        return 50 + x * 20            # Offset and scale for physical realism

    

class Rs_soil(nn.Module):
    def __init__(self):
        super(Rs_soil, self).__init__()
        self.fc1 = nn.Linear(4, 14)  # Input: 4 values → Hidden: 4 nodes
        self.fc2 = nn.Linear(14, 1)  # Hidden: 4 nodes → Output: 1 value

        # Xavier initialization for weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
       x = F.relu(self.fc1(x))        # Hidden layer with ReLU
       x = self.fc2(x)                # Raw output
       x = F.softplus(x)              # Enforce positivity
       return 50 + x * 20           # Offset and scale for physical realism







def split_train_eval_by_year_extremes(Qo: torch.Tensor, days_per_year=365):
    Qo = Qo.flatten()  # Ensure 1D tensor
    total_days = Qo.size(0)
    num_years = total_days // days_per_year

    # Step 1: Reshape to (num_years, days_per_year)
    Qo_years = Qo[:num_years * days_per_year].reshape(num_years, days_per_year)

    # Step 2: Compute yearly totals
    yearly_sums = Qo_years.sum(dim=1)  # shape: (num_years,)
    print(yearly_sums)

    # Step 3: Sort years by total descending
    sorted_indices = torch.argsort(yearly_sums, descending=True)
    sorted_years = sorted_indices.tolist()

    # Step 4: Pair extremes
    pairs = []
    while len(sorted_years) > 1:
        pair = (sorted_years[0], sorted_years[-1])
        pairs.append(pair)
        sorted_years.pop(0)
        sorted_years.pop(-1)

    print(pairs)

    # Step 5: Alternate train/eval assignment
    train_pairs = []
    eval_pairs = []
    for i, pair in enumerate(pairs):
        if i % 2 == 0:
            train_pairs.append(pair)
        else:
            eval_pairs.append(pair)

    # Calculate desired training pairs count (60%)
    total_pairs = len(pairs)
    desired_train_count = round(0.8 * total_pairs)

    # Move pairs from end of eval to train until train has 60% pairs
    while len(train_pairs) < desired_train_count and eval_pairs:
        pair_to_move = eval_pairs.pop()  # from the end
        train_pairs.append(pair_to_move)

    # Surplus year goes to train if exists
    if sorted_years:
        train_pairs.append((sorted_years[0],))  # surplus year as a tuple for consistency

    # Flatten pairs to years
    train_years = []
    for pair in train_pairs:
        train_years.extend(pair)

    # Step 6: Sort train_years by descending yearly sum
    train_years = sorted(train_years, key=lambda i: -yearly_sums[i].item())
    print(train_years)

    # Step 7: Collect train data and indices
    Qo_train_processed = []
    Qo_train_indices = []

    for year_idx in train_years:
        start_idx = year_idx * days_per_year
        end_idx = start_idx + days_per_year
        Qo_train_processed.append(Qo[start_idx:end_idx])
        Qo_train_indices.extend(range(start_idx, end_idx))

    Qo_train_processed = torch.cat(Qo_train_processed)
    Qo_train_indices = torch.tensor(Qo_train_indices, dtype=torch.long)

    return Qo_train_processed, Qo_train_indices



def extract_Qs_train(Qs: torch.Tensor, Qo_indices: torch.Tensor):
    Qs = Qs.flatten()
    Qs_train_processed = Qs[Qo_indices]
    return Qs_train_processed


class GKGE_v1:
    def __init__(self, Qo, year_length=365, quantile_splits_perYear=2, verbose=True):
        """
        Qo : torch.Tensor [N] observed discharge
        year_length : int, days per hydrological year (e.g. 365)
        quantile_splits_perYear : int >= 1
            - 1 = no split, 1 group per year
            - 2 = median split, 2 groups per year
            - k = quantile split into k groups per year
        """
        self.device = Qo.device
        self.year_length = year_length
        self.quantile_splits_perYear = quantile_splits_perYear

        n = Qo.shape[0]

        # truncate to full years
        if year_length == 999:  # -1 means full length
            self.n_years = 1
            max_valid_length = n
            
        else:
            n_years = n // year_length
            max_valid_length = n_years * year_length
            if verbose and max_valid_length < n:
                print(f"Truncated Qo from {n} to {max_valid_length} to fit full years.")
            self.n_years = n_years
        
        Qo = Qo[:max_valid_length]
        self.Qo = Qo


        # Precompute static groupings (Qo transformed, rearrangement order, group indices)
        self.Qo_transformed, self.Qs_rearrangement_order, self.group_indices = \
            self._prepare_groups(Qo)

    def _prepare_groups(self, Qo):
        """Split each year into quantile_splits_perYear groups."""
        n = Qo.shape[0]
        device = Qo.device
        group_indices = []
        Qs_rearrangement_order = []
        Qo_transformed = []
        
        current_index = 0
        for year in range(self.n_years):
            year_mask = torch.arange(n, device=device) // self.year_length == year
            Qo_year = Qo[year_mask]
            if Qo_year.numel() == 0:
                continue
            year_indices = year_mask.nonzero(as_tuple=True)[0]

            if self.quantile_splits_perYear == 1:
                # no split, one group per year
                Qo_transformed.append(Qo_year)
                Qs_rearrangement_order.append(year_indices)
                group_indices.append(torch.full((Qo_year.shape[0],), current_index, device=device))
                current_index += 1
            else:
                # split into quantiles
                quantiles = torch.quantile(
                    Qo_year, 
                    torch.linspace(0, 1, self.quantile_splits_perYear + 1, device=device)
                )
                for q in range(self.quantile_splits_perYear):
                    q_low, q_high = quantiles[q].item(), quantiles[q + 1].item()
                    mask = (Qo_year >= q_low) & (Qo_year <= q_high)
                    Qo_sub = Qo_year[mask]
                    idx_sub = year_indices[mask]
                    if Qo_sub.numel() == 0:
                        continue
                    Qo_transformed.append(Qo_sub)
                    Qs_rearrangement_order.append(idx_sub)
                    group_indices.append(torch.full((Qo_sub.shape[0],), current_index, device=device))
                    current_index += 1
       
        return torch.cat(Qo_transformed), torch.cat(Qs_rearrangement_order), torch.cat(group_indices)

    def compute_gkge(self, Qs):
        """
        Compute GKGE value for given simulated discharge Qs.
        """
        Qs = Qs[:self.Qo.shape[0]]

        # rearrange simulated to match group rearrangement order
        Qs_rearranged = Qs[self.Qs_rearrangement_order]
        Qo_transformed = self.Qo_transformed
        group_indices = self.group_indices

        # group-wise sums
        unique_groups = torch.unique(group_indices)
        n_groups = unique_groups.numel()

        group_counts = torch.zeros(n_groups, device=self.device).scatter_add_(
            0, group_indices, torch.ones_like(Qo_transformed)
        )
        Qs_sum = torch.zeros(n_groups, device=self.device).scatter_add_(
            0, group_indices, Qs_rearranged
        )
        Qo_sum = torch.zeros(n_groups, device=self.device).scatter_add_(
            0, group_indices, Qo_transformed
        )

        Qs_mean_per_group = Qs_sum / (group_counts + 1e-8)
        Qo_mean_per_group = Qo_sum / (group_counts + 1e-8)

        Zs = Qs_mean_per_group[group_indices]
        Zo = Qo_mean_per_group[group_indices]
        
        # GKGE components
        Bg = torch.mean((1 - Zs / (Zo + 1e-8)) ** 2)
        #print(Qs_rearranged )
        psi_s = torch.sqrt(torch.mean((Qs_rearranged - Zs) ** 2))
        psi_o = torch.sqrt(torch.mean((Qo_transformed - Zo) ** 2))
        alpha_g = psi_s / (psi_o + 1e-8)
        Ag = (1 - alpha_g) ** 2
        Rg = torch.mean(((Qs_rearranged - Zs) / (psi_s + 1e-8)) * ((Qo_transformed - Zo) / (psi_o + 1e-8)))
        Rg_sq = (1 - Rg) ** 2

        GKGE = 1 - torch.sqrt((Bg + Ag + Rg_sq) / 2)
        return -GKGE
    
    def compute_gkge_V2(self, Qs):
        """
        Std also varying per group
        """
        Qs = Qs[:self.Qo.shape[0]]
    
        # rearrange simulated to match group rearrangement order
        Qs_rearranged = Qs[self.Qs_rearrangement_order]
        Qo_transformed = self.Qo_transformed
        group_indices = self.group_indices
    
        # group-wise counts
        unique_groups = torch.unique(group_indices)
        n_groups = unique_groups.numel()
    
        group_counts = torch.zeros(n_groups, device=self.device).scatter_add_(
            0, group_indices, torch.ones_like(Qo_transformed)
        )
        Qs_sum = torch.zeros(n_groups, device=self.device).scatter_add_(
            0, group_indices, Qs_rearranged
        )
        Qo_sum = torch.zeros(n_groups, device=self.device).scatter_add_(
            0, group_indices, Qo_transformed
        )
    
        # group means
        Qs_mean_per_group = Qs_sum / (group_counts + 1e-8)
        Qo_mean_per_group = Qo_sum / (group_counts + 1e-8)
    
        # expand means back to timeseries
        Zs = Qs_mean_per_group[group_indices]
        Zo = Qo_mean_per_group[group_indices]
    
        # ---- NEW: group standard deviations ----
        Qs_sq_diff = (Qs_rearranged - Zs) ** 2
        Qo_sq_diff = (Qo_transformed - Zo) ** 2
    
        Qs_sq_sum = torch.zeros(n_groups, device=self.device).scatter_add_(
            0, group_indices, Qs_sq_diff
        )
        Qo_sq_sum = torch.zeros(n_groups, device=self.device).scatter_add_(
            0, group_indices, Qo_sq_diff
        )
    
        Std_S_per_group = torch.sqrt(Qs_sq_sum / (group_counts + 1e-8))
        Std_O_per_group = torch.sqrt(Qo_sq_sum / (group_counts + 1e-8))
    
        # expand stds back to timeseries
        Ss = Std_S_per_group[group_indices]
        So = Std_O_per_group[group_indices]
    
        # ---- GKGE components ----
        Bg = torch.mean((1 - Zs / (Zo + 1e-8)) ** 2)
    
        # variability ratio using per-group stds
        alpha_g = Ss / (So + 1e-8)
        Ag = torch.mean((1 - alpha_g) ** 2)   # <-- mean of squared deviations
    
        Rg = torch.mean(
            ((Qs_rearranged - Zs) / (Ss + 1e-8)) *
            ((Qo_transformed - Zo) / (So + 1e-8))
        )
        Rg_sq = (1 - Rg) ** 2
    
        GKGE = 1 - torch.sqrt((Bg + Ag + Rg_sq) / 2)
        return -GKGE
