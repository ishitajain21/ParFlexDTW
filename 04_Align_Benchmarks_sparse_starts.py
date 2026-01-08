#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os.path
from pathlib import Path
import pickle
import multiprocessing
import concurrent.futures
import time
import gc
from tqdm import tqdm


# In[2]:


import import_ipynb


# In[3]:


import DTW


# In[4]:


import NWTW


# In[5]:


import FlexDTW


# In[6]:


DATASET = 'train' # 'test'
VERSION = 'full'


# In[7]:


QUERY_LIST = Path(f'cfg_files/queries.{DATASET}.{VERSION}')


# In[8]:


# SYSTEMS = ['dtw1', 'dtw2', 'dtw3', 'subseqdtw1', 'subseqdtw2', 'subseqdtw3', 'nwtw', 'flexdtw', 'parflex']
SYSTEMS = [ 'flexdtw', 'parflex']

BENCHMARKS = ['matching', 'subseq_20', 'subseq_30', 'subseq_40', 'partialStart', 'partialEnd', 'partialOverlap', 
              'pre_5', 'pre_10', 'pre_20', 'post_5', 'post_10', 'post_20', 'prepost_5', 'prepost_10',
              'prepost_20']


# In[9]:


features_root = Path('/home/asharma/ttmp/Flex/FlexDTW/Chopin_Mazurkas_features')
FEAT_DIRS = {}

for benchmark in BENCHMARKS:
    if benchmark == 'partialOverlap':
        FEAT_DIRS[benchmark] = ([features_root/'partialStart', features_root/'partialEnd'])
    elif 'prepost' in benchmark:
        sec = benchmark.split('_')[-1]
        FEAT_DIRS[benchmark] = ([features_root/f'pre_{sec}', features_root/f'post_{sec}'])
    else:
        FEAT_DIRS[benchmark] = [features_root/f'{benchmark}', features_root/'original']


# In[10]:


steps = {'dtw1': np.array([1,1,1,2,2,1]).reshape((-1,2)),
        'dtw2': np.array([1,1,1,2,2,1]).reshape((-1,2)),
        'dtw3': np.array([1,1,1,2,2,1]).reshape((-1,2)),
        'subseqdtw1': np.array([1,1,1,2,2,1]).reshape((-1,2)),
        'subseqdtw2': np.array([1,1,1,2,2,1]).reshape((-1,2)),
        'subseqdtw3': np.array([1,1,1,2,2,1]).reshape((-1,2)),
        'nwtw': 0, # transitions are specified in NWTW algorithm
        'flexdtw': np.array([1,1,1,2,2,1]).reshape((-1,2)), 
        'parflex': np.array([1,1,1,2,2,1]).reshape((-1,2))
        }
weights = {'dtw1': np.array([2,3,3]),
          'dtw2': np.array([1,1,1]),
          'dtw3': np.array([1,2,2]),
          'subseqdtw1': np.array([1,1,2]),
          'subseqdtw2': np.array([2,3,3]),
          'subseqdtw3': np.array([1,2,2]),
          'nwtw': 0, # weights are specified in NWTW algorithm
          'flexdtw': np.array([1.25,3,3]),
          'parflex': np.array([1.25,3.0,3.0])
          }
other_params = {
                'flexdtw': {'beta': 0.1}, 
                'parflex': {'beta': 0.1}
               }


# # Benchmarks

# In[11]:


def get_outfile(outdir, benchmark, system, queryid):
    outpath = (outdir / benchmark / system)
    outpath.mkdir(parents=True, exist_ok=True)
    outfile = (outpath / queryid).with_suffix('.pkl')
    return outfile


# In[12]:


def plot_normalized_global_edge_cost(D_chunks, L_chunks, num_chunks_1, num_chunks_2):
    """
    Extracts and plots the normalized accumulated cost (Cost / Length) 
    along the global top edge and global right edge.
    
    Traversal order: Top-Left -> Top-Right -> Bottom-Right.
    
    Parameters:
    -----------
    D_chunks : list of list of dicts of lists (Accumulated Cost)
    L_chunks : list of list of dicts of lists (Accumulated Path Length)
    num_chunks_1, num_chunks_2 : int, int (Number of chunks in each dimension)
    """
 
    # Indices for the Global Boundary
    i_top = num_chunks_1 - 1
    j_right = num_chunks_2 - 1
    
    global_edge_data = [] # List of tuples: (normalized_cost, edge_type, global_position)
    
    # 1. Traversal: Global Top Edge (from Top-Left to Top-Right)
    # This corresponds to the top edge (edge_type=0) of the top row of chunks (i_top)
    
    #print("Extracting Global Top Edge (Left to Right)...")
    for j in range(num_chunks_2):
        chunk_key = (i_top, j)
        #print(chunk_key)
        
            
        costs = np.array(D_chunks[i_top][j][0][1:]) # edge_type 0
        lengths = np.array(L_chunks[i_top][j][0][1:]) # edge_type 0
        #print(len(costs))
        #print("0 and last costs, lengths ", costs[0], costs[-1], lengths[0], lengths[-1])
         
        #print('hereab')
        # Calculate normalized cost for finite values
        valid_indices = np.isfinite(costs) & (lengths > 0)
    
        
        normalized_costs = np.full_like(costs, np.nan, dtype=float)
        normalized_costs[valid_indices] = costs[valid_indices] / lengths[valid_indices]
        
        # Extend the data list
        ind = 0 
        #print("ABT TO #print!!!!")
        for cost in normalized_costs:
            # if cost <= 0.1: 
                # print(ind, cost)
            ind += 1
            
            global_edge_data.append((cost, 'Top'))
            
    # 2. Traversal: Global Right Edge (from Top-Right to Bottom-Right)
    # This corresponds to the right edge (edge_type=1) of the right column of chunks (j_right)
    
    #print("Extracting Global Right Edge (Top to Bottom)...")
    # Iterate DOWN from the second-to-last row (i_top - 1) to row 0
    
        
    for i in range(num_chunks_1 - 1, -1, -1):
        chunk_key = (i, j_right)
        #print(chunk_key)
        
            
        costs = np.array(D_chunks[i][j_right][1][1:]) # edge_type 1
        lengths = np.array(L_chunks[i][j_right][1][1:]) # edge_type 1
        #print("0 and last costs, lengths ", costs[0], costs[-1], lengths[0], lengths[-1])
        #print(len(costs))
        
        # Calculate normalized cost for finite values
        valid_indices = np.isfinite(costs) & (lengths > 0)
        
        normalized_costs = np.full_like(costs, np.nan, dtype=float)
        normalized_costs[valid_indices] = costs[valid_indices] / lengths[valid_indices]
        
        # Right edge positions are indexed from local row 0 (bottom) to local row R-1 (top).
        # We need to reverse this list to traverse from Top-Right (R-1) to Bottom-Right (0).
        reversed_normalized_costs = normalized_costs[::-1]
        
        # Extend the data list
        for cost in reversed_normalized_costs:
            global_edge_data.append((cost, 'Right'))


    # --- 3. Plotting the Lineplot ---
    
    #print("total data: ", len(global_edge_data))
    costs = [d[0] for d in global_edge_data]
    edge_types = [d[1] for d in global_edge_data]
    

    plt.figure(figsize=(15, 6))
    
    # Plotting using the color based on edge type
    top_indices = [i for i, type in enumerate(edge_types) if type == 'Top']
    right_indices = [i for i, type in enumerate(edge_types) if type == 'Right']
    
    # Extract segments for separate colors
    top_costs = [costs[i] for i, e in enumerate(edge_types) if e == "Top"]
    right_costs = [costs[i] for i, e in enumerate(edge_types) if e == "Right"]

    # assign negative → 0 → positive axis
    top_x = np.arange(-len(top_costs), 0)            # e.g. -120 … -1
    right_x = np.arange(1, len(right_costs)+1) 
    junction_x = 0

    # Plot Top Edge (Blue)
    plt.plot(top_x, top_costs, label='Global Top Edge', color='C0', linewidth=2)
    plt.scatter(top_x, top_costs, color='C0', s=10)

    plt.plot(right_x, right_costs, label='Global Right Edge', color='C3', linewidth=2)
    plt.scatter(right_x, right_costs, color='C3', s=10)

    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Corner (0)')

    
    # Add a vertical line to show the junction
    junction_x = 0
    plt.axvline(x=junction_x, color='gray', linestyle='--', alpha=0.7, 
                label='Top/Right Junction')


    plt.title(f"Normalized Accumulated Cost ($\mathbf{{Cost / Length}}$) along Global Edge", fontsize=14)
    plt.xlabel(f"Global Edge Position Index (Total Points: {len(costs)})", fontsize=12)
    plt.ylabel("Normalized Cost (Accumulated Cost / Path Length)", fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
    


# In[13]:


def chunk_flexdtw(C, L, steps=None, weights=None, buffer=1):
    """
    Break cost matrix C into overlapping chunks and run FlexDTW on each.
    Ensures exactly 1-cell overlap between chunks. Last chunks may be smaller
    than L×L to maintain the 1-cell overlap constraint.
    
    Now stores flexible hop information for each chunk to handle boundary truncation.
    
    Parameters:
    -----------
    C : ndarray
        Cost matrix of shape (L1, L2)
    L : int
        Standard chunk size
    steps : list, optional
        Step patterns for DTW
    weights : list, optional
        Weights for each step pattern
    buffer : int
        Buffer parameter for FlexDTW
        
    Returns:
    --------
    chunks_dict : dict
        Dictionary with chunk data, including 'bounds' and 'hop' for each chunk
    L : int
        Chunk size
    n_chunks_1, n_chunks_2 : int
        Number of chunks in each dimension
    """
    import math
    
    if steps is None:
        steps = [(1,1), (1,2), (2,1)]
    if weights is None:
        weights = [2, 3, 3]
    
    L1, L2 = C.shape
    #print(f"Matrix shape: {L1} × {L2}")
    hop = L - 1  # Standard overlap of 1
    
    # Compute number of chunks along each axis
    n_chunks_1 = math.ceil((L1 - 1) / hop)
    n_chunks_2 = math.ceil((L2 - 1) / hop)
    
    chunks_dict = {}
    
    for i in range(n_chunks_1):
        for j in range(n_chunks_2):
            # Standard start positions with hop size
            start_1 = i * hop
            start_2 = j * hop
            
            # Standard end positions
            end_1 = start_1 + L
            end_2 = start_2 + L
            
            # For boundary chunks, don't shift start - just truncate end
            # This ensures we always have exactly 1-frame overlap
            if end_1 > L1:
                end_1 = L1
            
            if end_2 > L2:
                end_2 = L2
            
            # Extract chunk
            C_chunk = C[int(start_1):int(end_1), int(start_2):int(end_2)]
            
            # Run FlexDTW on this chunk
            # Note: You'll need to import or have FlexDTW available
            # Assuming FlexDTW.flexdtw returns (best_cost, wp, D, P, B, debug)
            try:
                import FlexDTW
                best_cost, wp, D, P, B, debug = FlexDTW.flexdtw(
                    C_chunk, 
                    steps=steps, 
                    weights=weights, 
                    buffer=1
                )
            except ImportError:
                # Placeholder if FlexDTW not available
                #print("Warning: FlexDTW not imported, using placeholder values")
                best_cost = 0
                wp = []
                D = np.zeros_like(C_chunk)
                P = np.zeros_like(C_chunk)
                B = np.zeros_like(C_chunk)
                debug = {}
            
            # Calculate actual hop for this chunk
            # For boundary chunks, the hop to the next chunk may be different
            actual_hop_1 = hop if end_1 < L1 else (L1 - start_1)
            actual_hop_2 = hop if end_2 < L2 else (L2 - start_2)
            
            chunks_dict[(i, j)] = {
                'C': C_chunk,
                'D': D,
                'S': P,  # Start positions (signed encoding)
                'B': B,  # Backpointer matrix
                'debug': debug,
                'best_cost': best_cost,
                'wp': wp,
                'bounds': (start_1, end_1, start_2, end_2),
                'hop': (actual_hop_1, actual_hop_2),  # Store flexible hop sizes
                'shape': C_chunk.shape
            }
            
            #print(f"Chunk ({i},{j}): [{start_1}:{end_1}, {start_2}:{end_2}], "
                #   f"shape={C_chunk.shape}, hop=({actual_hop_1},{actual_hop_2}), "
                #   f"best_cost={best_cost:.4f}")
   
    return chunks_dict, L, n_chunks_1, n_chunks_2


# In[14]:


def align_system(system, F1, F2, outfile):
    L = 4000
    subseq = 'subseq' in system
    
    if system == 'parflex':
        L1 = F1.shape[1]
        L2 = F2.shape[1]
        buffer_global = min(L1, L2) * (1 - (1 - other_params[system]['beta']) * min(L1,L2) / max(L1, L2))
        #print(buffer_global)
        C = 1 - FlexDTW.L2norm(F1).T @ FlexDTW.L2norm(F2)
        L1, L2 = C.shape
         
        # Define chunk size L
        L = L
        
        # Run chunked FlexDTW
        chunks_dict, L, n_chunks_1, n_chunks_2 = chunk_flexdtw(
            C, 
            L=L, 
            steps=[(1,1), (1,2), (2,1)], 
            weights=[1.5, 3.0, 3.0], 
            buffer=1
        ) 
        
        tiled_result = convert_chunks_to_tiled_result(
        chunks_dict, L, n_chunks_1, n_chunks_2, C
        ) 
    
        D_chunks, L_chunks, valid_bottom_starts, valid_left_starts = chunked_flexdtw(chunks_dict, L, n_chunks_1, n_chunks_2)
        # plot_normalized_global_edge_cost(D_chunks, L_chunks, n_chunks_1, n_chunks_2)
        
        r = stage_2_backtrace(tiled_result, chunks_dict, D_chunks, L_chunks, L1, L2, 4000,
                              valid_bottom_starts, valid_left_starts, 
                             buffer_stage2=200, top_k=1)
 
        
        # (tiled_result, chunks_dict, D_chunks, L_chunks, L1, L2, 
        #                             L_block=4000, buffer_stage2=buffer_global, top_k=1)
        wp, best_cost = r['stitched_wp'], r['best_cost']
        wp = wp.T

        # Save stitched warping path so caller can consume it (consistent behavior)
        try:
            outp = Path(outfile)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, 'wb') as f:
                pickle.dump(wp, f)
        except Exception:
            pass

        # free large matrices
        try:
            del C
        except Exception:
            pass
        gc.collect()

        return D_chunks,  L_chunks, valid_bottom_starts, valid_left_starts, chunks_dict, tiled_result
         
        
    elif system == 'flexdtw':
        L1 = F1.shape[1]
        L2 = F2.shape[1]
        buffer = min(L1, L2) * (1 - (1 - other_params[system]['beta']) * min(L1,L2) / max(L1, L2))
        C = 1 - FlexDTW.L2norm(F1).T @ FlexDTW.L2norm(F2) # cos distance metric 
        # print(steps[system], weights[system], buffer)
        best_cost, wp, D, P, B, debug = FlexDTW.flexdtw(C, steps=steps[system], weights=weights[system], buffer=buffer)
        # print("flexdtw", wp) 
        
    elif system == 'nwtw':
        downsample = 1
        C = 1 - NWTW.L2norm(F1)[:,0::downsample].T @ NWTW.L2norm(F2)[:,0::downsample] # cos distance metric
        optcost, wp, D, B = NWTW.NWTW_faster(C, gamma=0.346)
    else:
        downsample = 1
        if subseq and (F2.shape[1] < F1.shape[1]):
            C = 1 - DTW.L2norm(F2)[:,0::downsample].T @ DTW.L2norm(F1)[:,0::downsample] # cos distance metric
            wp = DTW.alignDTW(C, steps=steps[system], weights=weights[system], downsample=downsample, outfile=outfile, subseq=subseq)
            wp = wp[::-1,:]
        else:
            C = 1 - DTW.L2norm(F1)[:,0::downsample].T @ DTW.L2norm(F2)[:,0::downsample] # cos distance metric
            wp = DTW.alignDTW(C, steps=steps[system], weights=weights[system], downsample=downsample, outfile=outfile, subseq=subseq)
            
    if wp is not None:
        try:
            outp = Path(outfile)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, 'wb') as f:
                pickle.dump(wp, f)
        except Exception:
            pass
            
    
    


# In[ ]:





# In[15]:


import numpy as np

def convert_edge_to_local(edge_type, position, chunk_shape):
    """Convert edge representation to local chunk coordinates."""
    if edge_type == 0:  # Top edge
        return chunk_shape[0] - 1, position  # Last row of actual chunk
    else:  # Right edge (edge_type == 1)
        return position, chunk_shape[1] - 1  # Last column of actual chunk

def convert_local_to_global(chunk_i, chunk_j, local_row, local_col, chunks_dict):
    """Convert local chunk coordinates to global matrix coordinates."""
    start_1, _, start_2, _ = chunks_dict[(chunk_i, chunk_j)]['bounds']
    global_row = start_1 + local_row
    global_col = start_2 + local_col
    return global_row, global_col

def get_starting_point(S_single, end_row, end_col):
    """Get the starting point for a path ending at (end_row, end_col)."""
    val = S_single[end_row, end_col]
    if val > 0:
        return 0, int(val)  # start on bottom edge
    elif val < 0:
        return abs(int(val)), 0  # start on left edge
    else:
        return 0, 0  # default / fallback

def is_on_bottom_edge(start_row, start_col, chunk_shape):
    """Check if starting point is on bottom edge."""
    return start_row == 0

def is_on_left_edge(start_row, start_col, chunk_shape):
    """Check if starting point is on left edge."""
    return start_col == 0

def global_to_prev_chunk_edge(global_row, global_col, prev_chunk_i, prev_chunk_j, chunks_dict, L):
    """Convert global coordinates to edge position in previous chunk."""
    start_1, _, start_2, _ = chunks_dict[(prev_chunk_i, prev_chunk_j)]['bounds']
    local_row = global_row - start_1
    local_col = global_col - start_2
    
    prev_chunk_shape = chunks_dict[(prev_chunk_i, prev_chunk_j)]['D'].shape
    if local_row == prev_chunk_shape[0] - 1:  # On top edge
        return 0, local_col
    elif local_col == prev_chunk_shape[1] - 1:  # On right edge
        return 1, local_row
    else:
        raise ValueError(f"Coordinates ({local_row}, {local_col}) not on edge of previous chunk")

def get_edge_length(chunk_shape, edge_type):
    """Returns the actual length of an edge for a given chunk shape."""
    if edge_type == 0:  # top edge
        return chunk_shape[1]  # width
    else:  # right edge (edge_type == 1)
        return chunk_shape[0]  # height

# ============================================================================
# STAGE 1: Extract Valid Starting Points
# ============================================================================

def extract_valid_starting_points(chunks_dict, num_chunks_1, num_chunks_2):
    """
    Stage 1: For each chunk, find all valid starting points by scanning S_single.
    
    Returns:
    --------
    valid_bottom_starts : list of list of sets
        valid_bottom_starts[i][j] = set of column indices on bottom edge
    valid_left_starts : list of list of sets
        valid_left_starts[i][j] = set of row indices on left edge
    """
    valid_bottom_starts = [[set() for _ in range(num_chunks_2)] for _ in range(num_chunks_1)]
    valid_left_starts = [[set() for _ in range(num_chunks_2)] for _ in range(num_chunks_1)]
    
    for i in range(num_chunks_1):
        for j in range(num_chunks_2):
            if (i, j) not in chunks_dict:
                continue
            
            S_single = chunks_dict[(i, j)]['S']
            chunk_shape = S_single.shape
            
            # Scan rightmost column (all paths ending on right edge)
            for r in range(chunk_shape[0]):
                c = chunk_shape[1] - 1  # rightmost column
                start_row, start_col = get_starting_point(S_single, r, c)
                
                if is_on_bottom_edge(start_row, start_col, chunk_shape):
                    valid_bottom_starts[i][j].add(start_col)
                
                if is_on_left_edge(start_row, start_col, chunk_shape):
                    valid_left_starts[i][j].add(start_row)
            
            # Scan bottom row (all paths ending on bottom edge)
            for c in range(chunk_shape[1]):
                r = chunk_shape[0] - 1  # bottom row
                start_row, start_col = get_starting_point(S_single, r, c)
                
                if is_on_bottom_edge(start_row, start_col, chunk_shape):
                    valid_bottom_starts[i][j].add(start_col)
                
                if is_on_left_edge(start_row, start_col, chunk_shape):
                    valid_left_starts[i][j].add(start_row)
    
    return valid_bottom_starts, valid_left_starts

# ============================================================================
# STAGE 2: Sparse Initialization and DP
# ============================================================================

def initialize_chunks(chunks_dict, num_chunks_1, num_chunks_2, L, 
                             valid_bottom_starts, valid_left_starts):
    """
    Initialize D_chunks and L_chunks with sparse computation.
    First row and first column are fully computed.
    """
    # Initialize storage
    D_chunks = [[{0: [], 1: []} for _ in range(num_chunks_2)] for _ in range(num_chunks_1)]
    L_chunks = [[{0: [], 1: []} for _ in range(num_chunks_2)] for _ in range(num_chunks_1)]
    
    # ========================================================================
    # Initialize chunk (0, 0) - FULL COMPUTATION
    # ========================================================================
    D_single_00 = chunks_dict[(0, 0)]['D']
    S_single_00 = chunks_dict[(0, 0)]['S']
    
    for edge_type in range(2):
        edge_len = get_edge_length(D_single_00.shape, edge_type)
        D_chunks[0][0][edge_type] = [np.inf] * edge_len
        L_chunks[0][0][edge_type] = [np.inf] * edge_len
        
        for position in range(edge_len):
            local_row, local_col = convert_edge_to_local(edge_type, position, D_single_00.shape)
            if local_row < D_single_00.shape[0] and local_col < D_single_00.shape[1]:
                start_row, start_col = get_starting_point(S_single_00, local_row, local_col)
                if is_on_bottom_edge(start_row, start_col, D_single_00.shape) or \
                   is_on_left_edge(start_row, start_col, D_single_00.shape):
                    D_chunks[0][0][edge_type][position] = D_single_00[local_row, local_col]
                    path_length = abs(local_row - start_row) + abs(local_col - start_col)
                    L_chunks[0][0][edge_type][position] = path_length
    
    # ========================================================================
    # Initialize first row (0, j) - FULL COMPUTATION
    # ========================================================================
    for j in range(1, num_chunks_2):
        if (0, j) not in chunks_dict:
            continue
        
        D_single = chunks_dict[(0, j)]['D']
        S_single = chunks_dict[(0, j)]['S']
        C_chunk = chunks_dict[(0, j)]['C'] if 'C' in chunks_dict[(0, j)] else None
        
        # Edge continuity
        D_chunks[0][j][0] = [D_chunks[0][j-1][0][-1]] + [np.inf] * (get_edge_length(D_single.shape, 0) - 1)
        L_chunks[0][j][0] = [L_chunks[0][j-1][0][-1]] + [np.inf] * (get_edge_length(D_single.shape, 0) - 1)
        
        edge_len_right = get_edge_length(D_single.shape, 1)
        D_chunks[0][j][1] = [np.inf] * edge_len_right
        L_chunks[0][j][1] = [np.inf] * edge_len_right
        
        # Process ALL positions (full computation for first row)
        for edge_type in range(2):
            edge_len = get_edge_length(D_single.shape, edge_type)
            for position in range(edge_len):
                if edge_type == 0 and position == 0:
                    continue  # Already set by continuity
                
                local_row, local_col = convert_edge_to_local(edge_type, position, D_single.shape)
                if local_row >= D_single.shape[0] or local_col >= D_single.shape[1]:
                    continue
                
                start_row, start_col = get_starting_point(S_single, local_row, local_col)
                
                if is_on_bottom_edge(start_row, start_col, D_single.shape):
                    D_chunks[0][j][edge_type][position] = D_single[local_row, local_col]
                    path_length = abs(local_row - start_row) + abs(local_col - start_col)
                    L_chunks[0][j][edge_type][position] = path_length
                
                elif is_on_left_edge(start_row, start_col, D_single.shape):
                    global_start_row, global_start_col = convert_local_to_global(
                        0, j, start_row, start_col, chunks_dict
                    )
                    prev_edge_type, prev_position = global_to_prev_chunk_edge(
                        global_start_row, global_start_col, 0, j - 1, chunks_dict, L
                    )
                    
                    prev_edge_len = len(D_chunks[0][j-1][prev_edge_type])
                    if prev_position >= prev_edge_len:
                        continue
                    
                    prev_cost = D_chunks[0][j-1][prev_edge_type][prev_position]
                    if np.isfinite(prev_cost):
                        overlap_cost = C_chunk[start_row, 0] if C_chunk is not None else 0
                        D_chunks[0][j][edge_type][position] = D_single[local_row, local_col] + prev_cost - overlap_cost
                        
                        prev_length = L_chunks[0][j-1][prev_edge_type][prev_position]
                        curr_length = abs(local_row - start_row) + abs(local_col - start_col)
                        L_chunks[0][j][edge_type][position] = prev_length + curr_length
    
    # ========================================================================
    # Initialize first column (i, 0) - FULL COMPUTATION
    # ========================================================================
    for i in range(1, num_chunks_1):
        if (i, 0) not in chunks_dict:
            continue
        
        D_single = chunks_dict[(i, 0)]['D']
        S_single = chunks_dict[(i, 0)]['S']
        C_chunk = chunks_dict[(i, 0)]['C'] if 'C' in chunks_dict[(i, 0)] else None
        
        # Edge continuity
        D_chunks[i][0][1] = [D_chunks[i-1][0][1][-1]] + [np.inf] * (get_edge_length(D_single.shape, 1) - 1)
        L_chunks[i][0][1] = [L_chunks[i-1][0][1][-1]] + [np.inf] * (get_edge_length(D_single.shape, 1) - 1)
        
        edge_len_top = get_edge_length(D_single.shape, 0)
        D_chunks[i][0][0] = [np.inf] * edge_len_top
        L_chunks[i][0][0] = [np.inf] * edge_len_top
        
        # Process ALL positions (full computation for first column)
        for edge_type in range(2):
            edge_len = get_edge_length(D_single.shape, edge_type)
            for position in range(edge_len):
                if edge_type == 1 and position == 0:
                    continue  # Already set by continuity
                
                local_row, local_col = convert_edge_to_local(edge_type, position, D_single.shape)
                if local_row >= D_single.shape[0] or local_col >= D_single.shape[1]:
                    continue
                
                start_row, start_col = get_starting_point(S_single, local_row, local_col)
                
                if is_on_left_edge(start_row, start_col, D_single.shape):
                    D_chunks[i][0][edge_type][position] = D_single[local_row, local_col]
                    path_length = abs(local_row - start_row) + abs(local_col - start_col)
                    L_chunks[i][0][edge_type][position] = path_length
                
                elif is_on_bottom_edge(start_row, start_col, D_single.shape):
                    global_start_row, global_start_col = convert_local_to_global(
                        i, 0, start_row, start_col, chunks_dict
                    )
                    prev_edge_type, prev_position = global_to_prev_chunk_edge(
                        global_start_row, global_start_col, i - 1, 0, chunks_dict, L
                    )
                    
                    prev_edge_len = len(D_chunks[i-1][0][prev_edge_type])
                    if prev_position >= prev_edge_len:
                        continue
                    
                    prev_cost = D_chunks[i-1][0][prev_edge_type][prev_position]
                    if np.isfinite(prev_cost):
                        overlap_cost = C_chunk[0, start_col] if C_chunk is not None else 0
                        D_chunks[i][0][edge_type][position] = D_single[local_row, local_col] + prev_cost - overlap_cost
                        
                        prev_length = L_chunks[i-1][0][prev_edge_type][prev_position]
                        curr_length = abs(local_row - start_row) + abs(local_col - start_col)
                        L_chunks[i][0][edge_type][position] = prev_length + curr_length
    
    return D_chunks, L_chunks

def dp_fill_chunks(chunks_dict, D_chunks, L_chunks, num_chunks_1, num_chunks_2, L,
                          valid_bottom_starts, valid_left_starts):
    """
    Fill D_chunks and L_chunks for interior chunks using SPARSE computation.
    Only compute edge positions that are valid starting points for next chunks.
    """
    for i in range(1, num_chunks_1):
        for j in range(1, num_chunks_2):
            if (i, j) not in chunks_dict:
                continue
            
            D_single = chunks_dict[(i, j)]['D']
            S_single = chunks_dict[(i, j)]['S']
            C_chunk = chunks_dict[(i, j)]['C'] if 'C' in chunks_dict[(i, j)] else None
            
            # Determine which positions to compute
            # Top edge (edge_type=0): only positions needed by chunk (i+1, j)
            if i + 1 < num_chunks_1 and (i + 1, j) in chunks_dict:
                positions_to_compute_top = valid_bottom_starts[i + 1][j]
            else:
                # Last row or no chunk below - compute all
                positions_to_compute_top = set(range(get_edge_length(D_single.shape, 0)))
            
            # Right edge (edge_type=1): only positions needed by chunk (i, j+1)
            if j + 1 < num_chunks_2 and (i, j + 1) in chunks_dict:
                positions_to_compute_right = valid_left_starts[i][j + 1]
            else:
                # Last column or no chunk to right - compute all
                positions_to_compute_right = set(range(get_edge_length(D_single.shape, 1)))
            
            # Initialize edge arrays
            D_chunks[i][j][0] = [np.inf] * get_edge_length(D_single.shape, 0)
            L_chunks[i][j][0] = [np.inf] * get_edge_length(D_single.shape, 0)
            D_chunks[i][j][1] = [np.inf] * get_edge_length(D_single.shape, 1)
            L_chunks[i][j][1] = [np.inf] * get_edge_length(D_single.shape, 1)
            
            # Process top edge (edge_type=0) - SPARSE
            for position in positions_to_compute_top:
                # Handle continuity for position 0
                if position == 0:
                    if j > 0:
                        left_edge_len = len(D_chunks[i][j-1][0])
                        if left_edge_len > 0:
                            rightmost_pos = left_edge_len - 1
                            left_cost = D_chunks[i][j-1][0][rightmost_pos]
                            left_length = L_chunks[i][j-1][0][rightmost_pos]
                            if np.isfinite(left_cost):
                                D_chunks[i][j][0][position] = left_cost
                                L_chunks[i][j][0][position] = left_length
                                continue
                
                _compute_edge_position(i, j, 0, position, D_single, S_single, C_chunk,
                                      D_chunks, L_chunks, chunks_dict, L)
            
            # Process right edge (edge_type=1) - SPARSE
            for position in positions_to_compute_right:
                # Handle continuity for position 0
                if position == 0:
                    if i > 0:
                        top_edge_len = len(D_chunks[i-1][j][1])
                        if top_edge_len > 0:
                            bottommost_pos = top_edge_len - 1
                            top_cost = D_chunks[i-1][j][1][bottommost_pos]
                            top_length = L_chunks[i-1][j][1][bottommost_pos]
                            if np.isfinite(top_cost):
                                D_chunks[i][j][1][position] = top_cost
                                L_chunks[i][j][1][position] = top_length
                                continue
                
                _compute_edge_position(i, j, 1, position, D_single, S_single, C_chunk,
                                      D_chunks, L_chunks, chunks_dict, L)
    
    return D_chunks, L_chunks

def _compute_edge_position(i, j, edge_type, position, D_single, S_single, C_chunk,
                          D_chunks, L_chunks, chunks_dict, L):
    """Helper function to compute a single edge position."""
    local_row, local_col = convert_edge_to_local(edge_type, position, D_single.shape)
    
    if local_row >= D_single.shape[0] or local_col >= D_single.shape[1]:
        return
    
    start_row, start_col = get_starting_point(S_single, local_row, local_col)
    
    # Determine previous chunk
    if is_on_bottom_edge(start_row, start_col, D_single.shape):
        prev_i, prev_j = i - 1, j
    elif is_on_left_edge(start_row, start_col, D_single.shape):
        prev_i, prev_j = i, j - 1
    else:
        # Path started within this chunk
        D_chunks[i][j][edge_type][position] = D_single[local_row, local_col]
        path_length = abs(local_row - start_row) + abs(local_col - start_col)
        L_chunks[i][j][edge_type][position] = path_length
        return
    
    # Get global coordinates of starting point
    global_start_row, global_start_col = convert_local_to_global(
        i, j, start_row, start_col, chunks_dict
    )
    
    # Map to edge position in previous chunk
    prev_edge_type, prev_position = global_to_prev_chunk_edge(
        global_start_row, global_start_col, prev_i, prev_j, chunks_dict, L
    )
    
    # Check bounds
    prev_edge_len = len(D_chunks[prev_i][prev_j][prev_edge_type])
    if prev_position >= prev_edge_len:
        return
    
    # Get accumulated cost from previous chunk
    prev_cost = D_chunks[prev_i][prev_j][prev_edge_type][prev_position]
    prev_length = L_chunks[prev_i][prev_j][prev_edge_type][prev_position]
    
    if not np.isfinite(prev_cost):
        return
    
    # Compute contribution from current chunk
    first_cell_cost = C_chunk[start_row, start_col] if C_chunk is not None else 0
    curr_cost_contribution = D_single[local_row, local_col] - first_cell_cost
    curr_length = abs(local_row - start_row) + abs(local_col - start_col)
    
    # Propagate forward
    D_chunks[i][j][edge_type][position] = prev_cost + curr_cost_contribution
    L_chunks[i][j][edge_type][position] = prev_length + curr_length

# ============================================================================
# Main Function
# ============================================================================

def chunked_flexdtw(chunks_dict, L, num_chunks_1, num_chunks_2):
    """
    Run chunked FlexDTW with sparse computation optimization.
    
    Stage 1: Extract valid starting points from each chunk
    Stage 2: Only compute edge values that will be used by subsequent chunks
    
    Parameters:
    -----------
    chunks_dict : dict
        Dictionary with chunk data
    L : int
        Standard chunk size
    num_chunks_1, num_chunks_2 : int
        Number of chunks in each dimension
    
    Returns:
    --------
    D_chunks, L_chunks : Filled cost and length tensors
    valid_bottom_starts, valid_left_starts : Sparse starting point sets
    """
    # print("Stage 1: Extracting valid starting points...")
    valid_bottom_starts, valid_left_starts = extract_valid_starting_points(
        chunks_dict, num_chunks_1, num_chunks_2
    )
    
    # Print statistics
    total_valid = 0
    total_possible = 0
    for i in range(num_chunks_1):
        for j in range(num_chunks_2):
            if (i, j) in chunks_dict:
                chunk_shape = chunks_dict[(i, j)]['D'].shape
                # print( len(valid_bottom_starts[i][j]) + len(valid_left_starts[i][j]),chunk_shape[0] + chunk_shape[1], (len(valid_bottom_starts[i][j]) + len(valid_left_starts[i][j]))/(chunk_shape[0] + chunk_shape[1]) ) 
                total_valid += len(valid_bottom_starts[i][j]) + len(valid_left_starts[i][j])
                total_possible += chunk_shape[0] + chunk_shape[1]
    
    sparsity = 100 * (1 - total_valid / total_possible) if total_possible > 0 else 0
    # print(f"  Valid starting points: {total_valid} / {total_possible} ({sparsity:.1f}% sparse)")
    
    # print("Stage 2: Sparse initialization...")
    D_chunks, L_chunks = initialize_chunks(
        chunks_dict, num_chunks_1, num_chunks_2, L,
        valid_bottom_starts, valid_left_starts
    )
    
    # print("Stage 3: Sparse dynamic programming...")
    D_chunks, L_chunks = dp_fill_chunks(
        chunks_dict, D_chunks, L_chunks, num_chunks_1, num_chunks_2, L,
        valid_bottom_starts, valid_left_starts
    )
    
    # print("Done!")
    return D_chunks, L_chunks, valid_bottom_starts, valid_left_starts


# ============================================================================
# Stage 2 Backtrace - Compatible with Sparse Computation
# ============================================================================

def stage_2_backtrace(tiled_result, all_blocks, D_chunks, L_chunks, L1, L2,
                             L_block, valid_bottom_starts, valid_left_starts,
                             buffer_stage2=200, top_k=1):
    """
    Stage 2 backtrace compatible with sparse computation structure.
    Only scans endpoints that were actually computed (have finite costs).
    
    Parameters:
    -----------
    tiled_result : dict
        Result from stage 1 tiling
    all_blocks : dict
        Dictionary of blocks with keys (i, j)
    D_chunks, L_chunks : list of list of dicts of lists
        Sparse data structure: [chunk_i][chunk_j][edge_type][position]
    L1, L2 : int, int
        Global matrix dimensions
    L_block : int
        Block/chunk size
    valid_bottom_starts, valid_left_starts : list of list of sets
        Sparse starting point information from Stage 1
    buffer_stage2 : int
        Buffer size for edge scanning
    top_k : int
        Number of top endpoints to track
        
    Returns:
    --------
    dict with backtrace results
    """
    
    INF = 1e9
    n_row = len(D_chunks)
    n_col = len(D_chunks[0]) if n_row > 0 else 0
    
    def edge_to_local(edge, idx, rows, cols):
        """Convert edge representation to local coordinates."""
        if edge == 0:  # top
            return rows - 1, idx
        else:  # right
            return idx, cols - 1
    
    def backtrace_within_chunk(B_single, steps, start_r, start_c, end_r, end_c, 
                                global_r_offset, global_c_offset):
        """
        Backtrace from (end_r, end_c) back to (start_r, start_c) within a chunk.
        Returns path in GLOBAL coordinates, in END → START order.
        """
        path = []
        r, c = end_r, end_c
        max_iters = B_single.shape[0] * B_single.shape[1]
        iters = 0

        while iters < max_iters:
            path.append((r + global_r_offset, c + global_c_offset))

            if r == start_r and c == start_c:
                if len(path) == 0 or path[-1] != (start_r + global_r_offset, start_c + global_c_offset):
                    path.append((start_r + global_r_offset, start_c + global_c_offset))
                break

            step_idx = int(B_single[r, c])
            if step_idx < 0 or step_idx >= len(steps):
                if (r != start_r or c != start_c):
                    path.append((start_r + global_r_offset, start_c + global_c_offset))
                break

            dr, dc = steps[step_idx]
            prev_r = r - dr
            prev_c = c - dc

            if (prev_r < 0 or prev_c < 0 or 
                prev_r >= B_single.shape[0] or prev_c >= B_single.shape[1]):
                if (r != start_r or c != start_c):
                    path.append((start_r + global_r_offset, start_c + global_c_offset))
                break

            r, c = prev_r, prev_c
            iters += 1

        if iters >= max_iters:
            if (r != start_r or c != start_c):
                path.append((start_r + global_r_offset, start_c + global_c_offset))

        return path

    def backtrace_and_stitch(start_i, start_j, start_edge, start_idx):
        """
        Backtrack from a specific edge endpoint and stitch across chunks.
        Returns the GLOBAL path in START → END order.
        """
        path = []
        cur_i, cur_j, cur_edge, cur_idx = start_i, start_j, start_edge, start_idx
        steps = tiled_result['stage1_params']['steps']
        visited_chunks = set()
        stop_reason = None
        iteration = 0

        while True:
            iteration += 1
            chunk_key = (cur_i, cur_j, cur_edge, cur_idx)

            if chunk_key in visited_chunks:
                stop_reason = f"loop at chunk {chunk_key}"
                break
            visited_chunks.add(chunk_key)

            if (cur_i, cur_j) not in all_blocks:
                stop_reason = f"missing chunk ({cur_i},{cur_j})"
                break

            b = all_blocks[(cur_i, cur_j)]
            rows, cols = b['shape']
            r_start, r_end, c_start, c_end = b['bounds']

            end_r, end_c = edge_to_local(cur_edge, cur_idx, rows, cols)
            S_val = b['S'][end_r, end_c]

            if S_val >= 0:
                start_r = 0
                start_c = int(S_val)
            else:
                start_r = int(-S_val)
                start_c = 0

            chunk_path = backtrace_within_chunk(
                b['B'], steps, start_r, start_c, end_r, end_c, r_start, c_start
            )

            for pt in chunk_path:
                if not path or path[-1] != pt:
                    path.append(pt)

            g_start_row = r_start + start_r
            g_start_col = c_start + start_c

            if g_start_row == 0:
                stop_reason = f"hit bottom edge at global row 0, col {g_start_col}"
                break
            if g_start_col == 0:
                stop_reason = f"hit left edge at global row {g_start_row}, col 0"
                break

            if S_val >= 0:
                prev_i, prev_j = cur_i - 1, cur_j
                prev_edge = 0
            else:
                prev_i, prev_j = cur_i, cur_j - 1
                prev_edge = 1

            if (prev_i, prev_j) not in all_blocks:
                stop_reason = f"previous chunk ({prev_i},{prev_j}) missing"
                break

            prev_b = all_blocks[(prev_i, prev_j)]
            prev_r_start, prev_r_end, prev_c_start, prev_c_end = prev_b['bounds']
            prev_rows, prev_cols = prev_b['shape']

            prev_lr = g_start_row - prev_r_start
            prev_lc = g_start_col - prev_c_start

            prev_idx = prev_lc if prev_edge == 0 else prev_lr
            max_prev_idx = prev_cols if prev_edge == 0 else prev_rows

            if prev_idx < 0 or prev_idx >= max_prev_idx:
                stop_reason = f"prev_idx out of bounds({prev_idx}/{max_prev_idx})"
                break

            cur_i, cur_j, cur_edge, cur_idx = prev_i, prev_j, prev_edge, prev_idx

            if iteration > 100:
                stop_reason = f"iteration limit ({iteration})"
                break

        path = path[::-1]
        return path

    def get_valid_edge_positions(chunk_i, chunk_j, edge_type):
        """
        Get all valid (computed) positions for a given chunk edge.
        Returns list of (position, D_val, L_val) tuples for positions with finite costs.
        """
        valid_positions = []
        
        if chunk_i >= len(D_chunks) or chunk_j >= len(D_chunks[chunk_i]):
            return valid_positions
        
        if edge_type not in D_chunks[chunk_i][chunk_j]:
            return valid_positions
        
        edge_len = len(D_chunks[chunk_i][chunk_j][edge_type])
        
        for position in range(edge_len):
            D_val = D_chunks[chunk_i][chunk_j][edge_type][position]
            L_val = L_chunks[chunk_i][chunk_j][edge_type][position]
            
            if np.isfinite(D_val) and D_val < INF and L_val > 0:
                valid_positions.append((position, D_val, L_val))
        
        return valid_positions

    def global_to_chunk_edge(g_row, g_col):
        """
        Given a GLOBAL (g_row, g_col), find corresponding chunk, edge, and position.
        Returns (chunk_i, chunk_j, edge_type, position) or None.
        Only returns if position has finite cost (sparse validation).
        """
        for (bi, bj), b in all_blocks.items():
            r_start, r_end, c_start, c_end = b['bounds']

            if r_start <= g_row < r_end and c_start <= g_col < c_end:
                rows, cols = b['shape']
                local_r = g_row - r_start
                local_c = g_col - c_start

                edge_type = None
                position = None
                
                if local_r == rows - 1:  # On top edge
                    edge_type = 0
                    position = local_c
                elif local_c == cols - 1:  # On right edge
                    edge_type = 1
                    position = local_r
                
                if edge_type is not None:
                    # Sparse validation
                    if bi < len(D_chunks) and bj < len(D_chunks[bi]):
                        if edge_type in D_chunks[bi][bj]:
                            if position < len(D_chunks[bi][bj][edge_type]):
                                D_val = D_chunks[bi][bj][edge_type][position]
                                if np.isfinite(D_val) and D_val < INF:
                                    return (bi, bj, edge_type, position)
        
        return None

    # print("Scanning endpoints on global TOP and RIGHT edges (sparse)")

    top_D = np.full(L2, np.nan)
    top_L = np.zeros(L2)
    right_D = np.full(L1, np.nan)
    right_L = np.zeros(L1)

    best_overall_cost = float('inf')
    best_overall_end = None
    best_per_segment = {}
    candidate_endpoints = []

    # Scan TOP edge - iterate through chunks and only check valid positions
    valid_endpoints_found = 0
    buf = int(buffer_stage2)

    for chunk_i in range(n_row):
        for chunk_j in range(n_col):
            if (chunk_i, chunk_j) not in all_blocks:
                continue
            
            b = all_blocks[(chunk_i, chunk_j)]
            r_start, r_end, c_start, c_end = b['bounds']
            rows, cols = b['shape']
            
            # Check if this chunk has a top edge on the global top edge
            if r_start + rows - 1 == L1 - 1:
                # Get all valid positions on top edge (edge_type=0)
                valid_positions = get_valid_edge_positions(chunk_i, chunk_j, 0)
                
                for position, D_val, L_val in valid_positions:
                    g_col = c_start + position
                    g_row = L1 - 1
                    
                    if buf > 0 and g_col < buf:
                        continue
                    
                    valid_endpoints_found += 1
                    norm_cost = D_val / L_val
                    
                    top_D[g_col] = D_val
                    top_L[g_col] = L_val
                    
                    if norm_cost < best_overall_cost:
                        best_overall_cost = norm_cost
                        best_overall_end = (g_row, g_col, chunk_i, chunk_j, 0, position)
                    
                    seg_idx = int(g_col // L_block)
                    key = ('top', seg_idx)
                    
                    prev = best_per_segment.get(key)
                    if (prev is None) or (norm_cost < prev['norm_cost']):
                        best_per_segment[key] = {
                            'chunk_i': chunk_i,
                            'chunk_j': chunk_j,
                            'edge': 0,
                            'idx': position,
                            'norm_cost': norm_cost,
                            'global_coord': (g_row, g_col),
                            'segment': key,
                        }
                    
                    candidate_endpoints.append({
                        'g_row': g_row,
                        'g_col': g_col,
                        'chunk_i': chunk_i,
                        'chunk_j': chunk_j,
                        'edge': 0,
                        'idx': position,
                        'norm_cost': norm_cost
                    })

    # print(f"Found {valid_endpoints_found} valid endpoints on TOP global edge")

    # Scan RIGHT edge - iterate through chunks and only check valid positions
    valid_endpoints_found = 0

    for chunk_i in range(n_row):
        for chunk_j in range(n_col):
            if (chunk_i, chunk_j) not in all_blocks:
                continue
            
            b = all_blocks[(chunk_i, chunk_j)]
            r_start, r_end, c_start, c_end = b['bounds']
            rows, cols = b['shape']
            
            # Check if this chunk has a right edge on the global right edge
            if c_start + cols - 1 == L2 - 1:
                # Get all valid positions on right edge (edge_type=1)
                valid_positions = get_valid_edge_positions(chunk_i, chunk_j, 1)
                
                for position, D_val, L_val in valid_positions:
                    g_row = r_start + position
                    g_col = L2 - 1
                    
                    if buf > 0 and g_row < buf:
                        continue
                    
                    valid_endpoints_found += 1
                    norm_cost = D_val / L_val
                    
                    right_D[g_row] = D_val
                    right_L[g_row] = L_val
                    
                    if norm_cost < best_overall_cost:
                        best_overall_cost = norm_cost
                        best_overall_end = (g_row, g_col, chunk_i, chunk_j, 1, position)
                    
                    seg_idx = int(g_row // L_block)
                    key = ('right', seg_idx)
                    
                    prev = best_per_segment.get(key)
                    if (prev is None) or (norm_cost < prev['norm_cost']):
                        best_per_segment[key] = {
                            'chunk_i': chunk_i,
                            'chunk_j': chunk_j,
                            'edge': 1,
                            'idx': position,
                            'norm_cost': norm_cost,
                            'global_coord': (g_row, g_col),
                            'segment': key,
                        }
                    
                    candidate_endpoints.append({
                        'g_row': g_row,
                        'g_col': g_col,
                        'chunk_i': chunk_i,
                        'chunk_j': chunk_j,
                        'edge': 1,
                        'idx': position,
                        'norm_cost': norm_cost
                    })

    # print(f"Found {valid_endpoints_found} valid endpoints on RIGHT global edge")

    if not candidate_endpoints:
        raise ValueError("Stage 2: No valid endpoint found on global top/right edges.")

    candidate_endpoints_sorted = sorted(candidate_endpoints, key=lambda c: c['norm_cost'])

    best = candidate_endpoints_sorted[0]
    best_overall_cost = best['norm_cost'] 
    best_overall_end = (
        best['g_row'], best['g_col'], best['chunk_i'], 
        best['chunk_j'], best['edge'], best['idx']
    )

    g_row, g_col, best_i, best_j, best_edge, best_idx = best_overall_end

    K = min(top_k, len(candidate_endpoints_sorted))
    
    # Compute normalized arrays
    top_norm = np.full(L2, np.nan)
    right_norm = np.full(L1, np.nan)

    top_mask = (top_L > 0) & np.isfinite(top_D)
    right_mask = (right_L > 0) & np.isfinite(right_D)

    top_norm[top_mask] = top_D[top_mask] / top_L[top_mask]
    right_norm[right_mask] = right_D[right_mask] / right_L[right_mask]

    # Backtrace per segment
    paths_per_segment = {}

    for seg_key, meta in best_per_segment.items():
        ci = meta['chunk_i']
        cj = meta['chunk_j']
        ce = meta['edge']
        cidx = meta['idx']
        norm_cost = meta['norm_cost']
        g_row, g_col = meta['global_coord']

        path = backtrace_and_stitch(ci, cj, ce, cidx)
        path_len = len(path)

        paths_per_segment[seg_key] = {
            'endpoint': meta,
            'path': path
        }

    # Get best overall path
    stitched_wp = np.array([], dtype=int).reshape(0, 2)
    if best_overall_end is not None:
        g_row, g_col, best_i, best_j, best_edge, best_idx = best_overall_end
        best_path = backtrace_and_stitch(best_i, best_j, best_edge, best_idx)
        if best_path:
            stitched_wp = np.array(best_path, dtype=int)

    return {
        'D_chunks': D_chunks,
        'L_chunks': L_chunks,
        'best_cost': best_overall_cost,
        'best_end': best_overall_end,
        'stitched_wp': stitched_wp,
        'n_row': n_row,
        'n_col': n_col,
        'edge_summary': {
            'top':   {'D': top_D,   'L': top_L,   'norm': top_norm},
            'right': {'D': right_D, 'L': right_L, 'norm': right_norm},
        },
        'best_per_segment': best_per_segment,
        'paths_per_segment': paths_per_segment,
    }


# In[16]:


def convert_chunks_to_tiled_result(chunks_dict, L, n_chunks_1, n_chunks_2, C, stage1_params=None):
    """
    Convert your chunk_flexdtw output format to the format expected by 
    plot_parflex_with_chunk_S_background.
    
    Parameters:
    -----------
    chunks_dict : dict
        Dictionary from chunk_flexdtw with keys (i,j)
    L : int
        Chunk size (L_block)
    n_chunks_1, n_chunks_2 : int
        Number of chunks in each dimension
    C : ndarray
        Full cost matrix
    stage1_params : dict, optional
        FlexDTW parameters (steps, weights, buffer)
        
    Returns:
    --------
    tiled_result : dict
        Dictionary compatible with plot_parflex_with_chunk_S_background
    """
    L1, L2 = C.shape
    hop = L - 1  # Your code uses 1-frame overlap
    
    # Convert chunks dictionary to list of block dicts
    blocks = []
    
    for (i, j), chunk_data in chunks_dict.items():
        # Extract bounds
        start_1, end_1, start_2, end_2 = chunk_data['bounds']
        rows, cols = chunk_data['shape']
        
        # Get the warping path (ensure it's in the right format)
        wp_local = np.array(chunk_data['wp'])
        if wp_local.size == 0:
            #print(f"WARNING: Block ({i}, {j}) has empty path!")
            continue
            
        # Ensure wp_local is (N, 2)
        if wp_local.ndim == 2 and wp_local.shape[0] == 2:
            wp_local = wp_local.T
        
        # Calculate raw cost and path length
        C_chunk = chunk_data['C']
        raw_cost_blk = float(C_chunk[wp_local[:, 0], wp_local[:, 1]].sum())
        path_len_blk = int(np.abs(np.diff(wp_local, axis=0)).sum(axis=1).sum() + 1)
        
        # Map local path to global coordinates
        wp_global = np.column_stack([
            wp_local[:, 0] + start_1,
            wp_local[:, 1] + start_2
        ])
        
        block_dict = {
            'bi': i,
            'bj': j,
            'rows': (int(start_1), int(end_1)),
            'cols': (int(start_2), int(end_2)),
            'Ck_shape': (rows, cols),
            'best_cost': float(chunk_data['best_cost']),
            'wp_global': wp_global,
            'wp_local': wp_local.copy(),
            'raw_cost': raw_cost_blk,
            'path_len': path_len_blk,
            'D_single': chunk_data['D'],
            'B_single': chunk_data['B'],
            'S_single': chunk_data['S']  # Your 'S' becomes 'S_single'
        }
          
        blocks.append(block_dict)
    
    # Default stage1 parameters if not provided
    if stage1_params is None:
        stage1_params = {
            'steps': np.array([[1, 1], [1, 2], [2, 1]], dtype=int),
            'weights': np.array([1.5, 3.0, 3.0], dtype=float),
            'buffer': 1.0
        }
    
    # Build the tiled_result dictionary
    tiled_result = {
        'C_shape': (L1, L2),
        'L_block': L,
        'hop': hop,
        'n_row': n_chunks_1,
        'n_col': n_chunks_2,
        'blocks': blocks,
        'C': C,
        'stage1_params': stage1_params
    }
    
    return tiled_result


# In[17]:


import numpy as np
import plotly.graph_objects as go

def plot_parflex_with_chunk_S_background(tiled_result, C_global, flex_wp, parflex_res,
                                         L_div=4000, use_valid_edges_only=True):
    """
    Combined visualization:

    BACKGROUND:
      - 'S-like' start→edge line segments from chunk-level S_single,
        mapped into GLOBAL coordinates, lightly drawn.

    FOREGROUND:
      - Global FlexDTW path
      - ParFlex stitched global-best path
      - ParFlex best-per-segment paths

    Inputs:
      tiled_result : dict from Stage 1 (has 'blocks', 'L_block', 'hop', etc.)
      C_global     : full cost matrix (for shape only)
      flex_wp      : global FlexDTW path, shape (N,2) as (row=f1, col=f2)
      parflex_res  : dict returned by parflex_2a(...)
    """
    blocks = tiled_result['blocks']
    L_block = tiled_result['L_block']
    hop = tiled_result['hop']
    D_chunks = parflex_res['D_chunks']
    n_row, n_col = parflex_res['n_row'], parflex_res['n_col']

    L1, L2 = C_global.shape

    # ---------------- FIGURE SIZE ----------------
    base_px = 900
    max_side = max(L1, L2)
    scale = base_px / max_side
    fig_width = int(max(L2 * scale, 400))
    fig_height = int(max(L1 * scale, 400))

    fig = go.Figure()

    # ---------------- 1) CHUNK-S BACKGROUND (spiky lines) ----------------
    x_S, y_S = [], []
    INF = 1e17  # to interpret D_chunks

    def edge_to_local(edge, idx, rows, cols):
        # edge 0 = top edge in your convention → (rows-1, idx)
        # edge 1 = right edge → (idx, cols-1)
        return (rows - 1, idx) if edge == 0 else (idx, cols - 1)

    for b in blocks:
        i, j = b['bi'], b['bj']
        if i >= n_row or j >= n_col:
            continue

        S_single = b['S_single']
        rows, cols = b['Ck_shape']
        r_start, r_end = b['rows']
        c_start, c_end = b['cols']

        for edge in (0, 1):  # 0=top edge, 1=right edge
            edge_len = min(L_block, cols if edge == 0 else rows)

            for idx in range(edge_len):
                if use_valid_edges_only:
                    D_val = D_chunks[i][j][edge][idx]
                    if not np.isfinite(D_val) or D_val >= INF:
                        continue

                lr, lc = edge_to_local(edge, idx, rows, cols)
                if lr < 0 or lc < 0 or lr >= rows or lc >= cols:
                    continue

                s_val = S_single[lr, lc]

                # Decode local start coord inside this chunk (same as your visualize_chunk_edges)
                if s_val > 0:
                    start_local_r, start_local_c = 0, int(s_val)
                elif s_val < 0:
                    start_local_r, start_local_c = abs(int(s_val)), 0
                else:
                    start_local_r, start_local_c = 0, 0

                # Convert local start/end to GLOBAL coords
                g_start_r = r_start + start_local_r
                g_start_c = c_start + start_local_c
                g_end_r   = r_start + lr
                g_end_c   = c_start + lc

                # Make sure in bounds
                if not (0 <= g_start_r < L1 and 0 <= g_start_c < L2):
                    continue
                if not (0 <= g_end_r < L1 and 0 <= g_end_c < L2):
                    continue

                # Append this short line segment: (start) → (end)
                x_S.extend([g_start_c, g_end_c, None])
                y_S.extend([g_start_r, g_end_r, None])

    if x_S:
        fig.add_trace(
            go.Scattergl(
                x=x_S,
                y=y_S,
                mode="lines",
                name="Chunk S start→edge segments",
                line=dict(width=1, color="rgba(100,100,100,0.02)"),  # light grey-ish
                showlegend=False

            )
        )

    # ---------------- PARFLEX STITCHED GLOBAL-BEST ----------------
    stitched_wp = parflex_res['stitched_wp']
    if stitched_wp.size > 0:
        fig.add_trace(
            go.Scattergl(
                x=stitched_wp[:, 1],   # cols (F2)
                y=stitched_wp[:, 0],   # rows (F1)
                mode="lines",
                name="ParFlex stitched (global best)",
                line=dict(width=6,color="rgba(247,14,14,0.5)")
            )
        )

    # ---------------- 4) PARFLEX BEST-PER-SEGMENT PATHS ----------------
    paths_per_segment = parflex_res['paths_per_segment']

    for (edge_name, seg_idx), info in paths_per_segment.items():
        path = np.array(info['path'], dtype=int)
        if path.size == 0:
            continue
        fig.add_trace(
            go.Scattergl(
                x=path[:, 1],   # col
                y=path[:, 0],   # row
                mode="lines",
                name=f"{edge_name} seg={seg_idx}",
                line=dict(width=3, color="rgba(0,128,255,0.5)")
            )
        )


   # -------- GLOBAL FLEXDTW PATH (handle both (N,2) and (2,N)) --------
    flex_wp = np.asarray(flex_wp)

    if flex_wp.shape[1] == 2:
        # shape (N, 2): columns = (row, col)
        f1_frames = flex_wp[:, 0]
        f2_frames = flex_wp[:, 1]
    elif flex_wp.shape[0] == 2:
        # shape (2, N): rows = (row, col)
        f1_frames = flex_wp[0, :]
        f2_frames = flex_wp[1, :]
    else:
        raise ValueError(f"Unexpected flex_wp shape: {flex_wp.shape}")

    fig.add_trace(
        go.Scattergl(
            x=f2_frames,
            y=f1_frames,
            mode="lines",
            name="Global FlexDTW",
            line=dict(width=4, color="rgba(0,0,0,1)")
        )
    )
    # ---------------- 5) AXES / LAYOUT ----------------
    x_lo, x_hi = -0.5, L2 - 0.5
    y_lo, y_hi = -0.5, L1 - 0.5

    fig.update_layout(
        title="Global FlexDTW vs ParFlex (with chunk-S spiky background)",
        xaxis_title=f"F2 frames (0 … {L2-1})",
        yaxis_title=f"F1 frames (0 … {L1-1})",
        legend=dict(x=0.01, y=0.99),
        width=fig_width,
        height=fig_height,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_xaxes(range=[x_lo, x_hi], showgrid=False)
    fig.update_yaxes(range=[y_lo, y_hi], showgrid=False)

    # ---------------- 6) DOTTED GRID AT MULTIPLES OF L_div ----------------
    shapes = []
    for x in range(L_div, L2, L_div):
        shapes.append(dict(
            type="line",
            x0=x, x1=x,
            y0=y_lo, y1=y_hi,
            line=dict(width=1, dash="dot")
        ))
    for y in range(L_div, L1, L_div):
        shapes.append(dict(
            type="line",
            x0=x_lo, x1=x_hi,
            y0=y, y1=y,
            line=dict(width=1, dash="dot")
        ))
    fig.update_layout(shapes=shapes)

    fig.show()


# In[18]:


def run_all_benchmarks(outdir):
    parts_batch = []
    queryids = []
    with open(QUERY_LIST, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            assert len(parts) == 2
            queryid = os.path.basename(parts[0]) + '__' + os.path.basename(parts[1])
            
            if 'Czerny-Stefanska-1949_pid9086' in queryid:
                continue
            
            parts_batch.append(parts)
            queryids.append(queryid)
            
    for benchmark in tqdm(BENCHMARKS):
#         for i in range(len(parts_batch)):
#             run_benchmark(benchmark, FEAT_DIRS[benchmark][0], FEAT_DIRS[benchmark][1], parts_batch[i], outdir, queryids[i])
        run_benchmark_batch(benchmark, FEAT_DIRS[benchmark][0], FEAT_DIRS[benchmark][1], parts_batch, outdir, queryids, n_cores=4)


# In[19]:


def run_benchmark_batch(benchmark, featdir1, featdir2, parts_batch, outdir, queryids, n_cores):
    inputs = []
    assert len(parts_batch) == len(queryids)
    
    for i in range(len(parts_batch)):
        featfile1 = (featdir1 / parts_batch[i][0]).with_suffix('.npy')
        featfile2 = (featdir2 / parts_batch[i][1]).with_suffix('.npy')
        # Pass file paths to workers so arrays are loaded in child processes
        for system in SYSTEMS:
            outfile = get_outfile(outdir, benchmark, system, queryids[i])
            if not os.path.isfile(outfile):
                inputs.append((system, str(featfile1), str(featfile2), str(outfile)))

    if not inputs:
        return

    # Worker wrapper: load features inside child process to avoid pickling large arrays
    def _worker(task):
        system, f1_path, f2_path, outfile = task
        try:
            F1 = np.load(f1_path)
            F2 = np.load(f2_path)
        except Exception as e:
            return (False, str(e), task)

        try:
            align_system(system, F1, F2, outfile)
            # free local arrays
            del F1, F2
            gc.collect()
            return (True, None, task)
        except Exception as e:
            return (False, str(e), task)

    max_workers = max(1, min(n_cores, (os.cpu_count() or 1) - 1))
    if max_workers < 1:
        max_workers = 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(_worker, task) for task in inputs]
        for fut in concurrent.futures.as_completed(futures):
            ok, err, task = fut.result()
            if not ok:
                system, f1_path, f2_path, outfile = task
                print(f"Task failed: {system} {f1_path} {f2_path} -> {err}")
     
    
    return


# In[20]:


def run_benchmark(benchmark, featdir1, featdir2, parts, outdir, queryid):
    featfile1 = (featdir1 / parts[0]).with_suffix('.npy')
    featfile2 = (featdir2 / parts[1]).with_suffix('.npy')

    F1 = np.load(featfile1)
    F2 = np.load(featfile2)
        
    # run all baselines
    for system in SYSTEMS:
        
        # only compute alignment if this hypothesis file doesn't already exist
        outfile = get_outfile(outdir, benchmark, system, queryid)
        if not os.path.isfile(outfile):
            align_system(system, F1, F2, outfile)


# In[21]:


# outdir = Path(f'experiments_{DATASET}/{VERSION}')
# run_all_benchmarks(outdir)


In[ ]:


import itertools
# test block
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import math
# from your_module import align_system_split_recordings  # <-- import your function

directory = Path("/home/asharma/ttmp/Flex/FlexDTW/Chopin_Mazurkas_features/matching/Chopin_Op017No4")
d2 = Path('/home/asharma/ttmp/Flex/FlexDTW/Chopin_Mazurkas_features/original/Chopin_Op017No4')

f2 = list(d2.glob("*.npy"))
f2,b = random.sample(f2, 2)
files = list(directory.glob("*.npy"))

 
for i in range(10):
    f1, a = random.sample(files, 2)   # pick 2 different random files
    #print(f"[{i+1}] {directory}'/{f1.name} vs {d2}'/{f2.name}")
    # f1 = str('/home/asharma/ttmp/Flex/FlexDTW/Chopin_Mazurkas_features/original/Chopin_Op017No4/Chopin_Op017No4_Magaloff-1977_pid9074f-13.npy')
    # f2 = str('/home/asharma/ttmp/Flex/FlexDTW/Chopin_Mazurkas_features/matching/Chopin_Op017No4/Chopin_Op017No4_Levy-1951_pid915406-13.npy')

    F1 = np.load(f1)
    F2 = np.load(f2)
    # steps = [(1,1), (1,2), (2,1)]
    # weights = [1.25,3.0,3.0]
    outfile_test = f"test_path_{i+1}.pkl"
    # C_full = 1.0 - FlexDTW.L2norm(F1).T @ FlexDTW.L2norm(F2)
    # L1 = F1.shape[1]
    # L2 = F2.shape[1]
    # buffer_flex = min(L1, L2) * (1 - (1 - other_params['flexdtw']['beta']) * min(L1,L2) / max(L1, L2))
    # beta_full = other_params['flexdtw']['beta']
    # best_cost_full, global_flex_path, debug_full,D,B,S = FlexDTW.flexdtw(
    #     C_full, steps=steps, weights=weights, buffer=buffer_flex
    #         )
    #print(global_flex_path.shape) 
    wp1 = align_system("parflex",F1,F2, outfile_test) 
    wp2 = align_system("flexdtw",F1,F2, outfile_test)
    
    # print(wp1, wp2)
    # tiled_result = convert_chunks_to_tiled_result(
    # chunks_dict, L, n_chunks_1, n_chunks_2, C_full
    # )

    # # Now you can use it with the plotting function:
    
    # D_chunks, L_chunks = chunked_flexdtw(chunks_dict, L, n_chunks_1, n_chunks_2, buffer_param=1)
    # r = stage_2_backtrace_compatible(tiled_result, chunks_dict, D_chunks, L_chunks, 
    #                               L_block=4000, buffer_stage2=200, top_k=10, show_fig=False)
    # plot_parflex_with_chunk_S_background(
    #     tiled_result, C_full, global_flex_path, r, L_div=4000
    # )
    
    


# In[ ]:




