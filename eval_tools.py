import os
import numpy as np
import pickle

def read_start_times(annotfile):
    start_times = []
    with open(annotfile, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('%'):
                continue
            fields = line.split()
            if len(fields) < 1:
                continue
            try:
                start_time = float(fields[0])
                start_times.append(start_time)
            except ValueError:
                continue
    return np.array(start_times)

def getGroundTruthTimestamps(query_annot_file, ref_annot_file):
    """
    Get the ground truth timestamps from the annotation files.
    """
    # Read annotation file, extract start times as numpy array
    gt_query = read_start_times(query_annot_file)
    gt_ref = read_start_times(ref_annot_file)
    gt = np.stack([gt_query, gt_ref], axis = 1)
    return gt

def eval_alignment_single(hypfile, query_annot_file, ref_annot_file, outfile = None):
    gt = getGroundTruthTimestamps(query_annot_file, ref_annot_file)
    if gt.shape[0] == 0:
        print(f'No measures to evaluate in {hypfile}')
        return None, None
    
    # check if hypfile exists
    if not os.path.exists(hypfile):
        print(f'{hypfile} does not exist')
        return None, None
    hypalign = np.load(hypfile)
    
    pred = np.interp(gt[:,0], hypalign[0,:], hypalign[1,:])
    err = pred - gt[:,1]
    return err

def eval_alignment_batch(exp_dir, scenarios_dir, out_dir, tsm = False, lag = 0, hypFileExt = ''):
    # evaluate all scenarios
    d = {}
    
    for scenario_id in os.listdir(scenarios_dir):
        if tsm:
            if lag == 0:
                hypFile = f'{exp_dir}/{scenario_id}/tsm{hypFileExt}.npy'
            else: # if lag is not 0, the hypothesis file is a TSM path file with the lag
                hypFile = f'{exp_dir}/{scenario_id}/tsm_lag{lag}{hypFileExt}.npy'
        else:
            hypFile = f'{exp_dir}/{scenario_id}/hyp{hypFileExt}.npy'
            
        query_annot_file = f'{scenarios_dir}/{scenario_id}/query.beats'
        ref_annot_file = f'{scenarios_dir}/{scenario_id}/ref.beats'
        err = eval_alignment_single(hypFile, query_annot_file, ref_annot_file)
        d[scenario_id] = err
    
    # save
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    outfile = f'{out_dir}/errs.pkl'
    pickle.dump(d, open(outfile, 'wb'))