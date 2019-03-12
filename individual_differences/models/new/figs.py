import pandas as pd


def get_assignments(trace_df):
    
    assignment_columns = [c for c in trace_df.columns if 'assignments' in c]
    index = list(set([a for b in [trace_df[c].unique() for c in assignment_columns] for a in b]))
    assignments = trace_df[assignment_columns]
    assignment_counts = pd.concat([assignments[c].value_counts().sort_index().reindex(index) for c in trace_df.columns if 'assignments' in c], axis=1)
    assignment_counts.columns = assignment_counts.columns
    return assignment_counts.fillna(0)