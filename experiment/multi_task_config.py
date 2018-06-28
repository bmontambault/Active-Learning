from random import sample

experiment="multi-task"
version="0.1.0"
functions=['pos_linear','neg_quad','sinc_compressed']
tasks=sample(['max_score','find_max','min_error'],3)
tasks=[tasks[:2]+[tasks[2]+'_last']]

bar_height=500
bar_width=15
nbars=80
trials=25
predict_trials=25

se_length=5
sinc_offset=15
neg_quad_offset=15
pos_quad_offset=-15

        
