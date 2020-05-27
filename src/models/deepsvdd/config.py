# deepsvdd
nce_factor = 1.
radius_factor = 1.

deepsvdd_lambda = 0.5
nce_t = 0.07
nce_k = 0
nce_m = 0
latent_size = 32

deepsvdd_sgd_base_lr = 3e-2
deepsvdd_sgd_weight_decay = 5e-4
deepsvdd_sgd_momentum = 0.9
deepsvdd_sgd_lr_decay = 0.999995
deepsvdd_sgd_schedule = 40

deepsvdd_pgd_eps = 0
deepsvdd_pgd_iterations = 20
deepsvdd_pgd_alpha = 0.05
deepsvdd_pgd_randomize = False

evaluation_interval = 1
validation_interval = 16
embedding_interval = 512
save_interval = 64
num_extreme_cases = 16
submit_latent_interval = 1
evaluate_train_interval = 1
