### 2026-08-29 origin
- train main and kernel separately and use pretraining (and pretrain guided loss) to train BNN

### 2026-08-31 update
- change KL normalize to full train dataset size instead of mini-batch size

### 2026-09-01 update
- Delete P (basis function coefficient $\theta$ uncertainty) of kernel from stage 2 ELBO training, add data restore folder. 

### 2026-09-02 update 1
- ELBO training only train std of network params

### 2026-09-02 update 2
- Delete guide loss from pretrain in stage 2
    