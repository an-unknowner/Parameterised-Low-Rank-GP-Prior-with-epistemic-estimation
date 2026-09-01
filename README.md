### 20260829 origin
    train main and kernel separately and use pretraining (and pretrain guided loss) to train BNN

### 20260831 update
    change KL normalize to full train dataset size instead of mini-batch size

### 20260901 update
    Delete P (basis function coeffient $\theta$ uncertainty) of kernel from stage 2 ELBO training, add data restore folder. 
