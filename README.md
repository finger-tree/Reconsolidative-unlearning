
sudo apt update
sudo apt install texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-extra-utils latexmk perl xzdec
sudo apt-get update && sudo apt-get install -y texlive-bibtex-extra


$\mathcal{K}(\Delta \tau) = \frac{1}{\Delta \tau}$

$
P(z_t = k \mid \mathcal{D}_{1:t}) \propto
P(\mathcal{D}_t \mid z_t = k, \mathcal{D}_{1:t-1}) \cdot
P(z_t = k \mid \mathbf{z}_{1:t-1})
$

$P(\mathcal{D}_t \mid z_t = k, \mathcal{D}_{1:t-1})=P(\mathbf{x}_t \mid z_t = k, \mathcal{D}_{1:t-1}) \,P(r_t \mid \mathbf{x}_t, z_t = k, \mathcal{D}_{1:t-1})$

$\hat{r}_t = \sum_k P(z_t = k \mid \mathcal{D}_{1:t-1}) \, \mathbf{w}_k^\top \mathbf{x}_t$

$\mathbf{w}_k \leftarrow \mathbf{w}_k + \eta \, P(z_t = k \mid \mathcal{D}_{1:t}) \, \mathbf{x}_t \, \delta_t$

$\delta_t = r_t - \hat{r}_t$