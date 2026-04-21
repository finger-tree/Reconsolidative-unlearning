# Literature review:

###
multiple trace theory
predictive coding theory
associative memory (hopfield networks is all you need)

metrics:
existing unlearning models
unlearning measures (neuralPS by example differential privacy thing)
###


extension:
nested learning (self evolution rules)
brain background activity
hippocampal formation CA memory indexing
self modifying code

self ideas:
brain energy rules (effort, reward)
neuron abbstraction concept (landmarks)




# system design (who knows):

GA
$$\mathcal{L}_{GA}(\theta{}) = - \mathbb{E}_{(x,y)\sim{}D_f}[-log(P(y|x;\theta{}))]$$
KL to align model:
$$\mathcal{L}_{KL}(\theta{}) = \mathbb{E}_{(x,y)\sim{}D_f}[D_{KL}(p(y|x;\theta{}_{orig})||p(y|x;\theta_{f}))]$$

PO: 
DPO (Direct preference optimisation):
$$\mathcal{L}_{DPO, \beta{}}(\theta{}) = - \frac{1}{\beta{}}\mathbb{E}_{D_p}[log\sigma{}(\beta{}log\frac{p(y_w|X;\theta{})}{p(y_w|X;\theta{}_{ori})} - \beta{}log\frac{p(y_l|X;\theta{})}{p(y_l|X;\theta{}_{ori})})]$$

NPO (negative preference optimisation):
$$\mathcal{L}_{NPO, \beta{}}(\theta{}) = \frac{1}{\beta{}}\mathbb{E}_{(x, y)\sim{}D_f}[log\sigma{}(-\beta{}log\frac{p(y|x;\theta{})}{p(y|x;\theta{}_{orig})})]$$


look into targeted unlearning methods:
DEPN Wu et al., 2023
WAGLE Jia et al., 2024
Needle (Hong et al., 2024)

ULD (Ji et al., 2024)
RKLD (Wang et al., 2024)
