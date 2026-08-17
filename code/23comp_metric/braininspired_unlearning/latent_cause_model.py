"""
paragraph{Cause Index} The latent cause theory model triangle relationship between conditioned stimuli, unconditioned stimuli, latent cause and conditional response. Would be reframed as data, unlearning marker, cause, and output. In which a model layer would be learnt to entirely dependent on the hidden causes in each layer to determine layer output. 
"""

# what the fuck translate the thing into python
from imm_run import imm_run
from imm_plot import imm_plot
from imm_localmap import imm_localmap

if __name__ == "__main__":
    # Run a single design (e.g. design 1: extinction without retrieval)
    results = imm_run(1)
    print(results.V)     # conditioned response on each trial
    print(results.Zp)    # latent cause posterior (per trial)

    # Reproduce a figure from the paper (e.g. Monfils et al. 2009)
    res = [imm_run(1), imm_run(2), imm_run(3)]
    fig = imm_plot('monfils09', res)
    fig.savefig('monfils09.png')