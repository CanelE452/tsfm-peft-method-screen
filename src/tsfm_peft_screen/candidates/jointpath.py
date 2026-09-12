"""Candidate stopped before implementation: direct core-method collision.

Baron et al., arXiv:2510.02224 §3 and Appendix B already combine frozen
TSFM marginal quantiles and a small context-conditioned neural Gaussian
copula, trained with sample-path scores. See docs/NOVELTY_BOUNDARY.md.
"""
VERDICT='NOVELTY_COLLISION'
def run(*args,**kwargs):
    raise RuntimeError('GPU pilot prohibited by novelty gate; no renamed rerun')
