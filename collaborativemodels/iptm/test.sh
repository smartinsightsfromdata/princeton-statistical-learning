#!/bin/bash

# Fit a model.
# Throw out users or docs with fewer than 6 votes.
static_model/legis \
  --mode=fit \
  --number_topics=8 \
  --sigma_u=1.0 \
  --dispersion=1.0 \
  --ip_dimension=2 \
  --alpha=0.01 \
  --min_votes_per_user=6 \
  --min_votes_per_doc=6 \
  --users=test/users.dat \
  --docs=test/train_mult.dat \
  --votes=test/train_votes.dat \
  --variational_anneal=false \
  --checkpoint_prefix=test/fit

# Make predictions on heldout documents / votes
static_model/legis \
  --mode=infer_docs \
  --number_topics=8 \
  --sigma_u=1.0 \
  --dispersion=1.0 \
  --ip_dimension=2 \
  --alpha=0.01 \
  --min_votes_per_user=6 \
  --min_votes_per_doc=6 \
  --users=test/users.dat \
  --docs=test/test_mult.dat \
  --votes=test/test_votes.dat \
  --checkpoint_prefix=test/predict \
  --use_checkpoint=test/fit

# Fit only the ideal point model (ignore document text).
static_model/legis \
  --mode=ideal \
  --sigma_u=1.0 \
  --dispersion=1.0 \
  --ip_dimension=2 \
  --min_votes_per_user=6 \
  --min_votes_per_doc=6 \
  --users=test/users.dat \
  --docs=test/train_mult.dat \
  --votes=test/train_votes.dat \
  --checkpoint_prefix=test/fit

# Fit a more aggressive dispersion, and use
# variational annealing.  Low dispersion will mean
# documents' ideal points are more closely tied with topics,
# and the annealing will give the ideal points more
# flexibility earlier in the model's convergence.
# Variational annealing defaults to true. 
static_model/legis \
  --mode=fit \
  --number_topics=8 \
  --sigma_u=1.0 \
  --dispersion=0.001 \
  --ip_dimension=2 \
  --alpha=0.01 \
  --min_votes_per_user=6 \
  --min_votes_per_doc=6 \
  --users=test/users.dat \
  --docs=test/train_mult.dat \
  --votes=test/train_votes.dat \
  --variational_anneal=true \
  --checkpoint_prefix=test/fit

# Make predictions on heldout documents / votes
static_model/legis \
  --mode=infer_docs \
  --number_topics=8 \
  --sigma_u=1.0 \
  --dispersion=1.0 \
  --ip_dimension=2 \
  --alpha=0.01 \
  --min_votes_per_user=6 \
  --min_votes_per_doc=6 \
  --users=test/users.dat \
  --docs=test/test_mult.dat \
  --votes=test/test_votes.dat \
  --checkpoint_prefix=test/predict \
  --use_checkpoint=test/fit