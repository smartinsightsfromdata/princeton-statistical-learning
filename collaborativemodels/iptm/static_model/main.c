#include <gflags/gflags.h>
#include <assert.h>

#include "corpus.h"
#include "infer.h"
#include "gsl-wrappers.h"

DECLARE_string(mode);
DECLARE_string(use_checkpoint);
DECLARE_int32(number_topics);
DECLARE_int32(ip_dimension);
DECLARE_bool(encode_difficulty);
DECLARE_bool(experimental_priors);
DECLARE_bool(variational_anneal);
DECLARE_double(vote_replication);

using legis::Corpus;
using legis::Posterior;
using legis::TestInferDoc;

static bool Converged(double old,
		      double current,
		      double criterion) {
  if ((fabs(old - current))
      / (fabs(old) / 2.0 + fabs(current) / 2.0 + 1.0)
      < criterion) {
    return true;
  }
  return false;
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, 0);

  if (FLAGS_mode == "test") {
     TestInferDoc();
     exit(0);  
  }

  Corpus data;
  data.ReadCorpus();

  Posterior p(data,
	      FLAGS_number_topics,
	      FLAGS_ip_dimension);


  if (FLAGS_mode == "infer_votes") {
    // Given a set of ideal points, infer votes.

    assert(FLAGS_use_checkpoint.size());
    // Measure the likelihood of these votes.
    p.ReadCheckpoint(/*read_document_stats=*/true,
		     /*read_user_stats=*/true,
		     data);
    double vote_lhood = p.VoteLikelihood(data);
    printf("Vote likelihood: %lf\n", vote_lhood);
    printf("Number votes: %d\n", data.number_votes());
    printf("Heldout log likelihood: %.4lf\n",
	   vote_lhood / data.number_votes());
    exit(0);
  } else if (FLAGS_mode == "fit"
	     || FLAGS_mode == "fit_predict"
	     || FLAGS_mode == "fixed_topics"
	     || FLAGS_mode == "fixed_topics_predict"
	     || FLAGS_mode == "ideal") {
    // Fit an ideal point topic model.

    if (FLAGS_mode == "fixed_topics"
	|| FLAGS_mode == "fixed_topics_predict") {
      p.ReadCheckpoint(/*read_document_stats=*/false,
		       /*read_user_stats=*/false,
		       data);
    } else {
      assert(FLAGS_use_checkpoint.size() == 0);
    }
    p.RandomInitializeSS();
    p.InitializeGamma(data);
    // Set these parameters to 1 for "variational annealing", to allow
    // other parts of the model to converge more quickly.
    if (FLAGS_variational_anneal) {
      p.sigma_u = 1.0 / FLAGS_vote_replication;
      p.dispersion = 1.0 / FLAGS_vote_replication;
    }
    p.sigma_kappa = sqrt(p.dispersion);
    p.sigma_lambda = p.sigma_u;
    if (FLAGS_experimental_priors) {
      p.ExpandIdealDimension(data);
    }
    p.RandomInitializeIdealPoint(data);
    double vote_likelihood = 10.0;
    double old_likelihood = 1.0;
    for (int i=0; i < 50
	   && !Converged(old_likelihood,
			 vote_likelihood,
			 1e-4); ++i) {
      p.FitIdealPoints(data,
		       -1,
		       /*second_order=*/true,
		       /*sparse=*/true,
		       /*only_users=*/false,
		       /*only_docs=*/false);
      old_likelihood = vote_likelihood;
      vote_likelihood = p.VoteLikelihood(data);
      for (int j=0; j < data.number_users(); ++j) {
	if (FLAGS_encode_difficulty) {
	  p.user_update_stats[j]->number_iterations = 1;
	  mset(p.user_ips, j, 0, 1.0);
	}
      }
    }

    for (int j=0; j < data.number_users(); ++j) {
      p.user_update_stats[j]->number_iterations = 1;
      if (FLAGS_encode_difficulty) {
	mset(p.user_ips, j, 0, 1.0);
      }
    }

    // Set these parameters to 1 for "variational annealing", to allow
    // other parts of the model to converge more quickly.
    if (FLAGS_variational_anneal) {
      p.sigma_u = 1.0 / FLAGS_vote_replication;
      p.dispersion = 1.0 / FLAGS_vote_replication;
    }
    p.sigma_kappa = p.sigma_u;
    p.sigma_lambda = sqrt(p.dispersion);

    p.WriteCheckpoint(data);

    p.Infer(data);
  } else if (FLAGS_mode == "resume") {
    // Given a partially run model, resume.
    // Note that this does not start out exactly from the saved run,
    // since zeta must be re-inferred.
    assert(FLAGS_use_checkpoint.size());
    // Here we read *most* relevant information.
    p.ReadCheckpoint(/*read_document_stats=*/true,
		     /*read_user_stats=*/true,
		     data);
    p.Infer(data);
  } else if (FLAGS_mode == "infer_docs") {
    // Infer documents' ideal points and predict votes using these
    // docs' parameters.

    assert(FLAGS_use_checkpoint.size());
    // Note that we don't read the docs' information
    // (since it's assumed to be irrelevant).
    p.ReadCheckpoint(/*read_document_stats=*/false,
		     /*read_user_stats=*/true,
		     data);
    p.RandomInitializeSS();
    p.InitializeGamma(data);
    p.PredictDocs(data);
    exit(0);
  } else if (FLAGS_mode == "ideal_predict") {
    // Infer ideal points only.

    p.ReadCheckpoint(/*read_document_stats=*/true,
		     /*read_user_stats=*/true,
		     data);
    p.WriteVotePredictions(data);
  } else if (FLAGS_mode == "lda") {
    // Just run LDA.

    p.RandomInitializeSS();
    p.InitializeGamma(data);
    p.FitLDAUntilConverged(data);
  } else {
    printf("Error.  Unhandled mode %s.\n",
	   FLAGS_mode.c_str());
    exit(1);
  }

  // Note that p should have been checkpointing all along, so there's
  // no need to output anything.
  
  return(0);
}



