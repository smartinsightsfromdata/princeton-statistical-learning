#include <gflags/gflags.h>

//DEFINE_string(input_prefix,
//	      "",
//	      "A prefix for all corpus datafiles.");
//DEFINE_string(working_directory,
//	      "",
//	      "The working directory for all temp and output files.");
DEFINE_int32(ip_dimension,
	     1,
	     "The number of dimensions in the space of ideal points."
	     "Have not experimented much with this over 1, so use at your "
	     "own risk.");
DEFINE_int32(number_topics,
	     -1,
	     "The number of LDA topics.");
DEFINE_string(use_checkpoint,
	      "",
	      "When testing / predicting, use a checkpoint (i.e., a fit model) "
	      "iven by this prefix.");
DEFINE_string(mode,
	      "fit",
	      "Whether to fit (from scratch), "
	      "infer_votes, infer_docs, or resume.");
DEFINE_string(miscfile,
	      "misc",
	      "A file into which miscellaneous (e.g., experimental) data "
	      "can be dumped.");
	      
DEFINE_string(checkpoint_prefix,
	      "checkpoint",
	      "A checkpoint prefix for saving files.");
DEFINE_double(alpha,
	      -1,
	      "The Alpha LDA parameter.");
DEFINE_double(dispersion,
	      1e-2,
	      "The dispersion (sigma) parameter for the supervised response.");
DEFINE_double(sigma_lambda,
	      1e-1,
	      "Variational poserior variance for user ideal points.");
DEFINE_double(sigma_u,
	      1e-1,
	      "Variance over user ideal points.");
DEFINE_double(sigma_kappa,
	      1e-1,
	      "Variational posterior variance for doc ideal points.");
DEFINE_bool(encode_difficulty,
	    true,
	    "Explicitly define difficulty as the first document parameter. "
	    "This is the same as forcing the first user ideal point to 1."
	    "User ideal points are in this case not saved to user_ips.csv.");
DEFINE_bool(experimental_priors,
	    false,
	    "If true, set Kennedy (300059) to 4.0 "
	    "and Enzi (300041) to -4.0."
	    "Do the same for Henry Waxman and Donald Young, "
	    "respectively.");
DEFINE_bool(second_order,
	    true,
	    "If true, use second-order stochastic gradient descent."
	    "Does not work well when set to false.");
DEFINE_bool(batch_update,
	    true,
	    "If true, use batch updates for second-order updates. "
	    "Only applies to second-order updates. "
	    "Does not work well when set to false.");
DEFINE_double(vote_replication,
	      1.0,
	      "A factor to describe how many times each "
	      "vote should be replicated in the dataset.");
DEFINE_string(votes,
	      "",
	      "A datafile containing votes.");
DEFINE_string(users,
	      "",
	      "A datafile containing users,");
DEFINE_string(docs,
	      "",
	      "A datafile containing docs.");
DEFINE_string(time_filter,
	      "",
	      "If nonempty, accept only docs having this as a prefix.");
DEFINE_bool(variational_anneal,
	    true,
	    "If true, set hyperparameters to converge from 1 "
	    "to their true values for better initial convergence.");
