/*
  Copyright Princeton University, 2010. All rights reserved.
  Author: Sean M. Gerrish (sgerrish@cs.princeton.edu)

  This file declares data structures for the supervised ideal point
  model.

*/

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <vector>
#include <ext/hash_set>
#include <queue>

using __gnu_cxx::hash_set;

namespace legis {

void TestInferDoc();

class Corpus;

typedef struct GradientStats {
  GradientStats(int ip_dimension)
    : gradient_average_length(1.0),
  		number_iterations(10.0),
      coefficient(0.1),
      inc_sum(0.0),
      update_count(0),
      gradient_batch(),
      hessian_batch() {
    gradient_batch = gsl_vector_calloc(ip_dimension);
    hessian_batch = gsl_matrix_calloc(ip_dimension,
				      ip_dimension);
  }
  // Given a gradient, an ideal point ip, and gradient statistics,
  // update ip per stochastic gradient.
  // **Note that this modifies gradient!!**
  // (This was done for performance).
  void Push(double distance,
	    gsl_vector* gradient,
	    gsl_matrix* hessian,
	    gsl_vector* ip);
  void PushSecondOrder(gsl_vector* gradient,
		       gsl_vector* ip);
  void PushHessian(gsl_vector* ip);
  void UpdateHessian(gsl_vector* gradient,
		     gsl_matrix* hessian);

  void Flush();

  double gradient_average_length;
  double number_iterations;
  double coefficient;
  // How far has the 
  double inc_sum;
  int update_count;
  //queue<gsl_vector*> obs;
  //  queue<double> inc;
  gsl_vector* gradient_batch;
  gsl_matrix* hessian_batch;

} GradientStats;

struct Posterior {
  Posterior(const Corpus& corpus,
	    int number_topics,
	    int ip_dimension);
  ~Posterior();
  
  //  Infer a model.
  void Infer(const Corpus& corpus);

  // Using an inferred model, predict documents' ideal points.
  void PredictDocs(const Corpus& corpus);

  void ReadCheckpoint(bool read_document_stats,
		      bool read_user_stats,
		      const Corpus& corpus);
  void WriteCheckpoint(const Corpus& corpus) const;
  void RandomInitializeSS();
  void InitializeGamma(const Corpus& corpus);
  void RandomInitializeIdealPoint(const Corpus& corpus);

  double FitLDA(const Corpus& corpus,
		bool fit_doc_ips,
		double convergence_criterion);
  double UserIdealLikelihood(const Corpus& corpus, double sigma_u);
  double DocIdealLikelihood(const Corpus& corpus, double dispersion);
  double VoteLikelihood(const Corpus& corpus);
  double VoteClassificationAccuracy(const Corpus& corpus) const;
  void WriteVotePredictions(const Corpus& corpus) const;

  // Update the two ideal points and eta by
  // making a sweep through the votes.
  void FitIdealPoints(const Corpus& corpus,
		      int ideal_iteration,
		      bool second_order,
		      bool sparse,
		      bool user_only,
		      bool doc_only);

  double UpdateEta(const Corpus& corpus);

  void UpdateSigmaK(const Corpus& corpus);
  void UpdateSigmaL(const Corpus& corpus);
  void UpdateSigmaU(const Corpus& corpus);

  void InferDocIPsFromTopics(const Corpus& corpus);
  void FitLDAUntilConverged(const Corpus& corpus);

  void ExpandIdealDimension(const Corpus& corpus);
  void CollapseIdealDimension(const Corpus& corpus);

  // Posterior data.
  int expanded_ip_dimension;
  int number_topics;
  int number_terms;
  int number_docs;
  int ip_dimension;
  int iteration;

  // Inference data.
  bool last_iteration;

  double alpha;
  double sigma_kappa;
  double sigma_lambda;
  double likelihood;
  double sigma_u;
  double dispersion;

  vector<GradientStats*> user_update_stats;
  vector<GradientStats*> doc_update_stats;
  GradientStats eta_update_stats;

  // Expected log probabilities.
  gsl_matrix* topics;   // K x V
  gsl_matrix* topic_suffstats;   // K x V

  // Variational parameters for ideal points.
  gsl_matrix* doc_ips;  // D x N
  gsl_matrix* user_ips; // U x N
  gsl_matrix* etas;     // N x K
  gsl_matrix* doc_gammas; // D x K

  gsl_matrix* doc_zbars; // D x K

  vector<gsl_matrix*> doc_zbar_zbar_t; // D x (K x K)

  // A list of user ids with invalid ideal points.
  hash_set<string> missing_user_ids_;
  hash_set<string> missing_doc_ids_;
};


}  // namespace legis
