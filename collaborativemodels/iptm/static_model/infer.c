/*
  Copyright Princeton University, 2010. All rights reserved.
  Author: Sean M. Gerrish (sgerrish@cs.princeton.edu)

  This file implements the supervised ideal point model.

 */

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sf_psi.h>
#include <vector>
#include <fstream>
#include <ext/hash_set>

// Utils
#include <gflags/gflags.h>

#include "math_utils.h"
#include "gsl-wrappers.h"
#include "strings.h"
#include "logging.h"

#include "corpus.h"
#include "infer.h"


// DECLARE_int32(ip_dimension);
// DECLARE_int32(number_topics);
DECLARE_string(use_checkpoint);
DECLARE_string(checkpoint_prefix);
DECLARE_string(miscfile);
DECLARE_string(mode);
DECLARE_double(alpha);
DECLARE_double(dispersion);
DECLARE_double(sigma_lambda);
DECLARE_double(sigma_kappa);
DECLARE_double(sigma_u);
DECLARE_double(vote_replication);
DECLARE_bool(encode_difficulty);
DECLARE_bool(second_order);
DECLARE_bool(batch_update);
DECLARE_bool(experimental_priors);
DECLARE_bool(variational_anneal);

// EXPERIMENTAL FLAGS
DEFINE_bool(fix_eta,
	    false,
	    "If true, fix values of eta to favor "
	    "formation of certain topics"
	    "which explain parts of the ideal space.");

DEFINE_double(sigma_eta,
	      0.0,
	      "Describes the variance given to eta for each document.");

DEFINE_bool(discerning_etas,
	    false,
	    "If true, allow topics to contribute (by eta) "
	    "to difficulty or discrimination, but not both.");

using std::endl;
using std::ios;
using std::min;
using std::max;

namespace legis {

using namespace slap_utils;
using slap_utils::log_sum;

static const int kDocMod = 4;
static const int kUserMod = 3;
static const double kPriorVariance = 1e-3 * 1e-3;
static const double kHessianEpsilon = 1e-5;
static const int kMaxLDAIterations = 65;


static void FixHessian(gsl_matrix* hessian, bool* fixed1, bool* fixed2) {
  *fixed1 = false;
  *fixed2 = false;

  bool hessian_okay = false;
  double diagonal_add = 10.0;

  while (!hessian_okay) {
    hessian_okay = true;
    for (int i=0; i < hessian->size1 && hessian_okay; ++i) {
      // If the hessian diagonal is less than kHessianEpsilon, change
      // that.
      for (int j=0; j < hessian->size1 && hessian_okay; ++j) {
	// If the hessian diagonal is less than kHessianEpsilon, change
	// that.	
	//    Log(StringPrintf("Fixing hessian diag. %d", i));
	if (mget(hessian, i, i) < kHessianEpsilon) {
	  // mset(hessian, i, i, kHessianEpsilon);
	  *fixed1 = true;
	  hessian_okay = false;
	}
      }

      for (int j=i + 1; j < hessian->size2 && hessian_okay; ++j) {
	double pairwise_det = (mget(hessian, i, i)
			       * mget(hessian, j, j)
			       - mget(hessian, j, i)
			       * mget(hessian, i, j));
	if (pairwise_det < (kHessianEpsilon
			    * kHessianEpsilon)) {
	  hessian_okay = false;
	  *fixed2 = true;
	  // If we need to fix this hessian, don't let the new Hessian
	  // be too small (which would throw us off track).
	}
      }
    }
    
    if (!hessian_okay) {
      for (int i=0; i < hessian->size1; ++i) {
	mset(hessian, i, i, mget(hessian, i, i) + diagonal_add);
	diagonal_add *= 2.0;
      }
    }
  }
}


// Solves the linear system Ax = b for x.
// Assumes that x is already allocated.
// Clobbers A!
void Solve(gsl_matrix* A,
	   const gsl_vector* b,
	   gsl_vector* x) {
    int permutation_sign;
    gsl_permutation* permutation = gsl_permutation_alloc(b->size);
    gsl_linalg_LU_decomp(A, permutation, &permutation_sign);
    gsl_linalg_LU_solve(A, permutation, b, x);
    gsl_permutation_free(permutation);

    // So that we make mistakes more obvious, set A to zero.  (A is
    // changed above).
    gsl_matrix_set_zero(A);
}


Posterior::~Posterior() {
  gsl_matrix_free(topics);

  gsl_matrix_free(topic_suffstats);

  gsl_matrix_free(user_ips);

  gsl_matrix_free(doc_ips);

  gsl_matrix_free(etas);

  gsl_matrix_free(doc_gammas);

  gsl_matrix_free(doc_zbars);

  for (vector<gsl_matrix*>::iterator it=doc_zbar_zbar_t.begin();
       it != doc_zbar_zbar_t.end();
       ++it) {
    gsl_matrix_free(*it);
  }
}


Posterior::Posterior(const Corpus& corpus,
		     int number_topics_in,
		     int ip_dimension_in)
  : expanded_ip_dimension(0),
    iteration(0),
    sigma_u(0.1),
    dispersion(0.01),
    topics(NULL),
    topic_suffstats(NULL),
    doc_ips(NULL),
    user_ips(NULL),
    etas(NULL),
    doc_gammas(NULL),
    eta_update_stats(ip_dimension_in) {
  number_topics = number_topics_in;
  number_terms = corpus.number_terms();
  number_docs = corpus.number_docs();
  ip_dimension = ip_dimension_in;
  alpha = FLAGS_alpha;
  sigma_u = FLAGS_sigma_u;
  dispersion = FLAGS_dispersion;
  sigma_lambda = FLAGS_sigma_u;
  sigma_kappa = sqrt(FLAGS_dispersion);

  topics = gsl_matrix_calloc(number_topics,
			     corpus.number_terms());
  
  topic_suffstats = gsl_matrix_calloc(number_topics,
				      corpus.number_terms());

  user_ips = gsl_matrix_calloc(corpus.number_users(),
			       ip_dimension_in);
  doc_ips = gsl_matrix_calloc(corpus.number_docs(),
			      ip_dimension_in);

  etas = gsl_matrix_calloc(ip_dimension_in, number_topics);
  doc_gammas = gsl_matrix_calloc(corpus.number_docs(),
				 number_topics);

  doc_zbars = gsl_matrix_calloc(corpus.number_docs(),
				number_topics);

  for (int d=0; d < corpus.number_docs(); ++d) {
    doc_zbar_zbar_t.push_back(gsl_matrix_calloc(number_topics,
						number_topics));
  }

  for (int d=0; d < corpus.number_docs(); ++d) {
    GradientStats* doc_gs = new GradientStats(ip_dimension_in);
    doc_update_stats.push_back(doc_gs);
  }

  for (int u=0; u < corpus.number_users(); ++u) {
    GradientStats* user_gs = new GradientStats(ip_dimension_in);
    user_update_stats.push_back(user_gs);
  }
}


static void WriteParameters(const string& filename,
			    const Posterior& post) {
  std::ofstream f;
  string line;
  f.open(filename.c_str(), ios::out);
  f << "FLAGS_sigma_kappa:" << post.sigma_kappa << endl;
  f << "FLAGS_sigma_lambda:" << post.sigma_lambda << endl;
  f << "iteration:" << post.iteration << endl;
  f << "likelihood:" << post.likelihood << endl;
  f << "sigma_u:" << post.sigma_u << endl;
  f << "dispersion:" << post.dispersion << endl;
  f.close();
}


static void ReadParameters(const string& filename,
			   Posterior* post) {
  std::ifstream f;
  string line;
  f.open(filename.c_str(), ios::in);
  while (!f.eof()) {
    getline(f, line);
    if (!line.size()) {
      break;
    }
    vector<string> line_parts;
    SplitStringUsing(line, ":", &line_parts);
    if (line_parts.size() != 2) {
      printf("Error parsing line: \n%s\n", line.c_str());
    }
    if (line_parts[0] == "FLAGS_sigma_kappa") {
      post->sigma_kappa = ParseLeadingDoubleValue(line_parts[1].c_str());
    } else if (line_parts[0] == "FLAGS_sigma_lambda") {
      post->sigma_lambda = ParseLeadingDoubleValue(line_parts[1].c_str());
    } else if (line_parts[0] == "iteration") {
      if (FLAGS_mode == "fixed_topics_predict"
	  || FLAGS_mode == "fixed_topics") {
	post->iteration = 0;
      } else {
	post->iteration = ParseLeadingIntValue(line_parts[1].c_str());
      }
    } else {
      printf("Error.  Unknown flag: %s.  Skipping.\n",
	     line_parts[0].c_str());
    }
  }  
  f.close();
}


static void ReadUserCSV(const Corpus& corpus,
			Posterior* post) {
  gsl_matrix* user_ips = post->user_ips;
  // We might have a user matrix of a different size.
  // Read users by id.
  for (int u=0; u < user_ips->size1; ++u) {
    post->missing_user_ids_.insert(corpus.users(u)->id());
    for (int j=0; j < user_ips->size2; ++j) {
      mset(user_ips, u, j, 0.0);
    }
  }

  string filename = StringPrintf(
    "%s_user_ips.csv",
    FLAGS_use_checkpoint.c_str());
  printf("Reading user csv: %s.\n", filename.c_str());
  string line;
  std::ifstream f;
  f.open(filename.c_str(), ios::in);
  if (!f.is_open()) {
    LogFatal(StringPrintf("Error.  Could not open users file %s.\nFailing.\n",
			  filename.c_str()));
  }
  vector<string> parts;
  getline(f, line);
  while (!f.eof()) {
    getline(f, line);
    if (!line.size()) {
      break;
    }
    SplitStringUsing(line, ",", &parts);
    if (parts.size() < 1) {
      LogFatal("Error.  Too few fields in users csv.");
    } else if (FLAGS_encode_difficulty ?
	       parts.size() > user_ips->size2 :
	       parts.size() > (user_ips->size2 + 1)) {
      LogFatal("Error.  Too many fields in users csv.");
    }
    int user = corpus.userid_user(parts[0]);
    hash_set<string>::iterator it
      = post->missing_user_ids_.find(parts[0]);
    if (user < 0 || user > user_ips->size1) {
      printf("Unknown UserId %s, index %d. Skipping.\n",
	     parts[0].c_str(),
	     user);
      if (it != post->missing_user_ids_.end()) {
	LogFatal("Programmer error.  Id exists.");
      }
      continue;
    } else {
      if (it == post->missing_user_ids_.end()) {
	LogFatal("Programmer error.  Id should not exist.");
      }
    }
    post->missing_user_ids_.erase(it);
    if (FLAGS_encode_difficulty) {
      mset(user_ips, user, 0, 1.0);
      for (int i=1; i < parts.size(); ++i) {
	double ip = ParseLeadingDoubleValue(parts[i]);
	mset(user_ips, user, i, ip);
      }
    } else { 
      for (int i=1; i < parts.size(); ++i) {
	double ip = ParseLeadingDoubleValue(parts[i]);
	mset(user_ips, user, i - 1, ip);
      }
    }
  }
  string user_id_string = "";
  for (hash_set<string>::iterator it
	 = post->missing_user_ids_.begin();
       it != post->missing_user_ids_.end();
       ++it) {
    if (!user_id_string.empty()) {
      user_id_string += ",";
    }
    user_id_string += *it;
  }
  if (!user_id_string.empty()) {
    printf("Found no ideal points for user ids %s.\n",
	   user_id_string.c_str());
  }  
}


static void ReadDocCSV(const Corpus& corpus,
		       Posterior* post) {
  gsl_matrix* doc_ips = post->doc_ips;
  // We might have a doc matrix of a different size.
  // Read docs by id.
  for (int d=0; d < doc_ips->size1; ++d) {
    post->missing_doc_ids_.insert(corpus.docs(d)->id());
    for (int j=0; j < doc_ips->size2; ++j) {
      mset(doc_ips, d, j, 0.0);
    }
  }

  string filename = StringPrintf(
    "%s_doc_ips.csv",
    FLAGS_use_checkpoint.c_str());

  printf("Reading doc csv: %s.\n", filename.c_str());
  string line;
  std::ifstream f;
  f.open(filename.c_str(), ios::in);
  if (!f.is_open()) {
    LogFatal(StringPrintf("Error.  Could not open docs file %s.\nFailing.\n",
			  filename.c_str()));
  }
  vector<string> parts;
  getline(f, line);
  while (!f.eof()) {
    getline(f, line);
    if (!line.size()) {
      break;
    }
    SplitStringUsing(line, ",", &parts);
    if (parts.size() < 1) {
      LogFatal("Error.  Too few fields in docs csv.");
    } else if (parts.size() > doc_ips->size2 + 1) {
      LogFatal("Error.  Too many fields in docs csv.");
    }
    int doc = corpus.docid_doc(parts[0]);
    hash_set<string>::iterator it
      = post->missing_doc_ids_.find(parts[0]);
    if (doc < 0 || doc > doc_ips->size1) {
      printf("Unknown DocId %s, index %d. Skipping.\n",
	     parts[0].c_str(),
	     doc);
      if (it != post->missing_doc_ids_.end()) {
	LogFatal("Programmer error.  Id exists.");
      }
      continue;
    } else {
      if (it == post->missing_doc_ids_.end()) {
	LogFatal("Programmer error.  Id should not exist.");
      }
    }
    post->missing_doc_ids_.erase(it);
    for (int i=1; i < parts.size(); ++i) {
      double ip = ParseLeadingDoubleValue(parts[i]);
      mset(doc_ips, doc, i - 1, ip);
    }
  }
  string doc_id_string = "";
  for (hash_set<string>::iterator it
	 = post->missing_doc_ids_.begin();
       it != post->missing_doc_ids_.end();
       ++it) {
    if (!doc_id_string.empty()) {
      doc_id_string += ",";
    }
    doc_id_string += *it;
  }
  if (!doc_id_string.empty()) {
    printf("Found no ideal points for doc ids %s.\n",
	   doc_id_string.c_str());
  }  
}


void Posterior::ReadCheckpoint(bool read_document_stats,
			       bool read_user_stats,
			       const Corpus& corpus) {
  // Note that we don't read topic suffstats.

  string filename;
  if (FLAGS_mode == "resume"
      || FLAGS_mode == "fit"
      || FLAGS_mode == "fit_predict"
      || FLAGS_mode == "fixed_topics" 
      || FLAGS_mode == "fixed_topics_predict"
      || FLAGS_mode == "infer_docs") {
    filename = StringPrintf(
      "%s_beta.dat",
      FLAGS_use_checkpoint.c_str());
    mtx_fscanf(filename.c_str(), topics);
  }

  if (read_user_stats
      || FLAGS_mode == "ideal_predict"
      || FLAGS_mode == "resume"
      || FLAGS_mode == "infer_docs") {
    ReadUserCSV(corpus, this);
  }

  filename = StringPrintf(
    "%s_etas.dat",
    FLAGS_use_checkpoint.c_str());
  mtx_fscanf(filename.c_str(), etas);

  if (read_document_stats
      || FLAGS_mode == "resume"
      || FLAGS_mode == "ideal_predict") {
    ReadDocCSV(corpus, this);
  }
  
  if ((!(FLAGS_mode == "ideal_predict")
       && read_document_stats)
      || FLAGS_mode == "resume") {
    filename = StringPrintf(
      "%s_gam.dat",
      FLAGS_use_checkpoint.c_str());
    mtx_fscanf(filename.c_str(), doc_gammas);

    filename = StringPrintf(
      "%s_zbar.dat",
      FLAGS_use_checkpoint.c_str());
    mtx_fscanf(filename.c_str(), doc_zbars);
  }

  filename = StringPrintf(
    "%s_parameters.dat",
    FLAGS_use_checkpoint.c_str());
  ReadParameters(filename, this);

  // Note that we don't dump the zbar zbar'.
}


static void WriteEtas(const Posterior& post) {
  string filename;
  std::ofstream f;
  f.open(StringPrintf("%s_etas.csv",
		      FLAGS_checkpoint_prefix.c_str()).c_str(),
	 ios::out);
  f << "IpIndex,Topic,Eta" << endl;
  for (int i=0; i < post.ip_dimension; ++i) {
    for (int k=0; k < post.number_topics; ++k) {
      f << StringPrintf("%d,%d,%lf", i, k, mget(post.etas, i, k));
      f << endl;
    }
  }
  f.close();
}


static void WriteGammas(const Posterior& post) {
  string filename;
  std::ofstream f;
  f.open(StringPrintf("%s_gams.csv",
		      FLAGS_checkpoint_prefix.c_str()).c_str(),
	 ios::out);
  for (int i=0; i < post.doc_gammas->size1; ++i) {
    for (int k=0; k < post.number_topics; ++k) {
      if (k) {
	f << " ";
      }
      f << mget(post.doc_gammas, i, k);
    }
    f << endl;
  }
  f.close();
}


void Posterior::WriteCheckpoint(const Corpus& corpus) const {
  // Note that we don't read topic suffstats.

  string filename;

  if (FLAGS_mode == "lda"
      || FLAGS_mode == "fit"
      || FLAGS_mode == "fit_predict"
      || FLAGS_mode == "fixed_topics"
      || FLAGS_mode == "fixed_topics_predict"
      || FLAGS_mode == "resume") {
    string filename = StringPrintf(
      "%s_beta.dat",
      FLAGS_checkpoint_prefix.c_str());
    mtx_fprintf(filename.c_str(), topics);
  }
 
  std::ofstream f;
  // Print user information.
  if (FLAGS_mode == "fit"
      || FLAGS_mode == "fit_predict"
      || FLAGS_mode == "ideal"
      || FLAGS_mode == "resume"
      || FLAGS_mode == "infer_docs"
      || FLAGS_mode == "fixed_topics"
      || FLAGS_mode == "fixed_topics_predict") {
    filename = StringPrintf("%s_user_ips.dat",
			    FLAGS_checkpoint_prefix.c_str());
    mtx_fprintf(filename.c_str(), user_ips);
    f.open(StringPrintf("%s_user_ips.csv",
			FLAGS_checkpoint_prefix.c_str()).c_str(),
	   ios::out);
    f << "UserId";
    for (int i=FLAGS_encode_difficulty ? 1 : 0; i < ip_dimension; ++i) {
      f << StringPrintf(",Ip%d", i);
    }
    f << endl;
    for (int u = 0; u < corpus.number_users(); ++u) {
      f << corpus.users(u)->id().c_str();
      for (int i=FLAGS_encode_difficulty ? 1 : 0; i < ip_dimension; ++i) {
	f << StringPrintf(",%lf", mget(user_ips, u, i));
      }
      f << endl;
    }
    f.close();
  }

  // Print eta information.  Don't print this if we're running in
  // fit_predict mode (in which case we print eta with each iteration).
  if (FLAGS_mode != "fit_predict"
      && FLAGS_mode != "fixed_topics_predict") {
    WriteEtas(*this);
    filename = StringPrintf(
        "%s_etas.dat",
	FLAGS_checkpoint_prefix.c_str());
    mtx_fprintf(filename.c_str(), etas);
  }

  // Print document information.
  filename = StringPrintf(
    "%s_doc_ips.dat",
    FLAGS_checkpoint_prefix.c_str());
  mtx_fprintf(filename.c_str(), doc_ips);
  f.open(StringPrintf("%s_doc_ips.csv",
		      FLAGS_checkpoint_prefix.c_str()).c_str(),
	 ios::out);
  f << "DocId";
  for (int i=0; i < ip_dimension; ++i) {
    f << StringPrintf(",Ip%d", i);
  }
  f << endl;
  for (int d = 0; d < corpus.number_docs(); ++d) {
    f << corpus.docs(d)->id().c_str();
    for (int i=0; i < ip_dimension; ++i) {
      f << StringPrintf(",%lf",
			mget(doc_ips, d, i));
    }
    f << endl;
  }
  f.close();

  WriteGammas(*this);
  filename = StringPrintf(
    "%s_gam.dat",
    FLAGS_checkpoint_prefix.c_str());
  mtx_fprintf(filename.c_str(), doc_gammas);

  filename = StringPrintf(
    "%s_zbar.dat",
    FLAGS_checkpoint_prefix.c_str());
  mtx_fprintf(filename.c_str(), doc_zbars);

  filename = StringPrintf(
    "%s_parameters.dat",
    FLAGS_checkpoint_prefix.c_str());
  WriteParameters(filename, *this);

  // Note that we don't dump the zbar zbar'

  // Finally, write vote predictions.
  if (FLAGS_mode == "fit"
      || FLAGS_mode == "fit_predict"
      || FLAGS_mode == "ideal"
      || FLAGS_mode == "resume"
      || FLAGS_mode == "ideal_predict"
      || FLAGS_mode == "fixed_topics"
      || FLAGS_mode == "fixed_topics_predict"
      || FLAGS_mode == "infer_docs") {
    WriteVotePredictions(corpus);
  }
}


static void Normalize(gsl_vector* v, double sum) {
  double v_sum = 0.0;
  for (int i=0; i < v->size; ++i) {
    v_sum += vget(v, i);
  }
  if (v_sum == 0.0) {
    return;
  }
  for (int i=0; i < v->size; ++i) {
    vset(v, i, vget(v, i) * sum / v_sum);
  }
}


static bool Converged(double old, double current) {
  // if ((fabs(old - current)) / (fabs(old) + 1.0) < 9e-6) {
  if ((fabs(old - current)) / (fabs(old) / 2.0 + fabs(current) / 2.0 + 1.0) < 1e-5) {
    return true;
  }
  return false;
}


static bool Converged(double old, double current, double threshold) {
  if ((fabs(old - current)) / (fabs(old) / 2.0 + fabs(current) / 2.0 + 1.0) < threshold) {
    return true;
  }
  return false;
}


static bool Converged(double old,
		      double current,
		      double threshold,
		      double* value) {
  *value = (fabs(old - current)) / (fabs(current) / 2.0 + fabs(old) / 2.0 + 1.0);
  if (*value < threshold) {
    return true;
  }
  return false;
}

static bool LDAConverged(double old, double current) {
  if ((fabs(old - current)) / (fabs(old) / 2.0 + fabs(current) / 2.0 + 1.0) < 2e-6) {
    return true;
  }
  return false;
}

void Posterior::FitLDAUntilConverged(const Corpus& corpus) {
  likelihood = 1.0;
  double old_likelihood = 0.0;
  iteration = 0;
  while (iteration < 4 || !Converged(old_likelihood, likelihood, 1e-5)) {
    old_likelihood = likelihood;
    likelihood = FitLDA(corpus, false, 1e-6);
    if (iteration % 6 == 0) {
      WriteCheckpoint(corpus);
    }
    ++iteration;
  }
  WriteCheckpoint(corpus);
}

void Posterior::RandomInitializeSS() {
  const int kNumberTerms = number_terms;
  for (int k=0; k < number_topics; ++k) {
    for (int n = 0; n < kNumberTerms; ++n) {
      mset(topic_suffstats, k, n, 2.0 * UniformRNG());
    }
  }
}
  /*
static double DocIdealLikelihood(Posterior* post,
				 int d,
				 double dispersion) {
  gsl_vector* tmp = gsl_vector_alloc(post->number_topics);
  double ip_likelihood;
  for (int i=0; i < post->ip_dimension; ++i) {
    gsl_vector_view doc_zbar = gsl_matrix_row(post->doc_zbars, d);
    double doc_ip = mget(post->doc_ips, d, i);
    double eta_zbar;
    gsl_vector_view eta_i = gsl_matrix_row(post->etas, i);
    gsl_blas_ddot(&doc_zbar.vector, &eta_i.vector, &eta_zbar);
    
    // Compute eta' E(zbar zbar') eta.
    gsl_blas_dgemv(CblasNoTrans,
		   1.0,
		   post->doc_zbar_zbar_t[d],
		   &eta_i.vector,
		   0.0,
		   tmp);
    double eta_zbar_zbar_eta;
    gsl_blas_ddot(tmp, &eta_i.vector, &eta_zbar_zbar_eta);
    
    // Why is eta_zbar_zbar_eta with a negative coefficient?
    // Changing this..
    // TODO(sgerrish): ensure that this was the right move.
    // Note that we use FLAGS_dispersion instead of any posterior
    // sense of dispersion.
    double term1 = - ((eta_zbar_zbar_eta
		       - 2.0 * doc_ip * eta_zbar
		       + doc_ip * doc_ip
		       + post->sigma_kappa * post->sigma_kappa)
		      / (2.0 * dispersion));
    double term2 = - log(2 * PI * dispersion) / 2.0;
    double term3 = ((1.0 + log(2 * PI * post->sigma_kappa * post->sigma_kappa))
		    / 2.0);
    ip_likelihood += term1 + term2 + term3;
  }
  gsl_vector_free(tmp);
  return ip_likelihood;
}
*/


static double DocIdealLikelihood(const Document& doc,
				 int doc_index,
				 Posterior* post,
				 gsl_matrix* phi) {
  double ideal_likelihood = 0.0;
  // Also add in the likelihood term for the doc.
  gsl_vector* tmp = gsl_vector_alloc(post->number_topics);
  
  double predicted_ip1;
  double predicted_ip2;
  double doc_ip;
  gsl_vector_view zbars = gsl_matrix_row(post->doc_zbars, doc_index);
  for (int i=0; i < post->ip_dimension; ++i) {
    gsl_vector_view eta = gsl_matrix_row(post->etas, i);
    gsl_vector_memcpy(tmp, &eta.vector);
    gsl_vector_mul(tmp, &zbars.vector);
    predicted_ip1 = sum(tmp);
    predicted_ip1 = predicted_ip1 * predicted_ip1;
    const int N = doc.term_counts_total();
    double diff = 0.0;
    for (int n=0; n < N; ++n) {
      gsl_vector_view phi_n = gsl_matrix_column(phi, n);
      gsl_vector_memcpy(tmp, &phi_n.vector);
      gsl_vector_mul(tmp, &eta.vector);
      double term_sum = sum(tmp);
      double adjustment = -term_sum * term_sum / N / N;

      for (int k=0; k < post->number_topics; ++k) {
	double eta_term = vget(&eta.vector, k);
	double phi_tmp = mget(phi, k, n);
	adjustment += eta_term * eta_term * phi_tmp / N / N;
      }
      predicted_ip1 = predicted_ip1 + adjustment;
    }
    gsl_blas_ddot(&zbars.vector, &eta.vector, &predicted_ip2);
    doc_ip = mget(post->doc_ips, doc_index, i);
    ideal_likelihood += -(predicted_ip1
			  - 2.0 * predicted_ip2 * doc_ip
			  + doc_ip * doc_ip
			  + post->sigma_kappa * post->sigma_kappa);
  }
  
  gsl_vector_free(tmp);
  ideal_likelihood /= (2.0 * post->dispersion);
  ideal_likelihood += post->ip_dimension * (-log(2 * PI * post->dispersion)) / 2.0;
  ideal_likelihood += post->ip_dimension * (1.0 + log(2 * PI * post->sigma_kappa * post->sigma_kappa)) / 2.0;

  return ideal_likelihood;
}


static double DocLikelihood(const Document& doc,
			    int doc_index,
			    Posterior* post,
			    gsl_matrix* phi,
			    gsl_vector* gamma,
			    bool include_ideal) {
  double likelihood = 0.0, digsum = 0.0,
    var_gamma_sum = 0.0, dig[post->number_topics];

  for (int k = 0; k < post->number_topics; ++k) {
    dig[k] = gsl_sf_psi(vget(gamma, k));
    var_gamma_sum += vget(gamma, k);
  }
  digsum = gsl_sf_psi(var_gamma_sum);

  likelihood =
    lgamma(post->alpha * post->number_topics)
    - post->number_topics * lgamma(post->alpha)
    - (lgamma(var_gamma_sum));

  if (isnan(likelihood)) {
    Log("Likelihood is nan.  Failing.\n");
  }

  for (int k = 0; k < post->number_topics; ++k) {
    double dig_k = dig[k];
    double gamma_k = vget(gamma, k);
    likelihood +=
      (post->alpha - 1) * (dig_k - digsum) + lgamma(gamma_k)
      - (gamma_k - 1) * (dig_k - digsum);
    int n=0;
    double average_phi = 0.0;
    for (int i=0; i < doc.unique_terms(); ++i) {
      const int kTermCount = doc.counts(i);
      const int kTerm = doc.terms(i);
      for (int j = 0; j < kTermCount; ++j) {
	double phi_tmp = mget(phi, k, n);
	if (phi_tmp > 0) {
	  likelihood += (phi_tmp * ((dig[k] - digsum) - log(phi_tmp)
				    + mget(post->topics, k, kTerm)));
	} else {
	  LogFatal("Negative phi.\n");
	}
	++n;

      }
    }
  }

  if (include_ideal) {
    {
      int nnn = 0;
      gsl_vector_view doc_zbars =
	gsl_matrix_row(post->doc_zbars, doc_index);
      gsl_vector_set_zero(&doc_zbars.vector);
      const double NNN = doc.term_counts_total();
      for (int iii=0; iii < doc.unique_terms(); ++iii) {
	const int kTermCount = doc.counts(iii);
	const int kTerm = doc.terms(iii);
	for (int jjjj=0; jjjj < kTermCount; ++jjjj) {
	  gsl_vector_view phi_term = gsl_matrix_column(phi, nnn);
	  gsl_blas_daxpy(1.0 / NNN, &phi_term.vector, &doc_zbars.vector);
	  ++nnn;
	}
      }
    }
      
    double ideal_likelihood = DocIdealLikelihood(doc,
						 doc_index,
						 post,
						 phi);

    likelihood += ideal_likelihood;
  }
  return(likelihood);
}


static void UpdateZBarZBarT(gsl_matrix* doc_zbar_zbar_t,
			    const Document& doc,
			    Posterior* post,
			    gsl_matrix* phi,
			    int N) {
  double N_double(N);
  // Update E[ zbar zbar' ] for this document.
  gsl_matrix_set_zero(doc_zbar_zbar_t);
  gsl_vector* mean_phi = gsl_vector_calloc(post->number_topics);
  int m=0;
  for (int term_m = 0; term_m < doc.unique_terms(); ++term_m) {
    const int kmCount = doc.counts(term_m);
    for (int m_i=0; m_i < kmCount; ++m_i) {
      gsl_vector_view m_phi = gsl_matrix_column(phi, m);
      gsl_vector_add(mean_phi, &m_phi.vector);
      ++m;
    }
  }
  gsl_vector_scale(mean_phi, 1.0 / N_double);
  gsl_blas_dger(1.0,
		mean_phi,
		mean_phi,
		doc_zbar_zbar_t);
  gsl_vector_free(mean_phi);

  for (int k = 0; k < post->number_topics; ++k) {
    m = 0;
    for (int term_m = 0; term_m < doc.unique_terms(); ++term_m) {
      const int kmCount = doc.counts(term_m);
      for (int m_i=0; m_i < kmCount; ++m_i) {
	double phi_tmp = mget(phi, k, m);
	//minc(doc_zbar_zbar_t, k, k, ((phi_tmp * phi_tmp)
	minc(doc_zbar_zbar_t, k, k,
	     ((phi_tmp - phi_tmp * phi_tmp) / (N_double * N_double)));
	for (int j=0; j < post->number_topics; ++j) {
	  if (j == k) {
	    continue;
	  }
	  double phi_tmp_j = mget(phi, j, m);
	  // minc(doc_zbar_zbar_t, k, j, (phi_tmp - phi_tmp * phi_tmp_j
	  // Why were we doing this?
	  // minc(doc_zbar_zbar_t, k, j,
	  //   (- phi_tmp * phi_tmp_j
	  //   / (N * N)));
	}

	++m;
      }
    }
  }
}


static double InferDoc(const Document& doc,
		       bool fit_doc_ips,
		       int doc_index,
		       gsl_matrix* phi,
		       gsl_vector* gamma,
		       Posterior* posterior,
		       bool fit_doc,
		       double convergence_criterion) {
  double* digamma_gamma = new double[posterior->number_topics];
  gsl_vector* gamma_tmp = gsl_vector_calloc(gamma->size);
  for (int k=0; k < posterior->number_topics; ++k) {
    int n=0;
    for (int i=0; i < doc.unique_terms(); ++i) {
      for (int j=0; j < doc.counts(i); ++j) {
	mset(phi, k, n, 1.0 / posterior->number_topics);
	++n;
      }
    }
  }

  int iterations = 0;
  bool converged = false;
  double last_lhood = 0.0;
  double current_lhood = 0.0;
  double phi_sum;
  double phi_tmp;
  gsl_vector_view current_doc_ips = gsl_matrix_row(posterior->doc_ips, doc_index);
  gsl_vector* tmp_1 = gsl_vector_alloc(posterior->number_topics);
  gsl_vector* tmp_ip_contribution = gsl_vector_alloc(posterior->ip_dimension);
  gsl_vector* term2_by_topic = gsl_vector_alloc(posterior->number_topics);
  gsl_vector* term3_all_phi_by_topic = gsl_vector_calloc(posterior->number_topics); 
  gsl_vector* term4_by_topic = gsl_vector_calloc(posterior->number_topics);
  //  gsl_vector* term3_test = gsl_vector_calloc(posterior->number_topics);
  
  // Compute doc_ips x (etas) / N / sigma
  gsl_blas_dgemv(CblasTrans,
		 1.0 / (doc.term_counts_total() * posterior->dispersion),
		 posterior->etas,
		 &current_doc_ips.vector,
		 0.0,
		 term2_by_topic);

  // Compute 1.0 / 2N^2 \sigma [ 2 (\eta^T \phi) \eta
  // Note that this includes all phis.  We remove individual
  // phis (\phi_{-j}) and update this sum as we iterate.
  for (int ip_d=0; ip_d < posterior->ip_dimension; ++ip_d) {
    gsl_vector_view etas_n = gsl_matrix_row(posterior->etas, ip_d);
    double etas_n_sum = sum(&etas_n.vector);
    for (int k=0; k < posterior->number_topics; ++k) {
      // No need to go through all words, because phi starts out at 1/k.
      // We may need to set this more frequently if floating-point errors
      // build up.
      vinc(term3_all_phi_by_topic, k,
	   (vget(&etas_n.vector, k)
	    * etas_n_sum
	    / (double) posterior->number_topics));
    }
  }
  // Note that here we multiply by doc.term_counts_total(), which removes
  // one copy from the denominator.
  gsl_vector_scale(term3_all_phi_by_topic,
		   (1.0
		    / doc.term_counts_total()
		    / posterior->dispersion));

  // Compute 1.0 / 2N^2 \sigma (\eta \circ \eta).
  for (int ip_d=0; ip_d < posterior->ip_dimension; ++ip_d) {
    gsl_vector_view etas_n = gsl_matrix_row(posterior->etas, ip_d);
    gsl_vector_memcpy(tmp_1, &etas_n.vector);
    gsl_vector_mul(tmp_1, tmp_1);
    gsl_vector_add(term4_by_topic, tmp_1);
  }
  gsl_vector_scale(term4_by_topic,
		   (1.0
		    / 2.0
		    / (doc.term_counts_total() * doc.term_counts_total())
		    / posterior->dispersion));

  double converged_rate = 1.0;
  int next_check = 5;
  int last_check = 0;
  while (iterations < kMaxLDAIterations && !converged) {
    bool phi_converged = false;
    double last_phi_l = 1.0;
    double phi_l = 2.0;

    for (int k=0; k < posterior->number_topics; ++k) {
      digamma_gamma[k] = gsl_sf_psi(vget(gamma, k));
    }

    int tmp = 0;
    while (!phi_converged) {
      phi_converged = true;
      int n=0;
     for (int i = 0;
	  i < doc.unique_terms();
	  ++i) {
       //       if (iterations > 0 && (tmp + i) % 3 == 0) {
       //	 continue;
       //       }
      const int kTerm = doc.terms(i);
      const int kTermCount = doc.counts(i);
      for (int term = 0; term < kTermCount; ++term) {
	/*
	gsl_vector_set_zero(term3_test);

	{
	  int m=0;
	  for (int j=0; j < doc.unique_terms(); ++j) {
	    for (int tj = 0; tj < doc.counts(j); ++tj) {
	      if (m == n) {
		++m;
		continue;
	      }
	      gsl_vector_view phi_term = gsl_matrix_column(phi, m);
	      
	      for (int ii=0; ii < posterior->ip_dimension; ++ii) {	
		double ip_eta;
		gsl_vector_view eta = gsl_matrix_row(posterior->etas, ii);
		gsl_blas_ddot(&eta.vector, &phi_term.vector, &ip_eta);
		for (int k=0; k < posterior->number_topics; ++k) {
		  vinc(term3_test, k,
		       mget(posterior->etas, ii, k) * ip_eta);
		}
	      }
	      ++m;
	    }
	  }
	  gsl_vector_scale(term3_test,
			   1.0 / doc.term_counts_total()
			   / doc.term_counts_total() / posterior->dispersion);
	}
	*/

	phi_sum = 0.0;

	//     for (int term = 0; term <  kCount; ++term) {
	// Compute doc_ips x (etas)
	gsl_vector_view phi_term = gsl_matrix_column(phi, n);

	// Find the contribution of this term to term3.
	gsl_blas_dgemv(CblasNoTrans,
		       1.0,
		       posterior->etas,
		       &phi_term.vector,
		       0.0,
		       tmp_ip_contribution);
	gsl_blas_dgemv(CblasTrans,
		       (1.0 / doc.term_counts_total()
			/ doc.term_counts_total()
			/ posterior->dispersion),
		       posterior->etas,
		       tmp_ip_contribution,
		       0.0,
		       tmp_1);
	// Subtract out any contribution of the old term (we add the new
	// term in below).
	gsl_vector_sub(term3_all_phi_by_topic, tmp_1);

	// REMOVE AFTER TESTING!
	//gsl_vector_memcpy(term3_all_phi_by_topic, term3_test);

	for (int k=0; k < posterior->number_topics; ++k) {
	  // Note: topics are expected log probabilities.
	  double term1 = digamma_gamma[k] + mget(posterior->topics, k, kTerm);
	  double term2 = vget(term2_by_topic, k);
	  double term3 = vget(term3_all_phi_by_topic, k);
	  double term4 = vget(term4_by_topic, k);

	  if (fit_doc_ips) {
	    phi_tmp = term1 + term2 - term3 - term4;
	  } else {
	    phi_tmp = term1;
	  }

	  // phi_tmp = (1.0 / kCount * phi_tmp
	  // + (kCount - 1.0) / kCount * vget(&phi_term.vector, k));
	  vset(&phi_term.vector, k, phi_tmp);
	  if (k > 0) {
	    phi_sum = log_sum(phi_sum, phi_tmp);
	  } else {
	    phi_sum = phi_tmp;
	  }
	}
	for (int k=0; k < posterior->number_topics; ++k) {
	  vset(&phi_term.vector, k, exp(vget(&phi_term.vector, k) - phi_sum));
	  if (vget(&phi_term.vector, k) < 1e-30) {
	    vset(&phi_term.vector, k, 1e-30);
	  }
	  //	  vinc(gamma_tmp, k, vget(&phi_term.vector, k));
	}

	// Finally, re-add the contribution of the term using the new
	// phi values.
	gsl_blas_dgemv(CblasNoTrans,
		       1.0,
		       posterior->etas,
		       &phi_term.vector,
		       0.0,
		       tmp_ip_contribution);
	gsl_blas_dgemv(CblasTrans,
	       (1.0 / doc.term_counts_total()
			/ doc.term_counts_total()
			/ posterior->dispersion),
		       posterior->etas,
		       tmp_ip_contribution,
		       0.0,
		       tmp_1);
	gsl_vector_add(term3_all_phi_by_topic, tmp_1);
	++n;
      }
     }
    }

    if (true || iterations >= next_check
	|| iterations >= kMaxLDAIterations - 1) {
      next_check =  iterations + 2 * ((-log(convergence_criterion)
				       + log(converged_rate))
				      - log(iterations - last_check));
      next_check = max(iterations + 2, next_check);
      next_check = min(iterations + 8, next_check);
      
      // Update the topic sufficient statistics for this document.
      // Note that and E[ zbar ] is updated by DocLikelihood.
      current_lhood = DocLikelihood(doc,
				    doc_index,
				    posterior,
				    phi,
				    gamma,
				    true);
      converged = Converged(last_lhood, current_lhood,
			    convergence_criterion
			    //			  * (iterations - last_check),
			    ,
			    &converged_rate);
      last_check = iterations;
      last_lhood = current_lhood;
    }
    
    if (fit_doc) {
      gsl_vector_set_all(gamma_tmp, posterior->alpha);
      for (int k=0; k < gamma->size; ++k) {
	for (int n=0; n < doc.term_counts_total(); ++n) {
	  vinc(gamma_tmp, k, mget(phi, k, n));
	}
      }
      gsl_vector_memcpy(gamma, gamma_tmp);
    }

    ++iterations;
  }

  if (!fit_doc) {
    delete[] digamma_gamma;
    gsl_vector_free(gamma_tmp);
    gsl_vector_free(tmp_1);
    gsl_vector_free(tmp_ip_contribution);
    gsl_vector_free(term3_all_phi_by_topic);
    gsl_vector_free(term2_by_topic);
    gsl_vector_free(term4_by_topic);
    return current_lhood;
  }

  // Update the topic sufficient statistics and E[ zbar ]
  // for this document.
  int n = 0;
  const double N = doc.term_counts_total();
  for (int i=0; i < doc.unique_terms(); ++i) {
    const int kTermCount = doc.counts(i);
    const int kTerm = doc.terms(i);
    for (int j=0; j < kTermCount; ++j) {
      gsl_vector_view phi_term = gsl_matrix_column(phi, n);
      gsl_vector_view topic_suffstats =
	gsl_matrix_column(posterior->topic_suffstats,
			  kTerm);
      gsl_blas_daxpy(1.0, &phi_term.vector, &topic_suffstats.vector);
      ++n;
    }
  }

  // Update E[ zbar zbar' ] for this document.
  UpdateZBarZBarT(posterior->doc_zbar_zbar_t[doc_index],
		  doc,
		  posterior,
		  phi,
		  N);

  delete[] digamma_gamma;
  gsl_vector_free(gamma_tmp);
  gsl_vector_free(tmp_1);
  gsl_vector_free(tmp_ip_contribution);
  gsl_vector_free(term3_all_phi_by_topic);
  gsl_vector_free(term2_by_topic);
  gsl_vector_free(term4_by_topic);
  //  gsl_vector_free(term3_test);

  current_lhood = DocLikelihood(doc,
				doc_index,
				posterior,
				phi,
				gamma,
				false);
  return current_lhood;

}
		       

static double LDAEStep(const Corpus& corpus,
		       bool fit_doc_ips,
		       Posterior* posterior,
		       bool fit_doc,
		       double convergence_criterion) {
  double likelihood = 0;
  
  int current_number_terms = 1000;
  gsl_matrix* phi = gsl_matrix_calloc(posterior->number_topics,
				      current_number_terms);
  gsl_matrix_set_zero(posterior->topic_suffstats);

  for (int d=0; d < corpus.number_docs(); ++d) {
    // Only need to resize this log(D) times on average.
    const Document& doc = *corpus.docs(d);
    if (doc.term_counts_total() > current_number_terms) {
      current_number_terms = doc.term_counts_total() + 100;
      gsl_matrix_free(phi);
      phi = gsl_matrix_calloc(posterior->number_topics,
			      current_number_terms);
    }

    gsl_vector_view doc_gamma = gsl_matrix_row(posterior->doc_gammas, d);

    gsl_vector_set_all(&doc_gamma.vector, ((double) doc.term_counts_total())
		       / posterior->number_topics + FLAGS_alpha);
    double doc_likelihood = InferDoc(doc,
				     fit_doc_ips,
				     d,
				     phi,
				     &doc_gamma.vector,
				     posterior,
				     fit_doc,
				     convergence_criterion);
    likelihood += doc_likelihood;

    double tmp_lhood = DocLikelihood(doc,
				     d,
				     posterior,
				     phi,
				     &doc_gamma.vector,
				     true);
  }

  gsl_matrix_free(phi);

  return likelihood;
}


void TestInferDoc() {
  Document* doc = new Document("102_foo 0:1 1:1");
  Corpus* corpus = new Corpus;
  User* user = new User("foo");

  corpus->AddDocument(doc);

  corpus->AddUser(user);
  
  Posterior* post = new Posterior(*corpus, 2, 1);
  gsl_matrix* phi = gsl_matrix_calloc(2, 2);
  mset(post->etas, 0, 0, 2.0);
  mset(post->etas, 0, 1, 0.5);

  mset(post->topics, 0, 0, 0.5);
  mset(post->topics, 1, 0, 0.5);
  mset(post->doc_ips, 0, 0, 1.0);
  post->dispersion = 0.04;
  post->alpha = 0.9;

  gsl_vector_view gammas = gsl_matrix_row(post->doc_gammas, 0);
  gsl_vector_set_all(&gammas.vector, ((double) doc->term_counts_total())
		     / post->number_topics + post->alpha);

  double likelihood = InferDoc(*doc,
			       true,
			       0,
			       phi,
			       &gammas.vector,
			       post,
			       true,
			       1e-6);

  printf("0 0 %lf\n", mget(post->doc_zbars, 0, 0));
  printf("0 1 %lf\n", mget(post->doc_zbars, 0, 1));
  if (likelihood < -2.329899 - 1e-5) {
    printf("Test failed.  Likelihood is too low.\n");
  }

  delete corpus;
  delete post;
  gsl_matrix_free(phi);

  // Now try a doc with multiple words.
  doc = new Document("102_foo 0:30 1:2 2:1");
  user = new User("foo");

  corpus = new Corpus;
  corpus->AddDocument(doc);
  corpus->AddUser(user);
  post = new Posterior(*corpus, 2, 1);
  phi = gsl_matrix_calloc(2, 144);
  mset(post->etas, 0, 0, 2.0);
  mset(post->etas, 0, 1, 0.5);

  mset(post->topics, 0, 0, 0.1);
  mset(post->topics, 1, 0, 0.8);
  mset(post->topics, 0, 1, 0.3);
  mset(post->topics, 1, 1, 0.05);
  mset(post->topics, 0, 2, 0.6);
  mset(post->topics, 1, 2, 0.15);
  mset(post->doc_ips, 0, 0, 0.3);
  post->dispersion = 0.04;
  post->alpha = 0.9;

  gammas = gsl_matrix_row(post->doc_gammas, 0);
  gsl_vector_set_all(&gammas.vector, ((double) doc->term_counts_total())
  		     / post->number_topics + post->alpha);

  likelihood = InferDoc(*doc,
			true,
			0,
			phi,
			&gammas.vector,
			post,
			true,
			1e-6);
  if (likelihood < 21.29663) {
    printf("Test failed.  Likelihood is too low.\n");
  }

  printf("0 0 %lf\n", mget(post->doc_zbars, 0, 0));
  printf("0 1 %lf\n", mget(post->doc_zbars, 0, 1));

  delete corpus;
  delete post;

  gsl_matrix_free(phi);

}
			 

static double LDAMStep(const Corpus& corpus,
		       Posterior* post) {
  for (int k=0; k < post->number_topics; ++k) {
    gsl_vector_view terms = gsl_matrix_row(
      post->topic_suffstats, k);
    const double kLogTotal = log(sum(&terms.vector));
    assert(post->number_terms == terms.vector.size);
    for (int w=0; w < post->number_terms; ++w) {
      double ss = vget(&terms.vector, w);
      if (ss > 0.0) {
	mset(post->topics, k, w,
	     log(ss) - kLogTotal);
      } else {
	mset(post->topics, k, w, -20.0);
      }
      if (mget(post->topics, k, w) < -20.0) {
	mset(post->topics, k, w, -20.0);
      }
    }
    gsl_vector_view topic = gsl_matrix_row(post->topics, k);
    double s = 0.0;
    for (int i=0; i < topic.vector.size; ++i) {
      s += exp(vget(&topic.vector, i));
    }
  }
  return 0.0;
}

void Posterior::InitializeGamma(const Corpus& corpus) {
  for (int d=0; d < corpus.number_docs(); ++d) {
    const Document& doc = *corpus.docs(d);
    for (int k=0; k < number_topics; ++k) {
      mset(doc_gammas, d, k,
	   alpha + doc.term_counts_total()
	   / number_topics);
    }
  }
}


static void PrintLikelihood(const char* message,
			    const Corpus& corpus,
			    Posterior* post,
			    double doc_lhood,
			    double eta_likelihood) {

  double vote_lhood = post->UserIdealLikelihood(corpus,
						FLAGS_sigma_u);
  vote_lhood += post->DocIdealLikelihood(corpus,
					 FLAGS_dispersion);
  vote_lhood += post->VoteLikelihood(corpus);
  vote_lhood += eta_likelihood;
  printf("Evidence bound %s: %lf\n  vote: %lf\n  user: %lf\n  doc: %lf\n",
	 message,
	 doc_lhood + vote_lhood,
	 post->VoteLikelihood(corpus),
	 post->UserIdealLikelihood(corpus,
				   FLAGS_sigma_u),
	 post->DocIdealLikelihood(corpus,
				  FLAGS_dispersion));

}


double Posterior::FitLDA(const Corpus& corpus,
			 bool fit_doc_ips,
			 double convergence_criterion) {
  double lda_likelihood = 0.0;
  // If we're resuming, run the m step iff we have
  // already run the e step.
  static bool resume_and_e_step = false;
  if (FLAGS_mode != "resume") {
    resume_and_e_step = true;
  }
  if (resume_and_e_step
      && (FLAGS_mode == "fit"
	  || FLAGS_mode == "fit_predict"
	  || FLAGS_mode == "resume"
	  || FLAGS_mode == "lda")) {
    /*    if (iteration > 3) {
      lda_likelihood = LDAEStep(corpus, fit_doc_ips,
				this, true, convergence_criterion);
    }
    printf("pre m step %lf\n", lda_likelihood);
    */
    if (true || iteration < 60) {
      printf("performing M step in iteration %d.\n", iteration);
      LDAMStep(corpus, this);
    } else {
      printf("skipping m step in iteration %d.\n", iteration);
    }
  } else if (FLAGS_mode == "fixed_topics"
	     || FLAGS_mode == "fixed_topics_predict") {
  } else if (FLAGS_mode != "infer_docs"
	     && resume_and_e_step) {
    LogFatal(StringPrintf("Error.  Unhandled mode: %s.\n",
			  FLAGS_mode.c_str()));
  }
  lda_likelihood = LDAEStep(corpus, fit_doc_ips,
			    this, true, convergence_criterion);
  resume_and_e_step = true;

  return lda_likelihood;
}


void GradientStats::PushHessian(gsl_vector* ip) {
  gsl_vector* tmp = gsl_vector_alloc(gradient_batch->size);
  bool fixed1;
  bool fixed2;
  FixHessian(hessian_batch, &fixed1, &fixed2);

  if (false && (fixed2 || fixed1)) {
    gsl_vector_free(tmp);
    gsl_vector_scale(gradient_batch, coefficient / pow(number_iterations, 0.65));
    ++number_iterations;
    gsl_vector_add(ip, gradient_batch);
    gsl_vector_set_zero(gradient_batch);
    gsl_matrix_set_zero(hessian_batch);
    update_count = 0;
    return;
  }

  // gsl_matrix_scale(hessian_batch);

  Solve(hessian_batch,
	gradient_batch,
	tmp);
  gsl_vector_scale(tmp,
		   (100.0 + coefficient)
		   / (100.0 + pow(number_iterations, 0.6)));
  

  double step_norm;
  gsl_blas_ddot(tmp, tmp, &step_norm);
  // No self-respecting step should be larger than 1.0.
  // Anything over this can always be adjusted later.
  if (fixed2 && sqrt(step_norm) > 0.1) {
    gsl_vector_scale(tmp, 0.1 / sqrt(step_norm));
  } else if (fixed1 && sqrt(step_norm) > 10) {
    gsl_vector_scale(tmp, 10 / sqrt(step_norm));
  } else if (sqrt(step_norm) > 1.0) {
    gsl_vector_scale(tmp, 1.0 / sqrt(step_norm));
  } else if (sqrt(step_norm) < 1e-5) {
    //    gsl_vector_scale(tmp, 
    //		     (1.0 / sqrt(step_norm)) / (number_iterations + 1.0)
    //		     + 1.0);
  }

  gsl_vector_add(ip, tmp);
  gsl_vector_set_zero(gradient_batch);
  gsl_matrix_set_zero(hessian_batch);
  ++number_iterations;
  gsl_vector_free(tmp);
  update_count = 0;
}


void GradientStats::Flush() {
  update_count = 0;
  gsl_vector_set_zero(gradient_batch);
  gsl_matrix_set_zero(hessian_batch);
}


void GradientStats::UpdateHessian(gsl_vector* gradient,
				  gsl_matrix* hessian) {
  /*
  for (int i=0; i < gradient->size; ++i) {
    if (fabs(vget(gradient, i)) > 50000.0) {
      vset(gradient, i, 50000.0
	   * (vget(gradient, i) > 0.0 ? 1.0 : -1.0));
    }
  }

  for (int i=0; i < hessian->size1; ++i) {
    for (int j=0; j < hessian->size2; ++j) {
      if (fabs(mget(hessian, i, j)) > 1000.0) {
	mset(hessian, i, j, 1000.0
	     * (mget(hessian, i, j) > 0.0 ? 1.0 : -1.0));
      }
    }
  }
  */

  gsl_vector_add(gradient_batch, gradient);
  gsl_matrix_add(hessian_batch, hessian);
  ++update_count;
}


void GradientStats::Push(double distance,
			 gsl_vector* gradient,
			 gsl_matrix* hessian,
			 gsl_vector* ip) {
  //  coefficient = (coefficient * 0.9
  //		 + distance * 0.1);
  double gradient_length;
  gsl_blas_ddot(gradient, gradient, &gradient_length);
  gradient_length = sqrt(gradient_length);
  gradient_average_length = (gradient_average_length * 0.9
			     + gradient_length * 0.1);

  // If the gradient average is nontrivial, scale upward.
  // If it's too small, scale it up.
  /*
  if (gradient_average_length > 1e-5) {
    gsl_vector_scale(gradient, 1.0 / gradient_average_length);
  } else {
    //    gsl_vector_scale(gradient, 1e5);
  }
  */
  
  gsl_matrix_scale(hessian_batch,
		   (1.0 * number_iterations - 1)
		   / number_iterations);
  gsl_matrix_scale(hessian, 1.0 / number_iterations);
  gsl_matrix_add(hessian_batch, hessian);
  gsl_vector_add(gradient_batch, gradient);

  if (UniformRNG() < 0.1) {
    gsl_vector_scale(gradient_batch,
		     (coefficient + 100.0)
		     / (5.0 + pow(number_iterations, 0.71)));
    gsl_vector* tmp = gsl_vector_alloc(
      gradient_batch->size);
    Solve(hessian_batch,
	  gradient_batch,
	  tmp);
    //    gsl_vector_add(ip, gradient_batch);

    // Don't do any steps which are too large
    // relative to the current ip.
    for (int i=0; i < tmp->size; ++i) {
      if (fabs(vget(tmp, i)) > 0.5) {
	vset(tmp, i, 0.5
	     * (vget(tmp, i) > 0.0 ? 1.0 : -1.0));
      }
    }

    gsl_vector_add(ip, tmp);
    gsl_vector_set_zero(gradient_batch);
    ++number_iterations;
    gsl_vector_free(tmp);
  }
}


void GradientStats::PushSecondOrder(gsl_vector* gradient,
				    gsl_vector* ip) {
  // Don't do any steps which are too large
  // relative to the current ip.
  for (int i=0; i < gradient->size; ++i) {

    if (fabs(vget(gradient, i)) > 100.0) {
      vset(gradient, i, 100.0
	   * (vget(gradient, i) > 0.0 ? 1.0 : -1.0));
      
    }
  }
  gsl_vector_scale(gradient,
		   coefficient
		   / (pow(number_iterations, 0.501)));

  gsl_vector_add(ip, gradient);
    
  gsl_vector_set_zero(gradient_batch);
  ++number_iterations;
}


static void UserGradient(const Posterior& post,
			 const Vote& vote,
			 const Corpus& corpus,
			 int user,
			 gsl_vector* tmp_lambda,
			 gsl_vector* tmp_kappa,			 
			 gsl_vector* user_ip,
			 gsl_vector* doc_ip,
			 gsl_vector* gradient) {
    double inner_product;
    gsl_blas_ddot(user_ip, doc_ip, &inner_product);
    double kappa_inner_product;
    gsl_blas_ddot(doc_ip, doc_ip, &kappa_inner_product);
    double lambda_inner_product;
    gsl_blas_ddot(user_ip, user_ip, &lambda_inner_product);

    double term_exp = exp(inner_product);
    // For convenience:
    double term_f = (term_exp + 1);

    double term1 = 0.0;
    if (vote.positive()) {
      term1 = 1.0;
    }

    double term2 = (post.sigma_lambda * post.sigma_lambda
		    * kappa_inner_product
		    + post.sigma_kappa * post.sigma_kappa
		    * lambda_inner_product);
    double term3;
    if (inner_product > 50.0) {
      term3 = 0.0;
    } else if (inner_product < -50.0) {
      term3 = 0.0;
    } else {
      term3 = ((term_exp / term_f)
	       * (1.0
		  - (term_exp / term_f)
		  - 2.0 * (term_exp / term_f)
		  + 2.0 * (term_exp * term_exp) / (term_f * term_f)));
    }
    double term_kappa = (FLAGS_vote_replication
			 * (term1
			    - (term_exp / term_f)
			    - term2 * term3 / 2.0));
    if (inner_product > 50.0) {
      term_kappa = FLAGS_vote_replication * (term1 - 1);
    } else if (inner_product < -50.0) {
      term_kappa = FLAGS_vote_replication * term1;
    }
    gsl_vector_memcpy(tmp_kappa, doc_ip);
    gsl_vector_scale(tmp_kappa, term_kappa);

    const int kUserNumberVotes =
      corpus.users(user)->number_votes();

    // Note that we multiply by vote_replication.
    double term_lambda = - ((1.0 / (post.sigma_u * post.sigma_u)
			     / kUserNumberVotes)
			    + (FLAGS_vote_replication
			       * post.sigma_kappa * post.sigma_kappa / 1.0
			       * (term_exp / term_f
				  - term_exp * term_exp / (term_f * term_f))));
    if (inner_product > 50.0 || inner_product < -50.0) {
      term_lambda = -(1.0 / (post.sigma_u * post.sigma_u)
		      / kUserNumberVotes);
    }

    gsl_vector_memcpy(tmp_lambda, user_ip);
    gsl_vector_scale(tmp_lambda, term_lambda);

    gsl_vector_set_zero(gradient);
    gsl_vector_add(gradient, tmp_kappa);
    gsl_vector_add(gradient, tmp_lambda);

    if (FLAGS_experimental_priors) {
      const char* user_str = corpus.users(user)->id().c_str();
      if (!strcmp(user_str, "300041")
	  || !strcmp(user_str, "400440")) {
	// This is Enzi (300041) or Donald Young (400440).
	// The prior is -4.0.
	//	vinc(gradient, 1, -1.0 / (post.sigma_u * post.sigma_u
	//			  * kUserNumberVotes));
	vinc(gradient, 1, (vget(user_ip, 1)
			   / (post.sigma_u * post.sigma_u * kUserNumberVotes)
			   - (vget(user_ip, 1)
			      / (kPriorVariance * kUserNumberVotes))
			   - 4.0 / (kPriorVariance
				    * kUserNumberVotes)));
	//	printf("User 300041, doc %s.\n",
	//     corpus.docs()[vote.doc()]->id().c_str());
      } else if (!strcmp(user_str, "300059")
		 || !strcmp(user_str, "400425")) {
	// This is Kennedy (300025) or Henry Waxman (400425).
	// Set the prior to 4.0.
	vinc(gradient, 1, (vget(user_ip, 1)
			   / (post.sigma_u * post.sigma_u * kUserNumberVotes)
			   - (vget(user_ip, 1)
			      / (kPriorVariance * kUserNumberVotes))
			   + 4.0 / (kPriorVariance
				    * kUserNumberVotes)));
	// vinc(gradient, 1, 1.0 / (post.sigma_u * post.sigma_u
	//			 * kUserNumberVotes));
      }
    }

    // Pretend there's no gradient in the first dimension.
    if (FLAGS_encode_difficulty) {
      gsl_vector_set(gradient, 0, 0.0);
    }
}


static void DocGradient(const Vote& vote,
			const Corpus& corpus,
			int doc,
			bool last_iteration,
			Posterior* post,
			gsl_vector* tmp_lambda,
			gsl_vector* tmp_kappa,			 
			gsl_vector* tmp_eta,
			gsl_vector* user_ip,
			gsl_vector* doc_ip,
			gsl_vector* gradient) {
    // Next, update the doc ideal point.
    double inner_product;
    gsl_blas_ddot(user_ip, doc_ip, &inner_product);
    double kappa_inner_product;
    gsl_blas_ddot(doc_ip, doc_ip, &kappa_inner_product);
    double lambda_inner_product;
    gsl_blas_ddot(user_ip, user_ip, &lambda_inner_product);

    double term1 = 0.0;
    if (vote.positive()) {
      term1 = 1.0;
    }

    double term_exp = exp(inner_product);
    // For convenience:
    double term_f = (term_exp + 1);

    double term2 = (post->sigma_lambda * post->sigma_lambda
		    * kappa_inner_product
		    + post->sigma_kappa * post->sigma_kappa
		    * lambda_inner_product);
    double term3;
    if (inner_product > 50.0) {
      term3 = 0.0;
    } else if (inner_product < -50.0) {
      term3 = 0.0;
    } else {
      term3 = ((term_exp / term_f)
	       * (1.0
		  - (term_exp / term_f)
		  - 2.0 * term_exp / term_f
		  + 2.0 * (term_exp * term_exp) / (term_f * term_f)));
    }

    double term_lambda = (FLAGS_vote_replication
			  * (term1
			     - (term_exp / term_f)
			     - (term2 * term3 / 2.0)));
    if (inner_product > 50.0) {
      term_lambda = FLAGS_vote_replication * (term1 - 1.0 - term2 * term3 / 2.0);
    } else if (inner_product < -50.0) {
      term_lambda = FLAGS_vote_replication * (term1 - term2 * term3 / 2.0);
    }
    gsl_vector_memcpy(tmp_lambda, user_ip);
    gsl_vector_scale(tmp_lambda, term_lambda);

    const int kDocNumberVotes =
      corpus.docs(doc)->number_votes();
    double dispersion = post->dispersion;
    // If this is the last iteration, allow the model
    // slightly more flexibility in fitting doc ideal
    // points.
    if (last_iteration) {
      double new_dispersion = sqrt(dispersion);
      dispersion = new_dispersion > dispersion ?
	new_dispersion : dispersion;
    }
    double term_kappa = - (1.0 / dispersion / kDocNumberVotes
			   + (FLAGS_vote_replication
			      * post->sigma_lambda * post->sigma_lambda / 1.0
			      * (term_exp / term_f
				 - term_exp * term_exp / (term_f * term_f))));
    if (inner_product > 50.0 || inner_product < -50.0) {
      term_kappa = -(1.0 / dispersion
		     / kDocNumberVotes);
    }
    gsl_vector_memcpy(tmp_kappa, doc_ip);
    gsl_vector_scale(tmp_kappa, term_kappa);

    gsl_vector_view doc_zbar = gsl_matrix_row(
      post->doc_zbars, doc);
    gsl_blas_dgemv(CblasNoTrans,
		   1.0,
		   post->etas,
		   &doc_zbar.vector,
		   0.0,
		   tmp_eta);
    gsl_vector_scale(tmp_eta,
		     (1.0 / dispersion
		      / kDocNumberVotes));

    gsl_vector_set_zero(gradient);
    gsl_vector_add(gradient, tmp_kappa);
    gsl_vector_add(gradient, tmp_lambda);

    gsl_vector_add(gradient, tmp_eta);
}


void Posterior::FitIdealPoints(const Corpus& corpus,
			       int ideal_iteration,
			       bool second_order,
			       bool sparse,
			       bool only_users,
			       bool only_docs) {
  gsl_vector* user_gradient = gsl_vector_calloc(ip_dimension);
  gsl_vector* tmp = gsl_vector_alloc(ip_dimension);
  gsl_vector* tmp2 = gsl_vector_alloc(ip_dimension);
  gsl_vector* tmp3 = gsl_vector_alloc(ip_dimension);
  gsl_matrix* hessian = gsl_matrix_alloc(ip_dimension, ip_dimension);
  gsl_vector* doc_gradient = gsl_vector_calloc(ip_dimension);
  gsl_vector* tmp_kappa = gsl_vector_calloc(ip_dimension);
  gsl_vector* tmp_lambda = gsl_vector_calloc(ip_dimension);
  gsl_vector* tmp_eta = gsl_vector_calloc(ip_dimension);

  static int doc_sparsity = 10;
  doc_sparsity *= 2;

  static int user_sparsity = 10;
  user_sparsity *= 2;

  for (int v=0; v < corpus.number_votes(); ++v) {
    const Vote& vote = *corpus.votes(v);

    int user = vote.user();
    int doc = vote.doc();

    // Don't update all users or all docs in this
    // iteration.
    if (ideal_iteration >= 0
	&& (doc % kDocMod == ideal_iteration % kDocMod
	    && user % kUserMod == ideal_iteration % kUserMod)) {
      continue;
    }

    // Do the first N iterations on a limited set of docs
    // to try to fit their ideal points first.
    if (sparse && doc > doc_sparsity) {
      continue;
    }

    // If we're still initializing, gradually increase
    // the number of users we consider, starting with
    // the users with fixed priors.
    if (sparse && !FLAGS_experimental_priors
	&& user > user_sparsity) {
      continue;
    }
    if (sparse && FLAGS_experimental_priors) {
      const char* user_str = corpus.users(user)->id().c_str();
      if (strcmp(user_str, "300041")
	  && strcmp(user_str, "400440")
	  && strcmp(user_str, "300059")
	  && strcmp(user_str, "400425")) { 
	if (user > user_sparsity) {
	  continue;
	}
      }
    }

    gsl_vector_view user_ip = gsl_matrix_row(
      user_ips, user);
    gsl_vector_view doc_ip = gsl_matrix_row(
      doc_ips, doc);

    // Only update the users if we are fitting or resuming
    // a fit and we intend to update the user.
    if ((FLAGS_mode == "fit"
	 || FLAGS_mode == "fit_predict"
	 || FLAGS_mode == "resume"
	 || FLAGS_mode == "ideal"
	 || FLAGS_mode == "fixed_topics"
	 || FLAGS_mode == "fixed_topics_predict")
	&& (!only_docs)) {
      UserGradient(*this,
		   vote,
		   corpus,
		   user,
		   tmp_lambda,
		   tmp_kappa,
		   &user_ip.vector,
		   &doc_ip.vector,
		   user_gradient);
      // Change the norm of the gradient to be 1e-4.
      for (int i=0; i < ip_dimension; ++i) {
	gsl_vector_memcpy(tmp2, &user_ip.vector);
	vinc(tmp2, i, 1e-4);
	gsl_vector_view hessian_column = gsl_matrix_row(hessian, i);
	UserGradient(*this,
		     vote,
		     corpus,
		     user,
		     tmp_lambda,
		     tmp_kappa,
		     tmp2,
		     &doc_ip.vector,
		     &hessian_column.vector);
	gsl_vector_sub(&hessian_column.vector, user_gradient);
	gsl_vector_scale(&hessian_column.vector, -1e4);
      }
      
      // Calculate likelihoods and gradients.
      if (sparse || !(second_order && FLAGS_batch_update)) {
	bool fixed;
	FixHessian(hessian, &fixed, &fixed);
	//	Solve(hessian,
	//user_gradient,
	//tmp3);
	if (second_order) {
	  user_update_stats[user]->PushSecondOrder(
	    tmp3,
	    &user_ip.vector);
	} else {
	  // Compute the well-scaled gradient, and put it into tmp3.
	  
	  double distance_tmp3;
	  double distance_gradient;
	  double inner_product_tmp3_gradient;
	  gsl_blas_ddot(tmp3, user_gradient,
			&inner_product_tmp3_gradient);
	  gsl_blas_ddot(user_gradient, user_gradient,
			&distance_gradient);
	  gsl_blas_ddot(tmp3, tmp3,
			&distance_tmp3);
	  distance_gradient = sqrt(distance_gradient);
	  distance_tmp3 = sqrt(distance_tmp3);
	  double distance = distance_tmp3;
	  if (distance_gradient > 0.0) {
	    distance = fabs((inner_product_tmp3_gradient
			     / distance_gradient));
	  }
	  
	  user_update_stats[user]->Push(distance,
					user_gradient,
					hessian,
					&user_ip.vector);
	}
     } else {
	if (FLAGS_encode_difficulty) {
	  for (int i=1; i < ip_dimension; ++i) {
	    mset(hessian, 0, i, 0.0);
	    mset(hessian, i, 0, 0.0);
	  }
	  mset(hessian, 0, 0, 1.0);
	  vset(user_gradient, 0, 0.0);
	}
	user_update_stats[user]->UpdateHessian(user_gradient,
					       hessian);
      }
      if (FLAGS_encode_difficulty &&
	  (!second_order)) {
	vset(&user_ip.vector, 0, 1.0);
      }
      
    }
    if (!only_users) {
      DocGradient(vote,
		  corpus,
		  doc,
		  last_iteration,
		  this,
		  tmp_lambda,
		  tmp_kappa,
		  tmp_eta,
		  &user_ip.vector,
		  &doc_ip.vector,
		  doc_gradient);

      // Change the norm of the gradient to be 1e-5.
      for (int i=0; i < ip_dimension; ++i) {
	gsl_vector_memcpy(tmp2, &doc_ip.vector);
	vinc(tmp2, i, 1e-4);
	gsl_vector_view hessian_column = gsl_matrix_row(hessian, i);
	DocGradient(vote,
		    corpus,
		    doc,
		    last_iteration,
		    this,
		    tmp_lambda,
		    tmp_kappa,
		    tmp_eta,
		    &user_ip.vector,
		    tmp2,
		    &hessian_column.vector);
	gsl_vector_sub(&hessian_column.vector, doc_gradient);
	gsl_vector_scale(&hessian_column.vector, -1e4);
      }
      
      // Update the doc ideal point.
      if (sparse || !(second_order && FLAGS_batch_update)) {
	bool fixed;
	FixHessian(hessian, &fixed, &fixed);
	//	Solve(hessian,
	//	doc_gradient,
	//	tmp3);
	if (second_order) {
	  doc_update_stats[doc]->PushSecondOrder(
	    tmp3,
	    &doc_ip.vector);
	} else {
	  // Compute the well-scaled gradient, and put it into tmp3.

	  double distance_tmp3;
	  double distance_gradient;
	  double inner_product_tmp3_gradient;
	  gsl_blas_ddot(tmp3, doc_gradient,
			&inner_product_tmp3_gradient);
	  gsl_blas_ddot(doc_gradient, doc_gradient,
			&distance_gradient);
	  gsl_blas_ddot(tmp3, tmp3,
			&distance_tmp3);
	  distance_gradient = sqrt(distance_gradient);
	  distance_tmp3 = sqrt(distance_tmp3);
	  double distance = distance_tmp3;
	  if (distance_gradient > 0.0) {
	    distance = fabs((inner_product_tmp3_gradient
			     / distance_gradient));
	  }
	  
	  doc_update_stats[doc]->Push(distance,
				      doc_gradient,
				      hessian,
				      &doc_ip.vector);
	}
      } else {
	  doc_update_stats[doc]->UpdateHessian(doc_gradient,
					       hessian);
      }
    }
  }

  gsl_vector_free(user_gradient);
  gsl_vector_free(doc_gradient);
  gsl_vector_free(tmp_eta);
  gsl_vector_free(tmp_kappa);
  gsl_vector_free(tmp_lambda);
  gsl_vector_free(tmp);
  gsl_vector_free(tmp2);
  gsl_vector_free(tmp3);
  gsl_matrix_free(hessian);
}


double Posterior::UpdateEta(const Corpus& corpus) {
  gsl_matrix* e_x_xt = gsl_matrix_calloc(number_topics, number_topics);
  gsl_matrix* e_x_xt_tmp = gsl_matrix_calloc(number_topics, number_topics);
  gsl_vector* e_x_y = gsl_vector_calloc(number_topics);

  double eta_variance = 1.0 / corpus.number_docs();
  if (FLAGS_mode == "fit_predict"
      || FLAGS_mode == "fixed_topics_predict") {
    // Perform exact regression.
    gsl_blas_dgemm(CblasTrans,
		   CblasNoTrans,
		   1.0,
		   doc_zbars,
		   doc_zbars,
		   0.0,
		   e_x_xt_tmp);
    for (int i=0; i < number_topics; ++i) {
      // minc(e_x_xt, i, i, 10);
      // Note: we must regularize to ensure that doc IPs do not go off
      // to infinity.
      //      minc(e_x_xt, i, i,
      //   //	 corpus.number_docs()
      //   FLAGS_dispersion / eta_variance);
    }

    for (int i=0; i < ip_dimension; ++i) {
      gsl_vector_view doc_ip_i = gsl_matrix_column(doc_ips, i);
      // Note that we don't need to do this multiplication and solve
      // every time if we calculate (X'X)^{-1} X' ahead of time.
      // This is not a bottleneck in the code, though, so we leave it.
      gsl_blas_dgemv(CblasTrans,
		     1.0,
		     doc_zbars,
		     &doc_ip_i.vector,
		     0.0,
		     e_x_y);

      gsl_vector_view eta = gsl_matrix_row(etas, i);
      gsl_matrix_memcpy(e_x_xt, e_x_xt_tmp);
      Solve(e_x_xt, e_x_y, &eta.vector);
    }
    WriteEtas(*this);
    string filename = StringPrintf(
        "%s_etas.dat",
	FLAGS_checkpoint_prefix.c_str());
    mtx_fprintf(filename.c_str(), etas);
  }

  if (FLAGS_sigma_eta) {
    eta_variance = FLAGS_sigma_eta * FLAGS_sigma_eta;
  }

  if (!last_iteration || (FLAGS_mode != "fit_predict"
			  && FLAGS_mode != "fixed_topics_predict")) {
    for (int d=0; d < corpus.number_docs(); ++d) {
      gsl_matrix_add(e_x_xt, doc_zbar_zbar_t[d]);
    }
    
    for (int i=0; i < number_topics; ++i) {
      if (fabs(mget(e_x_xt, i, i)) < 1e-8) {
	minc(e_x_xt, i, i, 2e-8);
      }
      // minc(e_x_xt, i, i, 10);
      // Note: we must regularize to ensure that doc IPs do not go off
      // to infinity.
      minc(e_x_xt, i, i,
	   //	 corpus.number_docs()
	   FLAGS_dispersion / eta_variance);
    }
  
    for (int i=0; i < ip_dimension; ++i) {
      gsl_matrix_memcpy(e_x_xt_tmp, e_x_xt);

      gsl_vector_view doc_ip_i = gsl_matrix_column(doc_ips, i);
      // Note that we don't need to do this multiplication and solve
      // every time if we calculate (X'X)^{-1} X' ahead of time.
      // This is not a bottleneck in the code, though, so we leave it.
      gsl_blas_dgemv(CblasTrans,
		     1.0,
		     doc_zbars,
		     &doc_ip_i.vector,
		     0.0,
		     e_x_y);

      if (FLAGS_discerning_etas) {
	// Only explain the difficulty with the first half.
	// Only explain the discrimination with the last half.
	for (int k1 = 0; k1 < number_topics; ++k1) {
	  for (int k2 = 0; k2 < number_topics; ++k2) {
	    if ((k2 * ip_dimension) / number_topics != i) {
	      mset(e_x_xt_tmp, k1, k2, 0.0);
	    }
	    if ((k1 * ip_dimension) / number_topics != i) {
	      mset(e_x_xt_tmp, k1, k2, 0.0);
	    }
	  }
	  if ((k1 * ip_dimension) / number_topics != i) {
	    vset(e_x_y, k1, 0.0);
	    mset(e_x_xt_tmp, k1, k1, 1.0);
	  }
	}
      }

      gsl_vector_view eta = gsl_matrix_row(etas, i);
      Solve(e_x_xt_tmp, e_x_y, &eta.vector);
    
      // Use geometric smoothing to blend the old eta and the
      // current eta.
      // gsl_vector* tmp = gsl_vector_alloc(etas->size2);
      //       Solve(e_x_xt_tmp, e_x_y, tmp);
      
      //       gsl_vector_view eta = gsl_matrix_row(etas, i);
      //       gsl_vector_scale(&eta.vector, 7.1);
      //       gsl_vector_add(&eta.vector, tmp);
      //       gsl_vector_scale(&eta.vector, 1.0 / (7.1 + 1.0));
      
      //       gsl_vector_free(tmp);

    }
  }

  // Who cares if we do the above steps when we don't need to -- this
  // is less prone to bugs.
  if (FLAGS_fix_eta) {
    gsl_matrix_set_zero(etas);
    // Ideal point 1 is explained by the first two topics.
    mset(etas, 0, 0, 1.0);
    mset(etas, 0, 1, -1.0);

    // Ideal point 2 is explained by the last two topics.
    mset(etas, 1, 2, 1.0);
    mset(etas, 1, 3, -1.0);
  }

  // Finally, find the likelihood of this matrix.  Note that this
  // is found primarily to track the elbo.
  double lhood = 0.0;
  double x = 0.0;
  for (int i=0; i < etas->size1; ++i) {
    for (int j=0; j < etas->size2; ++j) {
      x = mget(etas, i, j);
      lhood -= x * x / (2.0 * eta_variance);
    }
  }

  gsl_matrix_free(e_x_xt);
  gsl_matrix_free(e_x_xt_tmp);
  gsl_vector_free(e_x_y);

  return lhood;
}


void Posterior::UpdateSigmaL(const Corpus& corpus) {
  double f_k_sum = 0.0;

  for (int v=0; v < corpus.number_votes(); ++v) {
    const Vote& vote = *corpus.votes(v);
    int user = vote.user();
    int doc = vote.doc();

    double inner_product;
    gsl_vector_view user_ip = gsl_matrix_row(user_ips, user);
    gsl_vector_view doc_ip = gsl_matrix_row(doc_ips, doc);
    gsl_blas_ddot(&user_ip.vector, &doc_ip.vector, &inner_product);

    double term_exp = exp(inner_product);
    // For convenience:
    double term_f = term_exp + 1;

    double tmp = ((term_exp / term_f)
		  - term_exp * term_exp
		  / (term_f * term_f));
    if (fabs(inner_product) > 50) {
      tmp = 0.0;
    }
    gsl_blas_ddot(&doc_ip.vector, &doc_ip.vector,
		  &inner_product);
    double grad_lambda = tmp * inner_product;

    f_k_sum += grad_lambda;
  }
  int number_priored_users = 0;

  if (FLAGS_experimental_priors) {
    for (int u=0; u < corpus.number_users(); ++u) {
      if (corpus.users(u)->id() == "300041"
	  || corpus.users(u)->id() == "400440"
	  || corpus.users(u)->id() == "300059"
	  || corpus.users(u)->id() == "400425") {
	number_priored_users += 1;
      }
    }
  }

  //   printf("Number priored users: %d\n", number_priored_users);

  const int kNumberUsers = corpus.number_users();

  // Recall: we have hard priors on 4 individuals.
  double tmp = (f_k_sum * FLAGS_vote_replication
		+ ((kNumberUsers) * ip_dimension - number_priored_users)
		/ (sigma_u * sigma_u)
		+ number_priored_users
		/ (kPriorVariance));
  sigma_lambda = sqrt(kNumberUsers * ip_dimension
		      // * log(2 * PI)
		      / (double) tmp);
}


void Posterior::UpdateSigmaU(const Corpus& corpus) {
  LogOnce("updating_sigma_u",
	  "Warning: Updating sigma u might lead "
	  "to bad solutions.");

  const int kNumberUsers = corpus.number_users();

  gsl_matrix* m = gsl_matrix_alloc(kNumberUsers, ip_dimension);
  gsl_matrix_memcpy(m, user_ips);
  gsl_matrix_mul_elements(m, user_ips);
  double lambda_squared;
  for (int u=0; u < kNumberUsers; ++u) {
    for (int d=0; d < ip_dimension; ++d) {
      lambda_squared += mget(m, u, d);
    }
  }

  // Recall: we have hard priors on 4 individuals.
  sigma_u = (lambda_squared + sigma_lambda) / log(2.0 * PI);

  gsl_matrix_free(m);
}


void Posterior::UpdateSigmaK(const Corpus& corpus) {
  double f_l_sum = 0.0;
  for (int v=0; v < corpus.number_votes(); ++v) {
    const Vote& vote = *(corpus.votes(v));
    int user = vote.user();
    int doc = vote.doc();

    double inner_product;
    gsl_vector_view user_ip = gsl_matrix_row(user_ips, user);
    gsl_vector_view doc_ip = gsl_matrix_row(doc_ips, doc);
    gsl_blas_ddot(&user_ip.vector, &doc_ip.vector, &inner_product);
    
    double term_exp = exp(inner_product);
    // For convenience:
    double term_f = term_exp + 1;

    double tmp = ((term_exp / term_f)
		  - term_exp * term_exp
		  / (term_f * term_f));
    if (fabs(inner_product) > 50) {
      tmp = 0.0;
    }

    gsl_blas_ddot(&user_ip.vector, &user_ip.vector, &inner_product);
    double grad_lambda = tmp * inner_product;

    f_l_sum += grad_lambda;
  }
  const int kNumberDocs = corpus.number_docs();

  double tmp = (f_l_sum * FLAGS_vote_replication
		+ kNumberDocs * ip_dimension
		/ dispersion);
  sigma_kappa = sqrt(kNumberDocs * ip_dimension / (double) tmp);
}


double Posterior::UserIdealLikelihood(const Corpus& corpus,
				      double sigma_u) {
  double user_likelihood = 0.0;
  double h = 0.0;
  gsl_matrix* tmp = gsl_matrix_alloc(user_ips->size1,
				     user_ips->size2);
  gsl_matrix_memcpy(tmp, user_ips);
  gsl_matrix_mul_elements(tmp, tmp);
 
  gsl_vector* ones = gsl_vector_alloc(user_ips->size2);
  gsl_vector_set_all(ones, 1.0);
  gsl_vector* row_sums = gsl_vector_alloc(user_ips->size1);
  
  gsl_blas_dgemv(CblasNoTrans,
		 1.0,
		 tmp,
		 ones,
		 0.0,
		 row_sums);

  // Be sure to add (\sigma_\lambda^2) to these values.
  // Note that we use FLAGS_sigma_u instead of any posterior
  // sense of sigma_u.
  user_likelihood = -(((user_ips->size1 * user_ips->size2
			* sigma_lambda * sigma_lambda)
		       + sum(row_sums)) / (2 * sigma_u * sigma_u));
  user_likelihood -= ((user_ips->size1 * user_ips->size2)
		      * log(2 * PI
			    * sigma_u * sigma_u)
		      / 2.0);

  // Include the entropy term for sigma_lambda.
  user_likelihood += ((user_ips->size1 * user_ips->size2)
		      * (1.0 + log(2 * PI
				   * sigma_lambda * sigma_lambda))
		      / 2.0);

  
  if (FLAGS_experimental_priors) {
    for (int user=0; user < corpus.number_users(); ++user) {
      const char* user_str = corpus.users(user)->id().c_str();
      if (!strcmp(user_str, "300041")
	  || !strcmp(user_str, "400440")) {
	// This is Enzi or Donald Young.  Set the prior to -4.0.
	user_likelihood += ((2.0 * (-4.0) * mget(user_ips, user, 1) - 4.0 * 4.0)
			    / (2 * kPriorVariance)
			    + ((mget(user_ips, user, 1) * mget(user_ips, user, 1)
				+ sigma_lambda * sigma_lambda))
			    * (1.0 / (2 * sigma_u * sigma_u)
			       - 1.0 / (2 * kPriorVariance)));
      } else if (!strcmp(user_str, "300059")
		 || !strcmp(user_str, "400425")) {
	// This is Kennedy (300025) or Henry Waxman (400425).
	// Set the prior to 4.0.
	user_likelihood += ((2.0 * 4.0 * mget(user_ips, user, 1) - 4.0 * 4.0)
			    / (2 * kPriorVariance)
			    + ((mget(user_ips, user, 1) * mget(user_ips, user, 1)
				+ sigma_lambda * sigma_lambda))
			    * (1.0 / (2 * sigma_u * sigma_u)
			       - 1.0 / (2 * kPriorVariance)));
      }
    }
  }


  gsl_matrix_free(tmp);
  gsl_vector_free(row_sums);
  gsl_vector_free(ones);
  return user_likelihood;
}


double Posterior::DocIdealLikelihood(const Corpus& corpus,
				     double dispersion) {
  double doc_likelihood = 0.0;
  double h = 0.0;
  gsl_vector* tmp = gsl_vector_alloc(number_topics);

  // In the interest of readability, we compute these
  // values using many loops (instead of taking full advantage
  // of gsl's matrix algebra).
  for (int i=0; i < ip_dimension; ++i) {
    double ip_likelihood = 0.0;
    for (int d=0; d < corpus.number_docs(); ++d) {
      gsl_vector_view doc_zbar = gsl_matrix_row(doc_zbars, d);
      double doc_ip = mget(doc_ips, d, i);
      double eta_zbar;
      gsl_vector_view eta_i = gsl_matrix_row(etas, i);
      gsl_blas_ddot(&doc_zbar.vector, &eta_i.vector, &eta_zbar);

      // Compute eta' E(zbar zbar') eta.
      gsl_blas_dgemv(CblasNoTrans,
		     1.0,
		     doc_zbar_zbar_t[d],
		     &eta_i.vector,
		     0.0,
		     tmp);
      double eta_zbar_zbar_eta;
      gsl_blas_ddot(tmp, &eta_i.vector, &eta_zbar_zbar_eta);

      // Note that we use FLAGS_dispersion instead of any posterior
      // sense of dispersion.
      double term1 = - ((eta_zbar_zbar_eta
			 - 2.0 * doc_ip * eta_zbar
			 + doc_ip * doc_ip
			 + sigma_kappa * sigma_kappa)
			/ (2.0 * dispersion));
      double term2 = - log(2 * PI * dispersion) / 2.0;
      double term3 = ((1.0 + log(2 * PI * sigma_kappa * sigma_kappa))
		      / 2.0);
      ip_likelihood += term1 + term2 + term3;
    }
    doc_likelihood += ip_likelihood;
  }

  gsl_vector_free(tmp);
  return doc_likelihood;
}


double Posterior::VoteLikelihood(const Corpus& corpus) {
  double vote_likelihood = 0.0;
  for (int v=0; v < corpus.number_votes(); ++v) {
    const Vote& vote = *corpus.votes(v);
    int user = vote.user();
    int doc = vote.doc();

    double inner_product;
    gsl_vector_view user_ip = gsl_matrix_row(user_ips, user);
    gsl_vector_view doc_ip = gsl_matrix_row(doc_ips, doc);
    gsl_blas_ddot(&user_ip.vector, &doc_ip.vector, &inner_product);
    
    double term_exp = exp(inner_product);
    // For convenience:
    double term_f = term_exp + 1;

    double term1 = 0.0;
    if (vote.positive()) {
      term1 = inner_product;
    } else {
      term1 = 0.0;
    }

    double term2 = -log(term_f);
    if (inner_product > 50.0) {
      term2 = -inner_product;
    } else if (inner_product < -50.0) {
      term2 = 0.0;
    }

    double tmp = ((term_exp / term_f)
		  - term_exp * term_exp
		  / (term_f * term_f));
    if (fabs(inner_product) > 50) {
      tmp = 0.0;
    }
    gsl_blas_ddot(&doc_ip.vector, &doc_ip.vector, &inner_product);
    double grad_lambda = tmp * inner_product;

    gsl_blas_ddot(&user_ip.vector, &user_ip.vector, &inner_product);
    double grad_kappa = tmp * inner_product;

    double term3 = - (sigma_kappa * sigma_kappa
		      * grad_kappa
		      + sigma_lambda * sigma_lambda
		      * grad_lambda) / 2.0;
    
    vote_likelihood += term1 + term2 + term3;

  }
  return vote_likelihood * FLAGS_vote_replication;
}


void Posterior::WriteVotePredictions(const Corpus& corpus) const {
  double vote_accuracy = VoteClassificationAccuracy(corpus);
  printf("In-sample vote accuracy: %lf\n", vote_accuracy);
  //  printf("Number votes: %d\n", corpus.number_votes());

  std::ofstream f;
  f.open(StringPrintf("%s_predictions.csv", FLAGS_checkpoint_prefix.c_str()).c_str(),
	 ios::out);
  f << "UserId,DocId,Vote,Prediction,ExpectedPrediction" << endl;
  for (int v=0; v < corpus.number_votes(); ++v) {
    const Vote& vote = *corpus.votes(v);
    int user = vote.user();
    int doc = vote.doc();

    double inner_product;
    gsl_vector_view user_ip = gsl_matrix_row(user_ips, user);
    gsl_vector_view doc_ip = gsl_matrix_row(doc_ips, doc);
    gsl_blas_ddot(&user_ip.vector, &doc_ip.vector, &inner_product);

    double variance_inner_product = 0.0;
    for (int i=0; i < user_ip.vector.size; ++i) {
      double u_ip = vget(&user_ip.vector, i);
      double d_ip = vget(&doc_ip.vector, i);
      variance_inner_product += (
        sigma_kappa * sigma_kappa * u_ip * u_ip
	+ sigma_lambda * sigma_lambda * d_ip * d_ip
	+ sigma_kappa * sigma_kappa * sigma_lambda * sigma_lambda);
    }
    double term_exp = exp(inner_product
			  + variance_inner_product / 2.0);
    // Make two predictions: a simple one and one that gives the correct
    // expectation under the posterior.
    double prediction_simple = (term_exp / (1.0 + term_exp));
    double prediction_complex = (prediction_simple
				 + variance_inner_product
				 * (prediction_simple
				    - 3.0 * (prediction_simple
					     * prediction_simple)
				    + 2.0 * (prediction_simple
					     * prediction_simple
					     * prediction_simple))
				 / 2.0);

    string line = StringPrintf("%s,%s,%s,%.6lf,%.6lf",
			       corpus.users(user)->id().c_str(),
			       corpus.docs(doc)->id().c_str(),
			       vote.positive() ? "+" : "-",
			       prediction_simple,
			       prediction_complex);
    if (missing_user_ids_.find(corpus.users(user)->id())
	!= missing_user_ids_.end()) {
      // No point writing out this prediction, since we haven't seen
      // this user before.
      prediction_simple = *(double*) nan;
      prediction_complex = *(double*) nan;
      line = StringPrintf("%s,%s,%s,NA,NA",
			  corpus.users(user)->id().c_str(),
			  corpus.docs(doc)->id().c_str(),
			  vote.positive() ? "+" : "-");
    }

    f << line << endl;
  }
  f.close();
}


double Posterior::VoteClassificationAccuracy(const Corpus& corpus) const {
  double correct = 0.0;
  double number_votes = 0.0;
  for (int v=0; v < corpus.number_votes(); ++v) {
    const Vote& vote = *corpus.votes(v);
    int user = vote.user();
    int doc = vote.doc();

    double inner_product;
    gsl_vector_view user_ip = gsl_matrix_row(user_ips, user);
    gsl_vector_view doc_ip = gsl_matrix_row(doc_ips, doc);
    gsl_blas_ddot(&user_ip.vector, &doc_ip.vector, &inner_product);
    
    number_votes += 1.0;
    if (vote.positive() && inner_product > 0) {
      correct += 1.0;
    } else if (!vote.positive() && inner_product < 0) {
      correct += 1.0;
    }
  }  
  return correct / number_votes;
}


void Posterior::RandomInitializeIdealPoint(const Corpus& corpus) {
  // Change all of the first user's ideal values to 1.  This will
  // converge to a better value, but we need to initialize the model
  // to something unstable.
  UniformRNG();
  for (int u=0; u < user_ips->size1; ++u) {
    gsl_vector_view user_ip = gsl_matrix_row(user_ips, u);
    for (int i=0; i < user_ip.vector.size; ++i) {
      if (FLAGS_encode_difficulty && i == 0) {
	vset(&user_ip.vector, i, 1.0);
      } else {
	vset(&user_ip.vector, i, (UniformRNG() - 0.5) / (3.0));
      }
    }
  }
}


void Posterior::ExpandIdealDimension(const Corpus& corpus) {
  // Quick hack:
  Posterior* post = this;
  post->expanded_ip_dimension += 1;
  post->ip_dimension += 1;

  gsl_matrix* user_ips_old = user_ips;
  gsl_matrix* doc_ips_old = doc_ips;
  gsl_matrix* etas_old = etas;

  post->user_ips = gsl_matrix_calloc(corpus.number_users(),
				     post->ip_dimension);
  post->doc_ips = gsl_matrix_calloc(corpus.number_docs(),
				    post->ip_dimension);
  post->etas = gsl_matrix_calloc(post->ip_dimension, number_topics);
  
  for (int i=0; i < doc_ips_old->size2; ++i) {
    for (int d=0; d < doc_ips_old->size1; ++d) {
      mset(post->doc_ips, d, i, mget(doc_ips_old, d, i));
    }
    for (int u=0; u < user_ips_old->size1; ++u) {
      mset(post->user_ips, u, i, mget(user_ips_old, u, i));
    }
    for (int k=0; k < etas_old->size1; ++k) {
      mset(post->etas, i, k, mget(etas_old, i, k));
    }
  }

  gsl_matrix_free(user_ips_old);
  gsl_matrix_free(doc_ips_old);
  gsl_matrix_free(etas_old);

  for (int d=0; d < corpus.number_docs(); ++d) {
    delete doc_update_stats[d];
    doc_update_stats[d] = new GradientStats(post->ip_dimension);
  }
  
  for (int u=0; u < corpus.number_users(); ++u) {
    user_update_stats[u] = new GradientStats(post->ip_dimension);
  }
}


void Posterior::CollapseIdealDimension(const Corpus& corpus) {
  printf("Collapsing IP dimension.\n");
  Posterior* post = this;
  assert(post->expanded_ip_dimension > 0);
  post->expanded_ip_dimension -= 1;
  post->ip_dimension -= 1;

  gsl_matrix* user_ips_old = user_ips;
  gsl_matrix* doc_ips_old = doc_ips;
  gsl_matrix* etas_old = etas;

  post->user_ips = gsl_matrix_calloc(corpus.number_users(),
				     post->ip_dimension);
  post->doc_ips = gsl_matrix_calloc(corpus.number_docs(),
				    post->ip_dimension);
  post->etas = gsl_matrix_calloc(post->ip_dimension, number_topics);
  
  for (int i=0; i < post->ip_dimension; ++i) {
    for (int d=0; d < doc_ips_old->size1; ++d) {
      mset(post->doc_ips, d, i, mget(doc_ips_old, d, i));
    }
    for (int u=0; u < user_ips_old->size1; ++u) {
      mset(post->user_ips, u, i, mget(user_ips_old, u, i));
    }
    for (int k=0; k < etas_old->size2; ++k) {
      mset(post->etas, i, k, mget(etas_old, i, k));
    }
  }
  
  gsl_matrix_free(user_ips_old);
  gsl_matrix_free(doc_ips_old);
  gsl_matrix_free(etas_old);
  
  for (int d=0; d < corpus.number_docs(); ++d) {
    delete doc_update_stats[d];
    doc_update_stats[d] = new GradientStats(post->ip_dimension);
  }

  for (int u=0; u < corpus.number_users(); ++u) {
    user_update_stats[u] = new GradientStats(post->ip_dimension);
  }
}


void Posterior::Infer(const Corpus& corpus) {
  bool doc_converged = false;
  bool vote_converged = false;

  double last_doc_lhood = 10;
  double last_vote_lhood = 10;

  bool second_order = FLAGS_second_order;

  double doc_lhood = 0.0;
  double eta_likelihood = 0.0;

  double ideal_lhood = -1e8;
  double old_ideal_lhood = -1.1e8;
  double loop_convergence_criterion = 1e-2;

  double ideal_lhood_moving_average = 0.0;

  // Iterate if we have fewer than 5 iterations done.  Iterate until
  // all parts of the model are converged.  Stop iterating if the
  // entire model has not converged according to the moving average.
  last_iteration = false;
  int total_ideal_iterations = 0;
  while (!last_iteration) {

    if (FLAGS_mode != "ideal"
	&& iteration > 10) {
      // printf("Updating Eta.\n");
      eta_likelihood = UpdateEta(corpus);
    } else {
      // printf("Skipping eta update.\n");
    }


    last_iteration = (
	(doc_converged
	 && vote_converged
	 && (Converged(sigma_u, FLAGS_sigma_u, 1e-4)
	     || FLAGS_sigma_u <= 0)
	 && (Converged(dispersion, FLAGS_dispersion, 1e-4)
	     || FLAGS_dispersion <= 0))
	|| (Converged(ideal_lhood_moving_average, ideal_lhood, 1e-4)));
    if (iteration <= 11) {
      last_iteration = false;
    }
    if (expanded_ip_dimension > 0) {
      last_iteration = false;
    }

    if (last_iteration && (FLAGS_mode != "fit_predict"
			   && FLAGS_mode != "fixed_topics_predict")) {
      // It's the last iteration, with no need for a final eta
      // estimate.
      continue;
    }

    ideal_lhood_moving_average = ideal_lhood_moving_average * 0.99 +
      ideal_lhood * 0.01;

    double vote_lhood = 0.0;
    // Update ideal points.
    Log("Fitting ideal points.");

    //    printf("Iteration %d.\n", iteration);
    // printf("Updating ideal points.\n");
    double vote_convergence_criterion = 1e-4;

    // Perform a sort of variational annealing:
    // let these parameters start out close to 1 and converge
    // to their true values.  Note that this may make the model
    // move away from the ELBO at times.    
    if (FLAGS_variational_anneal) {
      if (iteration > 2) {
	sigma_u = pow(sigma_u, 0.95) * pow(FLAGS_sigma_u, 0.05);
	dispersion = pow(dispersion, 0.96) * pow(FLAGS_dispersion, 0.04);
      }
    } else {
      if (FLAGS_sigma_u > 0) {
	sigma_u = pow(sigma_u, 0.0) * pow(FLAGS_sigma_u, 1.0);
      }
      if (FLAGS_dispersion > 0) {
	dispersion = pow(dispersion, 0.0) * pow(FLAGS_dispersion, 1.0);
      }
    }

    if (Converged(ideal_lhood,
		  old_ideal_lhood,
		  vote_convergence_criterion / 2.0)) {
      old_ideal_lhood = ideal_lhood * (1.0 - vote_convergence_criterion);
    }

    int ideal_iteration = 0;
    double tmp_lcc;
    while (!Converged(ideal_lhood, old_ideal_lhood,
		      //vote_convergence_criterion / 2.0,
		      2e-6, // 0.000002
		      &tmp_lcc)
	   && (ideal_iteration < 101)) {
      loop_convergence_criterion = min(loop_convergence_criterion, tmp_lcc);
      loop_convergence_criterion = max(loop_convergence_criterion, 1e-7);
      // Gradually adjust sigma_u and dispersion to be FLAGS_sigma_u
      // and FLAGS_dispersion.  Otherwise convergence may take
      // forever.
      ++ideal_iteration;
      ++total_ideal_iterations;
      if (second_order && FLAGS_batch_update) {

	for (int i=0; i < 1; ++i) {
	  if (last_iteration) {
	    FitIdealPoints(corpus,
			   total_ideal_iterations,
			   second_order, false, false, true);
	  } else {
	    FitIdealPoints(corpus,
			   total_ideal_iterations,
			   second_order, false, false, false);
	  }
	  for (int d=0; d < corpus.number_docs(); ++d) {
	    if (d % kDocMod == total_ideal_iterations % kDocMod) {
	      doc_update_stats[d]->Flush();
	      continue;
	    }
	    gsl_vector_view doc_ip = gsl_matrix_row(doc_ips, d);

	    if (doc_update_stats[d]->update_count) {
	      //	      if (d ==0) {printf("Updating docs.\n");}
	      doc_update_stats[d]->PushHessian(&doc_ip.vector);
	    }
	  }
	  // FitIdealPoints(corpus, second_order, false, true, false);
	  for (int u=0;
	       u < corpus.number_users() && !last_iteration;
	       ++u) {
	    if (u % kUserMod == total_ideal_iterations % kUserMod) {
	      user_update_stats[u]->Flush();
	      continue;
	    }
	    gsl_vector_view user_ip = gsl_matrix_row(user_ips, u);
	    if (user_update_stats[u]->update_count) {
	      user_update_stats[u]->PushHessian(&user_ip.vector);
	    }
	    if (FLAGS_encode_difficulty) {
	      if (fabs(vget(&user_ip.vector, 0) - 1.0) > 1e-5) {
		LogFatal("Error.  User Ip is different from 1.\n");
	      }
	      vset(&user_ip.vector, 0, 1.0);
	    }
	  }
	}
      } else {
	FitIdealPoints(corpus,
		       total_ideal_iterations,
		       second_order, false,
		       /*only_users=*/false, /*only_docs=*/false);
      }
      if (isnan(ideal_lhood)) {
	LogFatal("Error.  Likelihood is nan.  Failing.");
      }
      old_ideal_lhood = ideal_lhood;
      ideal_lhood = (VoteLikelihood(corpus)
		     + UserIdealLikelihood(corpus,
					   sigma_u)
		     + DocIdealLikelihood(corpus,
					  dispersion)
		     + eta_likelihood);
      // PrintLikelihood("", corpus, this, doc_lhood, eta_likelihood);
      
      if ((FLAGS_mode == "fit"
	   || FLAGS_mode == "fit_predict"
	   || FLAGS_mode == "resume"
	   || FLAGS_mode == "fixed_topics"
	   || FLAGS_mode == "fixed_topics_predict"
	   || FLAGS_mode == "ideal")
	  && ideal_iteration % 2 == 0) {
	if (true || iteration < 20) {
	  UpdateSigmaL(corpus);
	}
	
	if (FLAGS_sigma_u <= 0) {
	  UpdateSigmaU(corpus);
	}

	if (true || iteration < 20) {
	  UpdateSigmaK(corpus);
	}
      }
    }

    Log("Running doc (E) step.");
    if (FLAGS_mode == "fit"
	|| FLAGS_mode == "fit_predict"
	|| FLAGS_mode == "resume"
	|| FLAGS_mode == "fixed_topics"
	|| FLAGS_mode == "fixed_topics_predict"
	|| FLAGS_mode == "ideal") {
      if (FLAGS_mode != "ideal"
	  && iteration > 10
	  && iteration < 1000) {
	// Generally we should converge within 1k iterations, but just in case...
	doc_lhood = FitLDA(corpus, !last_iteration,
			   2e-6);
      }

      UpdateSigmaL(corpus);
      
      if (FLAGS_sigma_u <= 0) {
	UpdateSigmaU(corpus);
      }

      UpdateSigmaK(corpus);

      // Note that we update eta in each LDA iteration.
    } else {
      printf("Unhandled mode: %s.\n", FLAGS_mode.c_str());
      exit(1);
    }

    if (FLAGS_mode != "ideal"
	&& iteration > 10) {
      // printf("Updating Eta.\n");
      eta_likelihood = UpdateEta(corpus);
    } else {
      // printf("Skipping eta update.\n");
    }

    vote_lhood += UserIdealLikelihood(corpus,
				      FLAGS_sigma_u);
    vote_lhood += DocIdealLikelihood(corpus,
				     FLAGS_dispersion);
    vote_lhood += VoteLikelihood(corpus);
    vote_lhood += eta_likelihood;

    double doc_convergence_criterion;

    // Note: These convergence values were 4e-5 for data in 020 and
    // 6e-5 for data in 021.
    // converged to 0.00004.
    doc_converged = Converged(last_doc_lhood, doc_lhood,
			      6e-5);
    // converged to 0.00004.
    vote_converged = Converged(
      last_vote_lhood, vote_lhood,
      6e-5, &vote_convergence_criterion);

    last_doc_lhood = doc_lhood;
    likelihood = doc_lhood + vote_lhood;
    last_vote_lhood = vote_lhood;

    Log(StringPrintf("Evidence lower bound: %lf. \n"
		     "  Vote: %lf; doc: %lf; \n"
		     "  Doc ideal: %lf; \n"
		     "  User ideal: %lf; \n"
		     "  Eta: %lf",
		     doc_lhood + vote_lhood,
		     vote_lhood,
		     doc_lhood,
		     DocIdealLikelihood(corpus,
					FLAGS_dispersion),
		     UserIdealLikelihood(corpus,
					 FLAGS_sigma_u),
		     eta_likelihood));
    ++iteration;

    if (iteration >= 2 && expanded_ip_dimension > 0) {
      CollapseIdealDimension(corpus);
    }

    if (!isnan(doc_lhood + vote_lhood)) {
      if (iteration % 5 == 0) {
	WriteCheckpoint(corpus);
      }
    } else {
      Log("Likelihood is nan. Failing.");
      exit(1);
    }
  }
  WriteCheckpoint(corpus);

}


void Posterior::InferDocIPsFromTopics(const Corpus& corpus) {
  gsl_blas_dgemm(CblasNoTrans,
		 CblasTrans,
		 1.0,
		 doc_zbars,
		 etas,
		 0.0,
		 doc_ips);
}


void Posterior::PredictDocs(const Corpus& corpus) {
  bool doc_converged = false;
  bool vote_converged = false;

  double last_doc_lhood;
  double last_vote_lhood;

  // Fit the docs.
  Log("Running doc (E) step.");
  double doc_lhood;
  doc_lhood = FitLDA(corpus, false, 2e-6);

  InferDocIPsFromTopics(corpus);

  if (!isnan(doc_lhood)) {
    WriteCheckpoint(corpus);
  } else {
    Log("Likelihood is nan. Failing.");
    exit(1);
  }
}

}  // namespace legis
