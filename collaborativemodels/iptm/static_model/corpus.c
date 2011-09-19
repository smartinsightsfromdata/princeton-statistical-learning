/*
  Copyright Princeton University, 2010. All rights reserved.
  Author: Sean M. Gerrish (sgerrish@cs.princeton.edu)

  This file implements a Corpus class for the Ideal Point Topic Model.
*/

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include <vector>
#include <ext/hash_map>
#include <ext/hash_set>
#include <string>
#include <fstream>

#include <gflags/gflags.h>

#include "corpus.h"
#include "strings.h"
#include "logging.h"

DECLARE_string(votes);
DECLARE_string(users);
DECLARE_string(docs);
DECLARE_string(working_directory);

// Experimental flags
DECLARE_string(time_filter);
DECLARE_string(mode);

DEFINE_bool(prune_terms,
	    false,
	    "If true, restrict docs' term counts to 1000.");
DEFINE_double(min_votes_per_user,
	      20,
	      "The minimum number of votes required per "
	      "user.  Users with fewer votes will be "
	      "dropped.");
DEFINE_double(min_votes_per_doc,
	      20,
	      "The minimum number of votes required per "
	      "doc.  Docs with fewer votes will be "
	      "dropped.");

using std::string;
using std::ifstream;
using std::ios;
using __gnu_cxx::hash_set;
using namespace slap_utils;

namespace legis {

User::User(const string& user_string)
  : id_(),
    data_(),
    number_votes_(0) {
  vector<string> user_parts;
  SplitStringUsing(user_string, ",", &user_parts);
  static int number_parts = -1;
  LogOnce("user_fields",
	  StringPrintf("Reading %d fields from users file.\n",
		       user_parts.size()));
  if (number_parts == -1) {
    number_parts = user_parts.size();
  } else if (number_parts != user_parts.size()) {
    printf("Error.  Rows in users file with differing "
	   "numbers of comma-delimited fields (%d vs. %d).\n"
	   "Current string: %s",
	   number_parts, user_parts.size(), user_string.c_str());
    exit(1);
  }
  id_ = user_parts[0];
  if (number_parts == 1) {
    data_ = "";
  } else {
    data_ = user_parts[1];
  }
}


Document::Document()
  : terms_counts_(),
    terms_(NULL),
    counts_(NULL),
    unique_terms_(0),
    number_votes_(0) {
}


Document::Document(const string& doc_string)
  : terms_counts_(),
    terms_(NULL),
    counts_(NULL),
    valid_(false),
    unique_terms_(0),
    number_votes_(0) {
  ParseDoc(doc_string);
}

void Document::ParseDoc(const string& doc_string) {
  terms_counts_.clear();
  vector<string> counts_str;
  SplitStringUsing(doc_string, " ", &counts_str);

  if (counts_str.empty()) {
    return;
  }

  id_ = counts_str[0];
  term_counts_total_ = 0;

  // If this doc does not pass our time filter, skip it.
  if (!FLAGS_time_filter.empty()
      && !(id_.find(FLAGS_time_filter) == 0)) {
    return;
  }

  int i = 0;
  vector<int> terms;
  vector<int> counts;
  terms_ = new int[counts_str.size() - 1];
  counts_ = new int[counts_str.size() - 1];
  for (vector<string>::const_iterator it=counts_str.begin();
       it != counts_str.end();
       ++it) {
    if (i == 0) {
      ++i;
      continue;
    }
    const string& kTermCounts = *it;
    size_t pos = kTermCounts.find(":");
    int index = ParseLeadingIntValue(kTermCounts.substr(0, pos));
    int count = ParseLeadingIntValue(kTermCounts.substr(pos + 1));
    terms_counts_[index] = count;
    term_counts_total_ += count;
    terms_[i - 1] = index;
    counts_[i - 1] = count;
    ++unique_terms_;
    ++i;
    terms.push_back(index);
    counts.push_back(count);
  }

  if (FLAGS_prune_terms) {
    double remove_factor = 1000.0 / term_counts_total_;
    remove_factor = remove_factor < 1.0 ? remove_factor : 1.0;
    remove_factor = remove_factor > 0.01 ? remove_factor : 0.01;
    if (remove_factor < 1.0) {
      int j = 0;
      for (int i = 0; i < terms.size(); ++i) {
	const int kCounts = counts[i];
	for (int n=0; n < kCounts; ++n) {
	  double cmp = (j * 17 * 13 * 19 * 23) % 100;
	  printf("%lf %lf\n", cmp,
		 100.0 * remove_factor);
	  if (cmp > 100.0 * remove_factor) {
	    counts[i] = counts[i] - 1;
	    term_counts_total_ -= 1;
	  }
	  ++j;
	}
	if (counts[i] == 0) {
	  --unique_terms_;
	}
      }
    }
    delete[] terms_;
    delete[] counts_;
    terms_ = new int[unique_terms_];
    counts_ = new int[unique_terms_];
    int n = 0;
    for (int i=0; i < terms.size(); ++i) {
      if (counts[i]) {
	terms_[n] = terms[i];
	counts_[n] = counts[i];
	++n;
      }
    }
    if(n != unique_terms_) {
      LogFatal("Error.  Unique terms != n.\n");
    }
  }

  valid_ = true;
}


Corpus::~Corpus() {
  for (int i=0; i < number_docs_; ++i) {
    delete docs_[i];
  }
  delete[] docs_;

  for (int i=0; i < number_users_; ++i) {
    delete users_[i];
  }
  delete[] users_;

  for (int i=0; i < number_votes_; ++i) {
    delete votes_[i];
  }
  delete[] votes_;
}


Vote::Vote(const string& vote_string,
	   hash_map<string, int>& docid_doc,
	   hash_map<string, int>& userid_user)
  : positive_(0),
    user_(-1),
    doc_(-1),
    valid_(false) {
  ParseVote(vote_string, docid_doc, userid_user);
}

void Vote::ParseVote(const string& vote_string,
		     hash_map<string, int>& docid_doc,
		     hash_map<string, int>& userid_user) {
  vector<string> vote_parts;
  SplitStringUsing(vote_string, ",", &vote_parts);
  if (vote_parts.size() != 3) {
    LogOnce("a", StringPrintf(
        "Warning. Skipping votes such as \n%s\n",
	vote_string.c_str()));
    return;
  }
  if (vote_parts[0] != "-"
      && vote_parts[0] != "+") {
    LogOnce("b", "Error parsing vote.  Expected \"+\" or \"-\".\n");
    return;
  }
  positive_ = (vote_parts[0] == "+");

  string id = vote_parts[1];
  // If this doc does not pass our time filter, skip it.
  if (!FLAGS_time_filter.empty()
      && id.find(FLAGS_time_filter) == string::npos) {
    return;
  }
  hash_map<string, int>::const_iterator it = docid_doc.find(id);
  if (it != docid_doc.end()) {
    doc_ = it->second;
  } else {
    // printf("Error. Doc Id not found: %s.\n", id.c_str());
    return;
  }

  id = vote_parts[2];
  it = userid_user.find(id);
  if (it != userid_user.end()) {
    user_ = it->second;
  } else {
    LogOnce("User missing.",
	    StringPrintf(
	      "Warning. User Id (%s) not found in users file.\n",
	      id.c_str()));
    return;
  }

  valid_ = true;
}

void Corpus::ReadCorpus() {
  // Read users' metadata.
  // For now, ters_his is just ids.
  vector<User*> users;    // X array of players.
  vector<Document*> docs;    // X array of players.
  vector<Vote*> votes;

  const string kUsersFilename = FLAGS_users;
  ifstream f;
  string line;
  f.open(kUsersFilename.c_str(), ios::in);
  if (!f.is_open()) {
    printf("Error opening file %s.  Failing.\n",
	   kUsersFilename.c_str());
    exit(1);
  }
  while (!f.eof()) {
    getline(f, line);
    if (!line.size()) {
      break;
    }
    User* u = new User(line);
    userid_user_[u->id()] = users.size();
    users.push_back(u);
  }
  f.close();

  // Next, read all documents.
  const string kDocsFilename = FLAGS_docs;
  f.open(kDocsFilename.c_str(), ios::in);
  if (!f.is_open()) {
    printf("Error opening file %s.  Failing.\n",
	   kDocsFilename.c_str());
    exit(1);
  }
  Log("Reading documents.");
  while (!f.eof()) {
    getline(f, line);
    if (!line.size()) {
      break;
    }
    Document* doc = new Document(line);
    if (!doc->valid()) {
      delete doc;
      continue;
    }
    docid_doc_[doc->id()] = docs.size();
    docs.push_back(doc);
  }
  f.close();

  // Finally, read the votes.
  const string kVotesFilename = FLAGS_votes;
  f.open(kVotesFilename.c_str(), ios::in);
  if (!f.is_open()) {
    printf("Error opening file %s.  Failing.\n",
	   kVotesFilename.c_str());
    exit(1);
  }

  int number_discarded_votes = 0;
  Log("Reading votes.");
  while (!f.eof()) {
    getline(f, line);
    if (!line.size()) {
      break;
    }
    Vote* vote = new Vote(line, docid_doc_, userid_user_);
    if (!vote->valid()) {
      LogOnce("invalid_vote",
	      StringPrintf("Invalid votes were "
			   "discarded (e.g.: %s).",
			   line.c_str()));
      delete vote;
      number_discarded_votes += 1;
      continue;
    }
    docs[vote->doc()]->add_vote(votes.size());
    users[vote->user()]->add_vote(votes.size());
    votes.push_back(vote);
  }
  f.close();
  Log(StringPrintf("Number discarded votes: %d.",
		   number_discarded_votes));

  Log("Reconciling documents, votes, and users.");
  // Go through the docs and remove any with too few votes.
  // These are not interesting.
  int d = 0;
  int doc_count = 0;
  hash_set<int> dropped_docs;
  for (vector<Document*>::iterator it = docs.begin();
       it != docs.end(); ) {
    vector<int> votes_tmp = (*it)->votes();
    if (votes_tmp.size() <= FLAGS_min_votes_per_doc
	&& FLAGS_mode != "infer_docs"
	&& FLAGS_mode != "infer_votes") {
      LogOnce("user_fields",
	      StringPrintf("Dropping doc %d for too few votes: %d.\n",
			   d,
			   votes_tmp.size()));
      dropped_docs.insert(doc_count);
      docid_doc_.erase((*it)->id());
      for (int v=0; v < votes_tmp.size(); ++v) {
	votes[votes_tmp[v]]->set_user(-1);
	votes[votes_tmp[v]]->set_doc(-1);
      }
      delete *it;
      it = docs.erase(it);
    } else {
      for (int v=0; v < votes_tmp.size(); ++v) {
	votes[votes_tmp[v]]->set_doc(d);
      }
      docid_doc_[(*it)->id()] = d;
      ++it;
      ++d;
    }
    doc_count += 1;
  }

  // Go through the users and remove any with too few votes.
  // These are not interesting.
  int u = 0;
  int user_count = 0;
  hash_set<int> dropped_users;
  for (vector<User*>::iterator it = users.begin();
       it != users.end(); ) {
    vector<int> votes_tmp = (*it)->votes();
    if (votes_tmp.size() <= FLAGS_min_votes_per_user
	&& FLAGS_mode != "infer_docs"
	&& FLAGS_mode != "infer_votes") {
      LogOnce("user_fields",
	      StringPrintf("Dropping user %d for too few votes: %d.\n",
			   u,
			   votes_tmp.size()));

      dropped_users.insert(user_count);
      for (int v=0; v < votes_tmp.size(); ++v) {
	votes[votes_tmp[v]]->set_user(-1);
	votes[votes_tmp[v]]->set_doc(-1);
      }
      userid_user_.erase((*it)->id());
      delete *it;
      it = users.erase(it);
    } else {
      for (int v=0; v < votes_tmp.size(); ++v) {
	votes[votes_tmp[v]]->set_user(u);
      }
      userid_user_[(*it)->id()] = u;
      ++it;
      ++u;
    }
    user_count += 1;
  }

  for (int v=0; v < number_votes_; ++v) {
    Vote* vote = votes[v];
    if (dropped_users.find(vote->user())
	!= dropped_users.end()) {
      vote->set_user(-1);
      vote->set_doc(-1);
    }
    if (dropped_docs.find(vote->doc())
	!= dropped_docs.end()) {
      vote->set_user(-1);
      vote->set_doc(-1);
    }
    
  }

  // Finally, determine the number of terms.  We start counting at
  // term 0.
  number_terms_ = -1;
  for (int i=0; i < docs.size(); ++i) {
    const hash_map<int, int>& terms_counts =
      docs[i]->terms_counts();
    for (hash_map<int, int>::const_iterator it=
	   terms_counts.begin();
	 it != terms_counts.end();
	 ++it) {
      if (it->first + 1 > number_terms_) {
	number_terms_ = it->first + 1;
      }
    }
  }

  // Copy the docs and users vectors to our internal representation.
  number_users_ = users.size();
  number_docs_ = docs.size();
  number_votes_  = votes.size();
  users_ = new User*[number_users_];
  for (int i=0; i < number_users_; ++i) {
    users_[i] = users[i];
  }
  docs_ = new Document*[number_docs_];
  for (int i=0; i < number_docs_; ++i) {
    docs_[i] = docs[i];
  }

  // Finally, copy over votes.  Note that we may have deleted some
  // users and docs, so we only include the relevant votes.
  int v = 0;
  vector<int> old_vote_to_new_vote(number_votes_);
  votes_ = (legis::Vote**) malloc(sizeof(legis::Vote*)
				  * number_votes_);
  for (int i=0; i < number_votes_; ++i) {
    Vote* vote = votes[i];
    if (vote->doc() == -1 || vote->user() == -1) {
      continue;
    }
    votes_[v] = votes[i];
    old_vote_to_new_vote[i] = v;
    v += 1;
  }
  votes_ = (legis::Vote**) realloc(votes_,
				   sizeof(legis::Vote*) * v);
  number_votes_ = v;
  for (int u=0; u < users.size(); ++u) {
    User* user = users_[u];
    for (int v=0; v < user->number_votes(); ++v) {
      (*user->mutable_votes())[v] =
	old_vote_to_new_vote[(*user->mutable_votes())[v]];
    }
  }
  for (int d=0; d < docs.size(); ++d) {
    Document* doc = docs_[d];
    for (int v=0; v < doc->number_votes(); ++v) {
      (*doc->mutable_votes())[v] =
	old_vote_to_new_vote[(*doc->mutable_votes())[v]];
    }
  }

  
  printf("Read %d users, %d documents, and %d votes.\n",
	 users.size(),
	 docs.size(),
	 votes.size());

  if (!users.size()
      || !docs.size()
      || !votes.size()) {
    LogFatal("Not enough data.  Failing.");
  }
}


}  // namespace legis
