/*
  Copyright Princeton University, 2010. All rights reserved.
  Author: Sean M. Gerrish (sgerrish@cs.princeton.edu)

*/

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include <vector>
#include <ext/hash_map>
#include <gflags/gflags.h>
#include <string>

using std::string;
using std::vector;
using __gnu_cxx::hash_map;

namespace __gnu_cxx {
template<> struct hash< std::string >
{
  size_t operator()( const std::string& x ) const
  {
    return hash< const char* >()( x.c_str() );
  }
};
}

namespace legis {

class Document;
class Corpus;

class User {
 public:
  User(const string& s);

  const string& id() const {
    return id_;
  }
  void set_id(const string& id) {
    id_ = id;
  }
  void add_vote(int i) {
    votes_.push_back(i);
    ++number_votes_;
  }

  vector<int>* mutable_votes() {
    return &votes_;
  }

  const vector<int>& votes() {
    return votes_;
  }

  int number_votes() const {
    return number_votes_;
  }

 private:
  string id_;
  string data_;
  int number_votes_;
  vector<int> votes_;
};

class Document {
 public:
  Document();
  ~Document() {
    if (terms_) {
      delete[] terms_;
    }
    if (counts_) {
      delete[] counts_;
    }
  }

  Document(const string& doc_string);

  void ParseDoc(const string& doc_string);

  const hash_map<int, int>& terms_counts() const {
    return terms_counts_;
  }
 
  const string& id() const {
    return id_;
  }

  void add_vote(int i) {
    votes_.push_back(i);
    ++number_votes_;
  }

  int term_counts_total() const {
    return term_counts_total_;
  }

  const vector<int>& votes() {
    return votes_;
  }

  vector<int>* mutable_votes() {
    return &votes_;
  }

  bool valid() const {
    return valid_;
  }

  int terms(int i) const {
    return terms_[i];
  }
  int counts(int i) const {
    return counts_[i];
  }
  int unique_terms() const {
    return unique_terms_;
  }

  int number_votes() const {
    return number_votes_;
  }

 private:
  string id_;
  hash_map<int, int> terms_counts_;
  int term_counts_total_;
  vector<int> votes_;
  int* terms_;
  int* counts_;
  bool valid_;
  int unique_terms_;
  int number_votes_;
};


class Vote {
 public:
  Vote(const string& s,
       hash_map<string, int>& docid_doc,
       hash_map<string, int>& userid_doc);

  ~Vote() {}

  int positive() const {
    return positive_;
  }
  int user() const {
    return user_;
  }
  int doc() const {
    return doc_;
  }
  int set_user(int u) {
    user_ = u;
		return u;
  }
  int set_doc(int d) {
    doc_ = d;
		return d;
  }
  bool valid() const {
    return valid_;
  }

 private:
  void ParseVote(const string& s,
		 hash_map<string, int>& docid_doc,
		 hash_map<string, int>& userid_user);

  int positive_;
  int user_;
  int doc_;
  bool valid_;
};


class Corpus {
 public:
  ~Corpus();
  Corpus()
    : users_(NULL),
      docs_(NULL),
      votes_(NULL),
      number_users_(0),
      number_docs_(0),
      number_votes_(0),
      number_terms_(0) {}

  void ReadCorpus();
 
  int number_terms() const {
    return number_terms_;
  }

  int userid_user(const string& user_id) const {
    hash_map<string, int>::const_iterator it
      = userid_user_.find(user_id);
    if (it != userid_user_.end()) {
      return it->second;
    }
    return -1;
  }

  int docid_doc(const string& doc_id) const {
    hash_map<string, int>::const_iterator it = docid_doc_.find(doc_id);
    if (it != docid_doc_.end()) {
      return it->second;
    }
    return -1;
  }

  User* users(int i) const {
    return users_[i];
  }
  Document* docs(int i) const {
    return docs_[i];
  }
  Vote* votes(int i) const {
    return votes_[i];
  }

  int number_users() const {
    return number_users_;
  }
  int number_docs() const {
    return number_docs_;
  }
  int number_votes() const {
    return number_votes_;
  }

  // For testing only.
  void AddDocument(Document* doc) {
    docid_doc_[doc->id()] = 0;
    ++number_docs_;
    Document** new_docs = new Document*[number_docs_];
    if (docs_) {
      for (int d=0; d < number_docs_ - 1; ++d) {
	new_docs[d] = docs_[d];
      }
      delete[] docs_;
    }
    new_docs[number_docs_ - 1] = doc;
    docs_ = new_docs;

    for (int t=0; t < doc->unique_terms(); ++t) {
      if (doc->terms(t) + 1 > number_terms_) {
	number_terms_ = doc->terms(t) + 1;
      }
    }
  }

  void AddUser(User* user) {
    userid_user_[user->id()] = 0;
    ++number_users_;
    User** new_users = new User*[number_users_];
    if (users_) {
      for (int u=0; u < number_users_ - 1; ++u) {
	new_users[u] = users_[u];
      }
      delete[] users_;
    }
    new_users[number_users_ - 1] = user;
    users_ = new_users;
  }

  void AddVote(Vote* vote) {
    ++number_votes_;
    Vote** new_votes = new Vote*[number_votes_];
    if (votes_) {
      for (int v=0; v < number_votes_ - 1; ++v) {
	new_votes[v] = votes_[v];
      }
      delete[] votes_;
    }
    new_votes[number_votes_ - 1] = vote;
    votes_ = new_votes;
  }

 private:
  User** users_;
  Document** docs_;
  Vote** votes_;
  int number_users_;
  int number_docs_;
  int number_votes_;

  hash_map<string, int> userid_user_;
  hash_map<string, int> docid_doc_;
  int number_terms_;
};

}  // namespace legis
