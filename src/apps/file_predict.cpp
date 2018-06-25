/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "../starspace.h"
#include <iostream>
#include <boost/algorithm/string/predicate.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <boost/tokenizer.hpp>

using namespace std;
using namespace starspace;
//using namespace boost;

int main(int argc, char** argv) {
  shared_ptr<Args> args = make_shared<Args>();
  if (argc < 3) {
    cerr << "usage: " << argv[0] << " <model> k [basedoc]\n";
    return 1;
  }
  std::string model(argv[1]);
  args->K = atoi(argv[2]);
  args->model = model;
  if (argc > 3) {
    args->fileFormat = "labelDoc";
    args->basedoc = argv[3];
  }

  string input_file;
  if (argc > 4) {
     input_file = argv[4];
  } else {
     input_file = "data.tsv";
  }

  StarSpace sp(args);
  if (boost::algorithm::ends_with(args->model, ".tsv")) {
    sp.initFromTsv(args->model);
  } else {
    sp.initFromSavedModel(args->model);
    cout << "------Loaded model args:\n";
    args->printArgs();
  }
  // Set dropout probability to 0 in test case.
  sp.args_->dropoutLHS = 0.0;
  sp.args_->dropoutRHS = 0.0;
  // Load basedocs which are set of possible things to predict.
  sp.loadBaseDocs();

  string data(input_file);
  ifstream in(data.c_str());
  if (!in.is_open()) return 1;

  typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
  boost::escaped_list_separator<char> Separator("", "\t", "\"\'");
  vector<string> vec;
  string line;

  while(getline(in, line)) {
    // Do the prediction
    Tokenizer tok(line, Separator);
    vec.assign(tok.begin(), tok.end());

    cout <<  vec.at(0);
    vector<Base> query_vec;
    sp.parseDoc(vec.at(1), query_vec, " ");
    vector<Predictions> predictions;
    sp.predictOne(query_vec, predictions);
    for (int i = 0; i < predictions.size() || i < 5; i++) {
      //cout << i << "[" << predictions[i].first << "]: ";
      // FIXME, cout cannot print a vector like this
      //cout << "\t" << sp.baseDocs_[predictions[i].second];
    }
    cout << "\n";
  }

  return 0;
}
