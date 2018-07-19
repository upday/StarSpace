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
#include <map>
#include <string>
#include <algorithm>
#include <iterator>
#include <boost/tokenizer.hpp>

using namespace std;
using namespace starspace;
//using namespace boost;

int main(int argc, char** argv) {
    shared_ptr<Args> args = make_shared<Args>();
    if (argc < 7) {
        cerr << "usage: " << argv[0] << " <model> k basedoc input_docs predictions_file [num_threads]\n";
        return 1;
    }
    std::string model(argv[1]);
    args->K = atoi(argv[2]);
    args->model = model;
    args->fileFormat = "labelDoc";
    args->basedoc = argv[3];
    string input_file(argv[4]);
    string output_file(argv[5]);

    int num_threads = 1;
    if (argv[6] != NULL)
        num_threads = atoi(argv[6]);

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
    sp.loadBaseDocsWithDocIds();

    typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
    boost::escaped_list_separator<char> Separator("", "\t", "\"\'");
    vector<string> vec;

    ofstream out(output_file.c_str());
    if (!out.is_open()) return 1;

    // Parallel predictions 
    vector<map<string, vector<string>>> predictions(num_threads);
    foreach_line(
        input_file,
        [&](std::string& line) {
            Tokenizer tok(line, Separator);
            vec.assign(tok.begin(), tok.end());
            vector<Base> query_vec;
            sp.parseDoc(vec.at(1), query_vec, " ");
            vector<Predictions> basedoc_indexes;
            sp.predictOneWithDocId(query_vec, basedoc_indexes);

            auto& pred = predictions[getThreadID()];
            for (int i = 0; i < basedoc_indexes.size(); i++) {
                pred[vec.at(0)].push_back(sp.idBaseDocs_[basedoc_indexes[i].second].first);
            }
        },
        num_threads
    );

    // Glue predictions together.
    map<string, vector<string>> results;
    for (int i = 0; i < predictions.size(); i++) {
        for (auto pred: predictions[i])
            results[pred.first] = pred.second;
    }

    // Write it down as TSV.
    for (auto res: results) {
        out << res.first << "\t";
        auto recos = res.second;
        for (auto reco = recos.begin(); reco != recos.end(); ++reco) {
            out << *reco;
            // print tab if not last column
            if (reco != recos.end())
                out << "\t";
            out << "\n";
        }
    }

    out.close();

    return 0;
}
