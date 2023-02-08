#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <random>
#include <tuple>
#include <set>
#include "assert.h"
#include "json.cpp"

namespace py = pybind11;
using namespace std;

#define NULL_STRING "\0"

typedef string ATTR_TYPE;
typedef string VALUE_TYPE;
typedef int COUNT_TYPE;
typedef unordered_map<ATTR_TYPE, unordered_map<VALUE_TYPE, COUNT_TYPE>> AV_COUNT_TYPE;
typedef unordered_map<ATTR_TYPE, set<VALUE_TYPE>> AV_KEY_TYPE;
typedef pair<double, string> OPERATION_TYPE;

class MultinomialCobwebTree;
class MultinomialCobwebNode;

random_device rd;
mt19937_64 gen(rd());
uniform_real_distribution<double> unif(0, 1);

double custom_rand() {
    return unif(gen);
}

string repeat(string s, int n) {
    string res = "";
    for (int i = 0; i < n; i++) {
        res += s;
    }
    return res;
}


VALUE_TYPE most_likely_choice(vector<tuple<VALUE_TYPE, double>> choices) {
    vector<tuple<double, double, string>> vals;

    for (auto &[val, prob]: choices){
        if (prob < 0){
            cout << "most_likely_choice: all weights must be greater than or equal to 0" << endl;
        }
        vals.push_back(make_tuple(prob, custom_rand(), val));
    }
    sort(vals.rbegin(), vals.rend());

    return get<2>(vals[0]);
}

VALUE_TYPE weighted_choice(vector<tuple<VALUE_TYPE, double>> choices) {
    cout << "weighted_choice: Not implemented yet" << endl;
    return get<0>(choices[0]);
}


class MultinomialCobwebNode {

public:
    static int counter;
    int concept_id;
    COUNT_TYPE count;
    unordered_map<ATTR_TYPE, COUNT_TYPE> counts;
    unordered_map<ATTR_TYPE, COUNT_TYPE> squared_counts;
    vector<MultinomialCobwebNode *> children;
    MultinomialCobwebNode *parent;
    MultinomialCobwebTree *tree;
    AV_COUNT_TYPE av_counts;

    MultinomialCobwebNode();
    MultinomialCobwebNode(MultinomialCobwebNode *otherNode);

    static void update_counter(int concept_id) {
        if(concept_id > counter) {
            counter = concept_id + 1;
        }
    }

    void increment_counts(const AV_COUNT_TYPE &instance);
    void update_counts_from_node(MultinomialCobwebNode *node);
    double expected_correct_guesses_insert(const AV_COUNT_TYPE &instance);
    double expected_correct_guesses_merge(MultinomialCobwebNode *other, const AV_COUNT_TYPE &instance);
    MultinomialCobwebNode* get_best_level(const AV_COUNT_TYPE &instance);
    MultinomialCobwebNode* get_basic_level();
    double basic_cu();
    double expected_correct_guesses();
    double category_utility();
    tuple<double, string> get_best_operation(const AV_COUNT_TYPE &instance, MultinomialCobwebNode *best1,
                                             MultinomialCobwebNode *best2, double best1Cu,
                                             bool best_op=true, bool new_op=true,
                                             bool merge_op=true, bool split_op=true);
    tuple<double, MultinomialCobwebNode *, MultinomialCobwebNode *> two_best_children(const AV_COUNT_TYPE &instance);
    double log_prob_class_given_instance(const AV_COUNT_TYPE &instance);
    double cu_for_insert(MultinomialCobwebNode *child, const AV_COUNT_TYPE &instance);
    MultinomialCobwebNode *create_new_child(const AV_COUNT_TYPE &instance);
    double cu_for_new_child(const AV_COUNT_TYPE &instance);
    MultinomialCobwebNode *merge(MultinomialCobwebNode *best1, MultinomialCobwebNode *best2);
    double cu_for_merge(MultinomialCobwebNode *best1, MultinomialCobwebNode *best2, const AV_COUNT_TYPE &instance);
    void split(MultinomialCobwebNode *best);
    double cu_for_split(MultinomialCobwebNode *best);
    bool is_exact_match(const AV_COUNT_TYPE &instance);
    long _hash();
    int gensym();
    string __str__();
    string pretty_print(int depth = 0);
    int depth();
    bool is_parent(MultinomialCobwebNode *otherConcept);
    int num_concepts();
    string avcounts_to_json();
    string ser_avcounts();
    string counts_to_json();
    string squared_counts_to_json();
    string dump_json();
    string output_json();
    vector<tuple<VALUE_TYPE, double>> get_weighted_values(ATTR_TYPE attr, bool allowNone = true);
    VALUE_TYPE predict(ATTR_TYPE attr, string choiceFn = "most likely", bool allowNone = true);
    double probability(ATTR_TYPE attr, VALUE_TYPE val);
    double log_likelihood(MultinomialCobwebNode *childLeaf);

};


class MultinomialCobwebTree {

public:
    float alpha;
    MultinomialCobwebNode *root;
    // AV_KEY_TYPE av_key;

    MultinomialCobwebTree() {
        this->alpha = 1.0;
        this->root = new MultinomialCobwebNode();
        this->root->tree = this;
    }

    string __str__(){
        return this->root->__str__();
    }

    MultinomialCobwebNode* load_json_helper(json_object_s* object) {
        MultinomialCobwebNode *new_node = new MultinomialCobwebNode();
        new_node->tree = this;

        // Get concept_id
        struct json_object_element_s* concept_id_obj = object->start;
        int concept_id_val = atoi(json_value_as_number(concept_id_obj->value)->number);
        new_node->concept_id = concept_id_val;
        new_node->update_counter(concept_id_val);

        // Get count
        struct json_object_element_s* count_obj = concept_id_obj->next;
        int count_val = atoi(json_value_as_number(count_obj->value)->number);
        new_node->count = count_val;

        // Get counts
        struct json_object_element_s* counts_obj = count_obj->next;
        struct json_object_s* counts_dict = json_value_as_object(counts_obj->value);
        struct json_object_element_s* counts_cursor = counts_dict->start;
        while(counts_cursor != NULL) {
            // Get attr name
            string count_attr_name = string(counts_cursor->name->string);

            // A count is stored with each attribute
            int count_value = atoi(json_value_as_number(counts_cursor->value)->number);
            new_node->counts[count_attr_name] = count_value;

            counts_cursor = counts_cursor->next;
        }

        // Get squared counts
        struct json_object_element_s* squared_counts_obj = counts_obj->next;
        struct json_object_s* squared_counts_dict = json_value_as_object(squared_counts_obj->value);
        struct json_object_element_s* squared_counts_cursor = squared_counts_dict->start;
        while(squared_counts_cursor != NULL) {
            // Get attr name
            string squared_count_attr_name = string(squared_counts_cursor->name->string);

            // A count is stored with each attribute
            int squared_count_value = atoi(json_value_as_number(squared_counts_cursor->value)->number);
            new_node->squared_counts[squared_count_attr_name] = squared_count_value;

            squared_counts_cursor = squared_counts_cursor->next;
        }

        // Get counts
        struct json_object_element_s* av_counts_obj = squared_counts_obj->next;
        struct json_object_s* av_counts_dict = json_value_as_object(av_counts_obj->value);
        struct json_object_element_s* av_counts_cursor = av_counts_dict->start;
        while(av_counts_cursor != NULL) {
            // Get attr name
            string attr_name = string(av_counts_cursor->name->string);

            // The attr val is a dict of strings to ints
            struct json_object_s* attr_val_dict = json_value_as_object(av_counts_cursor->value);
            struct json_object_element_s* inner_counts_cursor = attr_val_dict->start;
            while(inner_counts_cursor != NULL) {
                // this will be a word
                string val_name = string(inner_counts_cursor->name->string);
                // This will always be a number
                int attr_val_count = atoi(json_value_as_number(inner_counts_cursor->value)->number);

                // Update the new node's counts
                new_node->av_counts[attr_name][val_name] = attr_val_count;

                inner_counts_cursor = inner_counts_cursor->next;
            }

            av_counts_cursor = av_counts_cursor->next;
        }

        // At this point in the coding, I am supremely annoyed at
        // myself for choosing this approach.

        // Get children
        struct json_object_element_s* children_obj = av_counts_obj->next;
        struct json_array_s* children_array = json_value_as_array(children_obj->value);
        struct json_array_element_s* child_cursor = children_array->start;
        vector<MultinomialCobwebNode*> new_children;
        while(child_cursor != NULL) {
            struct json_object_s* json_child = json_value_as_object(child_cursor->value);
            MultinomialCobwebNode *child = load_json_helper(json_child);
            child->parent = new_node;
            new_children.push_back(child);
            child_cursor = child_cursor->next;
        }
        new_node->children = new_children;

        return new_node;

        // It's important to me that you know that this code
        // worked on the first try.
    }

    string dump_json(){
        return this->root->dump_json();
    }

    void load_json(string json) {
        struct json_value_s* root = json_parse(json.c_str(), strlen(json.c_str()));
        struct json_object_s* object = (struct json_object_s*)root->payload;
        delete this->root;
        this->root = this->load_json_helper(object);
    }



    void clear() {
        delete this->root;
        this->root = new MultinomialCobwebNode();
        this->root->tree = this;
    }

    MultinomialCobwebNode *ifit(AV_COUNT_TYPE instance) {
        // for (auto &[attr, val_map]: instance) {
        //     if (attr[0] == '_') continue;

        //     for (auto &[val, cnt]: val_map) {
        //         this->av_key[attr].insert(val);
        //     }
        // }

        return this->cobweb(instance);
    }

    void fit(vector<AV_COUNT_TYPE> instances, int iterations = 1, bool randomizeFirst = true) {
        for (int i = 0; i < iterations; i++) {
            if (i == 0 && randomizeFirst) {
                shuffle(instances.begin(), instances.end(), default_random_engine());
            }
            for (auto &instance: instances) {
                this->ifit(instance);
            }
            shuffle(instances.begin(), instances.end(), default_random_engine());
        }
    }

    MultinomialCobwebNode *cobweb(const AV_COUNT_TYPE &instance) {
        MultinomialCobwebNode *current = this->root;
        while (current != NULL) {
            if (current->children.empty() && (current->is_exact_match(instance) || current->count == 0)) {
                current->increment_counts(instance);
                break;
            } else if (current->children.empty()) {
                MultinomialCobwebNode *newNode = new MultinomialCobwebNode(current);
                current->parent = newNode;
                newNode->children.push_back(current);

                if (newNode->parent != NULL) {
                    newNode->parent->children.erase(remove(newNode->parent->children.begin(),
                                                                     newNode->parent->children.end(),
                                                                     current),
                                                              newNode->parent->children.end());
                    newNode->parent->children.push_back(newNode);
                } else {
                    this->root = newNode;
                }
                newNode->increment_counts(instance);
                current = newNode->create_new_child(instance);
                break;

            } else {
                auto[best1_cu, best1, best2] = current->two_best_children(instance);
                auto[_, bestAction] = current->get_best_operation(instance, best1, best2, best1_cu);
                if (bestAction == "best") {
                    current->increment_counts(instance);
                    current = best1;
                } else if (bestAction == "new") {
                    current->increment_counts(instance);
                    current = current->create_new_child(instance);
                    break;
                } else if (bestAction == "merge") {
                    current->increment_counts(instance);
                    MultinomialCobwebNode *newChild = current->merge(best1, best2);
                    current = newChild;
                } else if (bestAction == "split") {
                    current->split(best1);
                } else {
                    throw "Best action choice \"" + bestAction +
                          "\" not a recognized option. This should be impossible...";
                }
            }
        }
        return current;
    }

    MultinomialCobwebNode *_cobweb_categorize(const AV_COUNT_TYPE &instance) {
        auto current = this->root;
        // double best_logp = current->log_prob_class_given_instance(instance);

        while (current != NULL) {
            if (current->children.empty()) {
                return current;
            }

            double best_logp = -999999;
            auto parent = current;
            current = NULL;
            for (auto &child: parent->children) {
                double logp = child->log_prob_class_given_instance(instance);
                if (current == NULL || logp > best_logp){
                    best_logp = logp;
                    current = child;
                }
            }
        }

        return current;
    }

    MultinomialCobwebNode *categorize(const AV_COUNT_TYPE &instance) {
        return this->_cobweb_categorize(instance);
    }
};


inline MultinomialCobwebNode::MultinomialCobwebNode() {
    concept_id = gensym();
    count = 0;
    counts = unordered_map<ATTR_TYPE, COUNT_TYPE>();
    squared_counts = unordered_map<ATTR_TYPE, COUNT_TYPE>();
    parent = NULL;
    tree = NULL;

}

inline MultinomialCobwebNode::MultinomialCobwebNode(MultinomialCobwebNode *otherNode) {
    concept_id = gensym();
    count = 0;
    counts = unordered_map<ATTR_TYPE, COUNT_TYPE>();
    squared_counts = unordered_map<ATTR_TYPE, COUNT_TYPE>();

    parent = otherNode->parent;
    tree = otherNode->tree;

    update_counts_from_node(otherNode);

    for (auto child: otherNode->children) {
        children.push_back(new MultinomialCobwebNode(child));
    }

}

inline void MultinomialCobwebNode::increment_counts(const AV_COUNT_TYPE &instance) {
    this->count += 1;

    for (auto &[attr, val_map]: instance) {
        bool hidden = attr[0] == '_';

        for (auto &[val, cnt]: val_map) {
            if (!hidden and this->av_counts.count(attr) and this->av_counts.at(attr).count(val)){
                // this->squared_counts[attr] -= pow(this->av_counts[attr][val] + this->tree->alpha, 2);
                this->squared_counts[attr] -= pow(this->av_counts[attr][val], 2);
            }

            this->counts[attr] += cnt;
            this->av_counts[attr][val] += cnt;

            if (!hidden){
                // this->squared_counts[attr] += pow(this->av_counts[attr][val] + this->tree->alpha, 2);
                this->squared_counts[attr] += pow(this->av_counts[attr][val], 2);
            }
        }
    }
}

inline void MultinomialCobwebNode::update_counts_from_node(MultinomialCobwebNode *node) {
    this->count += node->count;

    for (auto &[attr, tmp]: node->av_counts) {
        bool hidden = attr[0] == '_';

        this->counts[attr] += node->counts.at(attr);

        for (auto&[val, tmp2]: node->av_counts.at(attr)) {
            if (!hidden and this->av_counts.count(attr) and this->av_counts.at(attr).count(val)){
                // this->squared_counts[attr] -= pow(this->av_counts[attr][val] + this->tree->alpha, 2);
                this->squared_counts[attr] -= pow(this->av_counts[attr][val], 2);
            }

            this->av_counts[attr][val] += node->av_counts.at(attr).at(val);

            if (!hidden){
                // this->squared_counts[attr] += pow(this->av_counts[attr][val] + this->tree->alpha, 2);
                this->squared_counts[attr] += pow(this->av_counts[attr][val], 2);
            }
        }
    }
}

inline double MultinomialCobwebNode::expected_correct_guesses_insert(const AV_COUNT_TYPE &instance){
    unordered_map<ATTR_TYPE, COUNT_TYPE> squared_counts = unordered_map<ATTR_TYPE, COUNT_TYPE>();
    unordered_map<ATTR_TYPE, COUNT_TYPE> counts = unordered_map<ATTR_TYPE, COUNT_TYPE>();
    unordered_map<ATTR_TYPE, COUNT_TYPE> val_counts = unordered_map<ATTR_TYPE, COUNT_TYPE>();

    for (auto &[attr, cnt]: this->squared_counts) {
        squared_counts[attr] = cnt;
        counts[attr] = this->counts[attr];
        val_counts[attr] = this->av_counts.at(attr).size();
    } 

    for (auto &[attr, val_map]: instance) {
        if (attr[0] == '_'){
            continue;    
        }

        for (auto&[val, cnt]: val_map) {
            COUNT_TYPE av_count = 0;
            if (this->av_counts.count(attr) and this->av_counts.at(attr).count(val)){
                av_count = this->av_counts.at(attr).at(val);
                // squared_counts[attr] -= pow(av_count + this->tree->alpha, 2);
                squared_counts[attr] -= pow(av_count, 2);
            }
            else{
                val_counts[attr] += 1;
            }

            // squared_counts[attr] += pow(av_count + cnt + this->tree->alpha, 2);
            squared_counts[attr] += pow(av_count + cnt, 2);
            counts[attr] += cnt;
        }
    }

    double expected_correct_guesses = 1.0;
    for (auto &[attr, cnt]: squared_counts) {
        if (attr[0] != '_'){
            double multiplier = 1.0;
            // double multiplier = this->counts[attr] / this->count;
            expected_correct_guesses *= (multiplier * (squared_counts[attr])) / pow(counts[attr], 2);
            
            // double num_vals = this->tree->av_key.at(attr).size();
            // double num_missing = num_vals - val_counts.at(attr);
            // double missing_counts = num_missing * pow(this->tree->alpha, 2);
            // expected_correct_guesses += (multiplier * (squared_counts[attr] + missing_counts)) /
            //     pow(counts[attr] + num_vals * this->tree->alpha, 2);
        }
    }

    return expected_correct_guesses;

}

inline double MultinomialCobwebNode::expected_correct_guesses_merge(MultinomialCobwebNode *other, const AV_COUNT_TYPE &instance) {

    MultinomialCobwebNode* big = this;
    MultinomialCobwebNode* small = other;

    if (count < other->count){
        small = this;
        big = other;
    }

    unordered_map<ATTR_TYPE, COUNT_TYPE> squared_counts = unordered_map<ATTR_TYPE, COUNT_TYPE>();
    unordered_map<ATTR_TYPE, COUNT_TYPE> counts = unordered_map<ATTR_TYPE, COUNT_TYPE>();

    for (auto &[attr, cnt]: this->squared_counts) {
        squared_counts[attr] = cnt;
        counts[attr] = this->counts[attr];
    } 

    for (auto &[attr, tmp]: small->av_counts) {
        if (attr[0] == '_'){
            continue;
        }

        for (auto&[val, tmp2]: small->av_counts.at(attr)) {
            int big_count = 0;
            if (big->av_counts.count(attr) && big->av_counts.at(attr).count(val)){
                big_count = big->av_counts.at(attr).at(val);
                // squared_counts[attr] -= pow(big_count + this->tree->alpha, 2);
                squared_counts[attr] -= pow(big_count, 2);
            }

            // squared_counts[attr] += pow(big_count + small->av_counts.at(attr).at(val) + this->tree->alpha, 2); 
            squared_counts[attr] += pow(big_count + small->av_counts.at(attr).at(val), 2); 
            counts[attr] += small->av_counts.at(attr).at(val);
        }
    }

    for (auto &[attr, val_map]: instance) {
        if (attr[0] == '_'){
            continue;    
        }

        for (auto&[val, cnt]: val_map) {
            int big_count = 0;
            if (big->av_counts.count(attr) && big->av_counts.at(attr).count(val)){
                big_count = big->av_counts.at(attr).at(val);
            }

            int small_count = 0;
            if (small->av_counts.count(attr) && small->av_counts.at(attr).count(val)){
                small_count = small->av_counts.at(attr).at(val);
            }

            if ((big_count + small_count) > 0){
                // squared_counts[attr] -= pow(big_count + small_count + this->tree->alpha, 2);
                squared_counts[attr] -= pow(big_count + small_count, 2);
            }

            // squared_counts[attr] += pow(big_count + small_count + cnt + this->tree->alpha, 2);
            squared_counts[attr] += pow(big_count + small_count + cnt, 2);
            counts[attr] += cnt;
        }
    }

    double expected_correct_guesses = 1.0;
    for (auto &[attr, cnt]: squared_counts) {
        if (attr[0] != '_'){
            double multiplier = 1.0;
            // double multiplier = this->counts[attr] / this->count;
            expected_correct_guesses *= (multiplier * squared_counts[attr]) / pow(counts[attr], 2);

            // double num_vals = this->tree->av_key.at(attr).size();
            // double num_missing = num_vals - this->av_counts.at(attr).size();
            // double missing_counts = num_missing * pow(this->tree->alpha, 2);
            // expected_correct_guesses += (multiplier * (squared_counts[attr] + missing_counts)) /
            //     pow(counts[attr] + num_vals * this->tree->alpha, 2);
        }
    }

    return expected_correct_guesses;

}

inline MultinomialCobwebNode* MultinomialCobwebNode::get_best_level(const AV_COUNT_TYPE &instance){
    MultinomialCobwebNode* curr = this;
    MultinomialCobwebNode* best = this;
    double best_cu = this->log_prob_class_given_instance(instance);

    while (curr->parent != NULL) {
        curr = curr->parent;
        double curr_cu = curr->log_prob_class_given_instance(instance);

        if (curr_cu > best_cu) {
            best = curr;
            best_cu = curr_cu;
        }
    }

    return best;
}

inline MultinomialCobwebNode* MultinomialCobwebNode::get_basic_level(){
    MultinomialCobwebNode* curr = this;
    MultinomialCobwebNode* best = this;
    double best_cu = this->basic_cu();

    while (curr->parent != NULL) {
        curr = curr->parent;
        double curr_cu = curr->basic_cu();

        if (curr_cu > best_cu) {
            best = curr;
            best_cu = curr_cu;
        }
    }

    return best;
}

inline double MultinomialCobwebNode::expected_correct_guesses() {
    double expected_correct_guesses = 1.0;
    for (auto &[attr, cnt]: this->squared_counts) {
        if (attr[0] != '_'){
            double multiplier = 1.0;
            // double multiplier = this->counts[attr] / this->count;
            expected_correct_guesses *= (multiplier * squared_counts[attr]) / pow(counts[attr], 2);

            // double num_vals = this->tree->av_key.at(attr).size();
            // double num_missing = num_vals - this->av_counts.at(attr).size();
            // double missing_counts = num_missing * pow(this->tree->alpha, 2);
            // expected_correct_guesses += (multiplier * (squared_counts[attr] + missing_counts)) /
            //     pow(counts[attr] + num_vals * this->tree->alpha, 2);
        }
    }

    return expected_correct_guesses;
}

inline double MultinomialCobwebNode::category_utility() {
    if (children.empty()) {
        return 0.0;
    }
    double childCorrectGuesses = 0.0;
    for (auto &child: children) {
        double pOfChild = (1.0 * child->count) / this->count;
        childCorrectGuesses += pOfChild * child->expected_correct_guesses();
    }
    return ((childCorrectGuesses - this->expected_correct_guesses()) / children.size());
}

inline tuple<double, string> MultinomialCobwebNode::get_best_operation(
        const AV_COUNT_TYPE &instance, MultinomialCobwebNode *best1,
        MultinomialCobwebNode *best2, double best1Cu,
        bool best_op, bool new_op,
        bool merge_op, bool split_op) {

    if (best1 == NULL) {
        throw "Need at least one best child.";
    }
    vector<tuple<double, double, string>> operations;
    if (best_op){
        operations.push_back(make_tuple(best1Cu, custom_rand(), "best"));
    }
    if (new_op){
        operations.push_back(make_tuple(cu_for_new_child(instance),
                    custom_rand(), "new"));
    }
    if (merge_op && children.size() > 2 && best2 != NULL) {
        operations.push_back(make_tuple(cu_for_merge(best1, best2,
                        instance), custom_rand(), "merge"));
    }
    if (split_op && best1->children.size() > 0) {
        operations.push_back(make_tuple(cu_for_split(best1), custom_rand(), "split"));
    }
    sort(operations.rbegin(), operations.rend());

    OPERATION_TYPE bestOp = make_pair(get<0>(operations[0]), get<2>(operations[0]));
    return bestOp;
}

inline tuple<double, MultinomialCobwebNode *, MultinomialCobwebNode *> MultinomialCobwebNode::two_best_children(
        const AV_COUNT_TYPE &instance) {

    if (children.empty()) {
        throw "No children!";
    }
    vector<tuple<double, double, double, MultinomialCobwebNode *>> relative_cus;
    for (auto &child: children) {
        relative_cus.push_back(
            make_tuple(
                ((child->count + 1) * child->expected_correct_guesses_insert(instance)) - 
                (child->count * child->expected_correct_guesses()),
                child->count,
                custom_rand(),
                child));
    }
    sort(relative_cus.rbegin(), relative_cus.rend());

    MultinomialCobwebNode *best1 = get<3>(relative_cus[0]);
    double best1_cu = cu_for_insert(best1, instance);
    MultinomialCobwebNode *best2 = relative_cus.size() > 1 ? get<3>(relative_cus[1]) : NULL;
    return make_tuple(best1_cu, best1, best2);
}

inline double MultinomialCobwebNode::cu_for_insert(MultinomialCobwebNode *child, const AV_COUNT_TYPE &instance) {
    double child_correct_guesses = 0.0;

    for (auto &c: children) {
        if (c == child) {
            child_correct_guesses += ((c->count + 1) *
                    c->expected_correct_guesses_insert(instance));
        }
        else{
            child_correct_guesses += (c->count *
                    c->expected_correct_guesses());
        }
    }

    child_correct_guesses /= (count + 1);
    double parent_correct_guesses = this->expected_correct_guesses_insert(instance);
    return ((child_correct_guesses - parent_correct_guesses) / children.size());
}

inline MultinomialCobwebNode* MultinomialCobwebNode::create_new_child(const AV_COUNT_TYPE &instance) {
    MultinomialCobwebNode *newChild = new MultinomialCobwebNode();
    newChild->parent = this;
    newChild->tree = this->tree;
    newChild->increment_counts(instance);
    this->children.push_back(newChild);
    return newChild;
};

inline double MultinomialCobwebNode::cu_for_new_child(const AV_COUNT_TYPE &instance) {
    double child_correct_guesses = 0.0;

    for (auto &c: children) {
        child_correct_guesses += (c->count * c->expected_correct_guesses()); 
    }

    // Create node and compute EC over it (we can't just say 1 because of multinomial)
    MultinomialCobwebNode new_child = MultinomialCobwebNode();
    new_child.parent = this;
    new_child.tree = this->tree;
    new_child.increment_counts(instance);
    child_correct_guesses += new_child.expected_correct_guesses();

    child_correct_guesses /= (count + 1);
    double parent_correct_guesses = expected_correct_guesses_insert(instance);

    return ((child_correct_guesses - parent_correct_guesses) / (children.size()+1));
}

inline MultinomialCobwebNode* MultinomialCobwebNode::merge(MultinomialCobwebNode *best1, MultinomialCobwebNode *best2) {
    MultinomialCobwebNode *newChild = new MultinomialCobwebNode();
    newChild->parent = this;
    newChild->tree = this->tree;

    newChild->update_counts_from_node(best1);
    newChild->update_counts_from_node(best2);
    best1->parent = newChild;
    best2->parent = newChild;
    newChild->children.push_back(best1);
    newChild->children.push_back(best2);
    children.erase(remove(this->children.begin(), this->children.end(), best1), children.end());
    children.erase(remove(this->children.begin(), this->children.end(), best2), children.end());
    children.push_back(newChild);

    return newChild;
}

inline double MultinomialCobwebNode::cu_for_merge(MultinomialCobwebNode *best1,
        MultinomialCobwebNode *best2, const AV_COUNT_TYPE &instance) {

    double child_correct_guesses = 0.0;

    for (auto &c: children) {
        if (c == best1 || c == best2){
            continue;
        }

        child_correct_guesses += (c->count * c->expected_correct_guesses());
    }

    child_correct_guesses += ((best1->count + best2->count + 1) *
            best1->expected_correct_guesses_merge(best2, instance));

    child_correct_guesses /= count + 1;
    double parent_correct_guesses = expected_correct_guesses_insert(instance);

    return ((child_correct_guesses - parent_correct_guesses) / (children.size()-1));

}

inline void MultinomialCobwebNode::split(MultinomialCobwebNode *best) {
    children.erase(remove(children.begin(), children.end(), best), children.end());
    for (auto &c: best->children) {
        c->parent = this;
        c->tree = this->tree;
        children.push_back(c);
    }
    delete best;
}

inline double MultinomialCobwebNode::cu_for_split(MultinomialCobwebNode *best){
    double child_correct_guesses = 0.0;

    for (auto &c: children) {
        if (c == best) continue;
        child_correct_guesses += (c->count * c->expected_correct_guesses());
    }

    for (auto &c: best->children) {
        child_correct_guesses += (c->count * c->expected_correct_guesses());
    }

    child_correct_guesses /= count;
    double parent_correct_guesses = expected_correct_guesses();

    return ((child_correct_guesses - parent_correct_guesses) /
            (children.size() - 1 + best->children.size()));

}

inline bool MultinomialCobwebNode::is_exact_match(const AV_COUNT_TYPE &instance) {
    set<ATTR_TYPE> allAttrs;
    for (auto &[attr, tmp]: instance) allAttrs.insert(attr);
    for (auto &[attr, tmp]: this->av_counts) allAttrs.insert(attr);

    for (auto &attr: allAttrs) {
        if (attr[0] == '_') continue;
        if (instance.count(attr) && !this->av_counts.count(attr)) {
            return false;
        }
        if (this->av_counts.count(attr) && !instance.count(attr)) {
            return false;
        }
        if (this->av_counts.count(attr) && instance.count(attr)) {
            set<VALUE_TYPE> allVals;
            for (auto &[val, tmp]: instance.at(attr)) allVals.insert(val);
            for (auto &[val, tmp]: this->av_counts.at(attr)) allVals.insert(val);

            for (auto &val: allVals) {
                if (instance.at(attr).count(val) && !this->av_counts.at(attr).count(val)) {
                    return false;
                }
                if (this->av_counts.at(attr).count(val) && !instance.at(attr).count(val)) {
                    return false;
                }
                if (this->av_counts.at(attr).at(val) != instance.at(attr).at(val)) {
                    return false;
                }
            }
        }
    }
    return true;
}

inline long MultinomialCobwebNode::_hash() {
    hash<string> hash_obj;
    return hash_obj("MultinomialCobwebNode" + to_string(concept_id));
}

inline int MultinomialCobwebNode::gensym() {
    counter++;
    return counter;
}

inline string MultinomialCobwebNode::__str__(){
    return this->pretty_print();
}

inline string MultinomialCobwebNode::pretty_print(int depth) {
    string ret = repeat("\t", depth) + "|-" + avcounts_to_json() + "\n";

    for (auto &c: children) {
        ret += c->pretty_print(depth + 1);
    }

    return ret;
}


inline int MultinomialCobwebNode::depth() {
    if (this->parent) {
        return 1 + this->parent->depth();
    }
    return 0;
}

inline bool MultinomialCobwebNode::is_parent(MultinomialCobwebNode *otherConcept) {
    MultinomialCobwebNode *temp = otherConcept;
    while (temp != NULL) {
        if (temp == this) {
            return true;
        }
        try {
            temp = temp->parent;
        } catch (string e) {
            cout << temp;
            assert(false);
        }
    }
    return false;
}

inline int MultinomialCobwebNode::num_concepts() {
    int childrenCount = 0;
    for (auto &c: children) {
        childrenCount += c->num_concepts();
    }
    return 1 + childrenCount;
}

inline string MultinomialCobwebNode::avcounts_to_json() {
    string ret = "{";

    ret += "\"_expected_guesses\": {\n";
    ret += "\"#ContinuousValue#\": {\n";
    ret += "\"mean\": " + to_string(this->expected_correct_guesses()) + ",\n";
    ret += "\"std\": 1,\n";
    ret += "\"n\": 1,\n";
    ret += "}},\n";

    ret += "\"_basic_cu\": {\n";
    ret += "\"#ContinuousValue#\": {\n";
    ret += "\"mean\": " + to_string(this->basic_cu()) + ",\n";
    ret += "\"std\": 1,\n";
    ret += "\"n\": 1,\n";
    ret += "}},\n";

    int c = 0;
    for (auto &[attr, vAttr]: av_counts) {
        ret += "\"" + attr + "\": {";
        int inner_count = 0;
        for (auto &[val, cnt]: vAttr) {
            ret += "\"" + val + "\": " + to_string(cnt);
            if (inner_count != int(vAttr.size()) - 1){
                ret += ", ";
            }
            inner_count++;
        }
        ret += "}";

        if (c != int(av_counts.size())-1){
            ret += ", ";
        }
        c++;
    }
    ret += "}";
    return ret;
}

inline string MultinomialCobwebNode::ser_avcounts() {
    string ret = "{";

    int c = 0;
    for (auto &[attr, vAttr]: av_counts) {
        ret += "\"" + attr + "\": {";
        int inner_count = 0;
        for (auto &[val, cnt]: vAttr) {
            ret += "\"" + val + "\": " + to_string(cnt);
            if (inner_count != int(vAttr.size()) - 1){
                ret += ", ";
            }
            inner_count++;
        }
        ret += "}";

        if (c != int(av_counts.size())-1){
            ret += ", ";
        }
        c++;
    }
    ret += "}";
    return ret;
}

inline string MultinomialCobwebNode::counts_to_json() {
    string ret = "{";

    bool first =true;
    for (auto &[attr, cnt]: this->counts) {
        if (!first) ret += ",\n";
        else first = false;
        ret += "\"" + attr + "\": " + to_string(cnt);
    }

    ret += "}";
    return ret;
}

inline string MultinomialCobwebNode::squared_counts_to_json() {
    string ret = "{";

    bool first =true;
    for (auto &[attr, cnt]: this->squared_counts) {
        if (!first) ret += ",\n";
        else first = false;
        ret += "\"" + attr + "\": " + to_string(cnt);
    }

    ret += "}";
    return ret;
}



inline string MultinomialCobwebNode::dump_json() {
    string output = "{";

    output += "\"concept_id\": " + to_string(this->concept_id) + ",\n";
    output += "\"count\": " + to_string(this->count) + ",\n";
    output += "\"counts\": " + this->counts_to_json() + ",\n";
    output += "\"squared_counts\": " + this->squared_counts_to_json() + ",\n";
    output += "\"av_counts\": " + this->ser_avcounts() + ",\n";

    output += "\"children\": [\n";
    bool first = true;
    for (auto &c: children) {
        if(!first) output += ",";
        else first = false;
        output += c->dump_json();
    }
    output += "]\n";

    output += "}\n";

    return output;
}

inline string MultinomialCobwebNode::output_json(){
    string output = "{";

    output += "\"name\": \"Concept" + to_string(this->concept_id) + "\",\n";
    output += "\"size\": " + to_string(this->count) + ",\n";
    output += "\"children\": [\n";

    for (auto &c: children) {
        output += c->output_json() + ",";
    }

    output += "],\n";
    output += "\"counts\": " + this->avcounts_to_json() + "\n";

    output += "}\n";

    return output;
}

inline vector<tuple<VALUE_TYPE, double>> MultinomialCobwebNode::get_weighted_values(
        ATTR_TYPE attr, bool allowNone) {

    vector<tuple<VALUE_TYPE, double>> choices;
    if (!this->av_counts.count(attr)) {
        choices.push_back(make_tuple(NULL_STRING, 1.0));
    }
    double valCount = 0;
    for (auto &[val, tmp]: this->av_counts.at(attr)) {
        COUNT_TYPE count = this->av_counts.at(attr).at(val);
        choices.push_back(make_tuple(val, (1.0 * count) / this->count));
        valCount += count;
    }
    if (allowNone) {
        choices.push_back(make_tuple(NULL_STRING, ((1.0 * (this->count - valCount)) / this->count)));
    }
    return choices;
}

inline VALUE_TYPE MultinomialCobwebNode::predict(ATTR_TYPE attr, string choiceFn, bool allowNone) {
    function<ATTR_TYPE(vector<tuple<VALUE_TYPE, double>>)> choose;
    if (choiceFn == "most likely" || choiceFn == "m") {
        choose = most_likely_choice;
    } else if (choiceFn == "sampled" || choiceFn == "s") {
        choose = weighted_choice;
    } else throw "Unknown choice_fn";
    if (!this->av_counts.count(attr)) {
        return NULL_STRING;
    }
    vector<tuple<VALUE_TYPE, double>> choices = this->get_weighted_values(attr, allowNone);
    return choose(choices);
}

inline double MultinomialCobwebNode::probability(ATTR_TYPE attr, VALUE_TYPE val) {
    if (val == NULL_STRING) {
        double c = 0.0;
        if (this->av_counts.count(attr)) {
            for (auto &[attr, vAttr]: this->av_counts) {
                for (auto&[val, cnt]: vAttr) {
                    c += cnt;
                }
            }
            return (1.0 * (this->count - c)) / this->count;
        }
    }
    if (this->av_counts.count(attr) && this->av_counts.at(attr).count(val)) {
        return (1.0 * this->av_counts.at(attr).at(val)) / this->count;
    }
    return 0.0;
}

inline double MultinomialCobwebNode::log_likelihood(MultinomialCobwebNode *childLeaf) {
    set<ATTR_TYPE> allAttrs;
    for (auto &[attr, tmp]: this->av_counts) allAttrs.insert(attr);
    for (auto &[attr, tmp]: childLeaf->av_counts) allAttrs.insert(attr);

    double ll = 0;

    for (auto &attr: allAttrs) {
        if (attr[0] == '_') continue;
        set<VALUE_TYPE> vals;
        vals.insert(NULL_STRING);
        if (this->av_counts.count(attr)) {
            for (auto &[val, tmp]: this->av_counts.at(attr)) vals.insert(val);
        }
        if (childLeaf->av_counts.count(attr)) {
            for (auto &[val, tmp]: childLeaf->av_counts.at(attr)) vals.insert(val);
        }
        for (auto &val: vals) {
            double op = childLeaf->probability(attr, val);
            if (op > 0) {
                double p = this->probability(attr, val) * op;
                if (p >= 0) {
                    ll += log(p);
                } else throw "Should always be greater than 0";
            }
        }
    }
    return ll;
}

inline double MultinomialCobwebNode::basic_cu(){
    double p_of_c = (1.0 * this->count) / this->tree->root->count;
    return (p_of_c * (this->expected_correct_guesses() -
                this->tree->root->expected_correct_guesses()));
}


inline double MultinomialCobwebNode::log_prob_class_given_instance(const AV_COUNT_TYPE &instance){
    
    double log_prob = 0;

    for (auto &[attr, vAttr]: instance) {
        bool hidden = attr[0] == '_';
        if (hidden || !this->tree->root->av_counts.count(attr)){
            continue;
        }

        // double num_vals = this->tree->av_key.at(attr).size();
        double num_vals = this->tree->root->av_counts.at(attr).size();

        for (auto &[val, cnt]: vAttr){
            if (!this->tree->root->av_counts.at(attr).count(val)){
                continue;
            }
        
            COUNT_TYPE av_count = this->tree->alpha;
            if (this->av_counts.count(attr) && this->av_counts.at(attr).count(val)){
                av_count += this->av_counts.at(attr).at(val);
                // cout << val << "(" << this->av_counts.at(attr).at(val) << ") ";
            }

            log_prob += cnt * log((1.0 * av_count) / (this->counts[attr] + num_vals * this->tree->alpha));
        }

        // cout << endl;
        // cout << "denom: " << to_string(this->counts[attr] + num_vals * this->tree->alpha) << endl;
        // cout << "denom (no alpha): " << to_string(this->counts[attr]) << endl;
        // cout << "node count: " << to_string(this->count) << endl;
        // cout << "num vals: " << to_string(num_vals) << endl;
    }

    // cout << endl;
    
    log_prob += log((1.0 * this->count) / this->tree->root->count);

    return log_prob;
}

int MultinomialCobwebNode::counter = 0;


PYBIND11_MODULE(multinomial_cobweb, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::class_<MultinomialCobwebNode>(m, "MultinomialCobwebNode")
        .def(py::init())
        .def("pretty_print", &MultinomialCobwebNode::pretty_print)
        .def("output_json", &MultinomialCobwebNode::output_json)
        .def("predict", &MultinomialCobwebNode::predict, py::arg("attr") = "",
                py::arg("choiceFn") = "most likely",
                py::arg("allowNone") = true )
        .def("get_best_level", &MultinomialCobwebNode::get_best_level, py::return_value_policy::copy)
        .def("get_basic_level", &MultinomialCobwebNode::get_basic_level, py::return_value_policy::copy)
        .def("log_prob_class_given_instance", &MultinomialCobwebNode::log_prob_class_given_instance)
        .def("expected_correct_guesses", &MultinomialCobwebNode::expected_correct_guesses)
        .def("__str__", &MultinomialCobwebNode::__str__)
        .def_readonly("count", &MultinomialCobwebNode::count)
        .def_readonly("concept_id", &MultinomialCobwebNode::concept_id)
        .def_readonly("children", &MultinomialCobwebNode::children)
        .def_readonly("parent", &MultinomialCobwebNode::parent)
        .def_readonly("av_counts", &MultinomialCobwebNode::av_counts)
        .def_readonly("counts", &MultinomialCobwebNode::counts)
        .def_readonly("squared_counts", &MultinomialCobwebNode::squared_counts)
        .def_readonly("tree", &MultinomialCobwebNode::tree);

    py::class_<MultinomialCobwebTree>(m, "MultinomialCobwebTree")
        .def(py::init())
        .def("ifit", &MultinomialCobwebTree::ifit, py::return_value_policy::copy)
        .def("fit", &MultinomialCobwebTree::fit, py::arg("instances") = vector<AV_COUNT_TYPE>(), py::arg("iterations") = 1, py::arg("randomizeFirst") = true)
        .def("categorize", &MultinomialCobwebTree::categorize, py::return_value_policy::copy)
        .def("clear", &MultinomialCobwebTree::clear)
        .def("__str__", &MultinomialCobwebTree::__str__)
        .def("dump_json", &MultinomialCobwebTree::dump_json)
        .def("load_json", &MultinomialCobwebTree::load_json)
        .def_readonly("root", &MultinomialCobwebTree::root);
//         .def_readonly("meanSq", &ContinuousValue::meanSq);
}
