#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <mutex>
#include <functional>
#include <random>
#include <tuple>
#include <unordered_set>
#include "assert.h"
#include "json.cpp"
#include <atomic>

#include <execution>
// #define PAR std::execution::par_unseq,
// #define PAR std::execution::seq, 
#define PAR

// #if PARALLEL
// #include <execution>
// #define PAR std::execution::par,
// #else
// #define PAR
// #endif

namespace py = pybind11;
using namespace std;

#define NULL_STRING "\0"

typedef string ATTR_TYPE;
typedef string VALUE_TYPE;
typedef int COUNT_TYPE;
typedef unordered_map<VALUE_TYPE, COUNT_TYPE> VAL_COUNT_TYPE;
typedef unordered_map<ATTR_TYPE, VAL_COUNT_TYPE> AV_COUNT_TYPE;
typedef unordered_map<ATTR_TYPE, unordered_set<VALUE_TYPE>> AV_KEY_TYPE;
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
    unordered_map<ATTR_TYPE, COUNT_TYPE> attr_counts;
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
    double entropy_insert(const AV_COUNT_TYPE &instance);
    double entropy_merge(MultinomialCobwebNode *other, const AV_COUNT_TYPE &instance);
    MultinomialCobwebNode* get_best_level(const AV_COUNT_TYPE &instance);
    MultinomialCobwebNode* get_basic_level();
    double basic_mi();
    double entropy();
    double original_entropy();
    double mutual_information();
    tuple<double, string> get_best_operation(const AV_COUNT_TYPE &instance, MultinomialCobwebNode *best1,
                                             MultinomialCobwebNode *best2, double best1Cu,
                                             bool best_op=true, bool new_op=true,
                                             bool merge_op=true, bool split_op=true);
    tuple<double, MultinomialCobwebNode *, MultinomialCobwebNode *> two_best_children(const AV_COUNT_TYPE &instance);
    double log_prob_class_given_instance(const AV_COUNT_TYPE &instance);
    double mi_for_insert(MultinomialCobwebNode *child, const AV_COUNT_TYPE &instance);
    MultinomialCobwebNode *create_new_child(const AV_COUNT_TYPE &instance);
    double mi_for_new_child(const AV_COUNT_TYPE &instance);
    MultinomialCobwebNode *merge(MultinomialCobwebNode *best1, MultinomialCobwebNode *best2);
    double mi_for_merge(MultinomialCobwebNode *best1, MultinomialCobwebNode *best2, const AV_COUNT_TYPE &instance);
    void split(MultinomialCobwebNode *best);
    double mi_for_split(MultinomialCobwebNode *best);
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
    string attr_counts_to_json();
    string dump_json();
    string output_json();
    vector<tuple<VALUE_TYPE, double>> get_weighted_values(ATTR_TYPE attr, bool allowNone = true);
    VALUE_TYPE predict(ATTR_TYPE attr, string choiceFn = "most likely", bool allowNone = true);
    double probability(ATTR_TYPE attr, VALUE_TYPE val);
    double log_likelihood(MultinomialCobwebNode *childLeaf);

};


class MultinomialCobwebTree {

public:
    // float alpha;
    MultinomialCobwebNode *root;
    AV_KEY_TYPE attr_vals;
    float alpha_weight;

    MultinomialCobwebTree() {
        this->alpha_weight = 1.0;
        this->root = new MultinomialCobwebNode();
        this->root->tree = this;
        this->attr_vals = AV_KEY_TYPE();
    }

    string __str__(){
        return this->root->__str__();
    }

    float alpha(string attr){
        COUNT_TYPE n_vals = this->attr_vals.at(attr).size();
        if (n_vals == 0){
            return 1.0;
        } else {
            return this->alpha_weight / n_vals;
        }
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

        // Get attr_counts
        struct json_object_element_s* attr_counts_obj = count_obj->next;
        struct json_object_s* attr_counts_dict = json_value_as_object(attr_counts_obj->value);
        struct json_object_element_s* attr_counts_cursor = attr_counts_dict->start;
        while(attr_counts_cursor != NULL) {
            // Get attr name
            string attr_name = string(attr_counts_cursor->name->string);

            // A count is stored with each attribute
            int count_value = atoi(json_value_as_number(attr_counts_cursor->value)->number);
            new_node->attr_counts[attr_name] = count_value;

            attr_counts_cursor = attr_counts_cursor->next;
        }

        // Get av counts
        struct json_object_element_s* av_counts_obj = attr_counts_obj->next;
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
        this->attr_vals = AV_KEY_TYPE();
    }

    MultinomialCobwebNode *ifit(AV_COUNT_TYPE instance) {
        for (auto &[attr, val_map]: instance) {
            if (attr[0] == '_') continue;

            for (auto &[val, cnt]: val_map) {
                this->attr_vals[attr].insert(val);
            }
        }

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
                auto[best1_mi, best1, best2] = current->two_best_children(instance);
                auto[_, bestAction] = current->get_best_operation(instance, best1, best2, best1_mi);
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
        double best_logp = current->log_prob_class_given_instance(instance);

        while (true) {
            if (current->children.empty()) {
                return current;
            }

            auto parent = current;
            current = NULL;

            bool found = false;
            for (auto &child: parent->children) {
                double logp = child->log_prob_class_given_instance(instance);
                if (current == NULL || logp > best_logp){
                // if (logp > best_logp){
                    found = true;
                    best_logp = logp;
                    current = child;
                }
            }
            if (!found){
                return current;
            }
        }
    }

    MultinomialCobwebNode *categorize(const AV_COUNT_TYPE &instance) {
        return this->_cobweb_categorize(instance);
    }
};


inline MultinomialCobwebNode::MultinomialCobwebNode() {
    concept_id = gensym();
    count = 0;
    attr_counts = unordered_map<ATTR_TYPE, COUNT_TYPE>();
    parent = NULL;
    tree = NULL;
}

inline MultinomialCobwebNode::MultinomialCobwebNode(MultinomialCobwebNode *otherNode) {
    concept_id = gensym();
    count = 0;
    attr_counts = unordered_map<ATTR_TYPE, COUNT_TYPE>();

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
        for (auto &[val, cnt]: val_map) {
            this->attr_counts[attr] += cnt;
            this->av_counts[attr][val] += cnt;
        }
    }
}

inline void MultinomialCobwebNode::update_counts_from_node(MultinomialCobwebNode *node) {
    this->count += node->count;

    for (auto &[attr, val_map]: node->av_counts) {
        this->attr_counts[attr] += node->attr_counts.at(attr);

        for (auto&[val, cnt]: val_map) {
            this->av_counts[attr][val] += cnt;
        }
    }
}

inline double MultinomialCobwebNode::entropy_insert(const AV_COUNT_TYPE &instance){
    unordered_set<ATTR_TYPE> all_attrs;
    for (auto &[attr, tmp]: instance){
        if (attr[0] == '_') continue;
        all_attrs.insert(attr);
    }
    for (auto &[attr, tmp]: this->av_counts){
        if (attr[0] == '_') continue;
        all_attrs.insert(attr);
    }

    return transform_reduce(PAR this->av_counts.begin(), this->av_counts.end(), 0.0,
                plus<>(), [&](const auto& attr_it){
        COUNT_TYPE attr_count = 0;
        unordered_set<VALUE_TYPE> all_vals;
        float alpha = this->tree->alpha(attr_it.first);
        int num_vals = this->tree->attr_vals.at(attr_it.first).size();

        if (this->av_counts.count(attr_it.first)){
            attr_count += this->attr_counts.at(attr_it.first);
            for (auto &[val, cnt]: this->av_counts.at(attr_it.first)) all_vals.insert(val);
        }
        if (instance.count(attr_it.first)){
            for (auto &[val, cnt]: instance.at(attr_it.first)){
                attr_count += cnt;
                all_vals.insert(val);
            }
        }

        double info = transform_reduce(PAR all_vals.begin(), all_vals.end(), 0.0,
                plus<>(), [&](const auto &val){
            COUNT_TYPE av_count = 0;

            if (this->av_counts.count(attr_it.first) and this->av_counts.at(attr_it.first).count(val)){
                av_count += this->av_counts.at(attr_it.first).at(val);
            }
            if (instance.count(attr_it.first) and instance.at(attr_it.first).count(val)){
                av_count += instance.at(attr_it.first).at(val);
            }

            double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
            return -p * log(p);
        });

        COUNT_TYPE num_missing = num_vals - all_vals.size();
        if (num_missing > 0 and alpha > 0){
            double p = (alpha / (attr_count + num_vals * alpha));
            info -= num_missing * p * log(p);
        }

        return info;
    });

    // return info;
}

inline double MultinomialCobwebNode::entropy_merge(MultinomialCobwebNode *other, const AV_COUNT_TYPE &instance) {

    unordered_set<ATTR_TYPE> all_attrs;
    for (auto &[attr, tmp]: instance){
        if (attr[0] == '_') continue;
        all_attrs.insert(attr);
    }
    for (auto &[attr, tmp]: this->av_counts){
        if (attr[0] == '_') continue;
        all_attrs.insert(attr);
    }
    for (auto &[attr, tmp]: other->av_counts){
        if (attr[0] == '_') continue;
        all_attrs.insert(attr);
    }

    return transform_reduce(PAR this->av_counts.begin(), this->av_counts.end(), 0.0,
                plus<>(), [&](const auto& attr_it){
        COUNT_TYPE attr_count = 0;
        unordered_set<ATTR_TYPE> all_vals;
        float alpha = this->tree->alpha(attr_it.first);
        int num_vals = this->tree->attr_vals.at(attr_it.first).size();

        if (this->av_counts.count(attr_it.first)){
            attr_count += this->attr_counts.at(attr_it.first);
            for (auto &[val, cnt]: this->av_counts.at(attr_it.first)) all_vals.insert(val);
        }
        if (other->av_counts.count(attr_it.first)){
            attr_count += other->attr_counts.at(attr_it.first);
            for (auto &[val, cnt]: other->av_counts.at(attr_it.first)) all_vals.insert(val);
        }
        if (instance.count(attr_it.first)){
            for (auto &[val, cnt]: instance.at(attr_it.first)){
                attr_count += cnt;
                all_vals.insert(val);
            }
        }

        double info = transform_reduce(PAR all_vals.begin(), all_vals.end(), 0.0,
                plus<>(), [&](const auto &val){
            COUNT_TYPE av_count = 0;

            if (this->av_counts.count(attr_it.first) and this->av_counts.at(attr_it.first).count(val)){
                av_count += this->av_counts.at(attr_it.first).at(val);
            }
            if (other->av_counts.count(attr_it.first) and other->av_counts.at(attr_it.first).count(val)){
                av_count += other->av_counts.at(attr_it.first).at(val);
            }
            if (instance.count(attr_it.first) and instance.at(attr_it.first).count(val)){
                av_count += instance.at(attr_it.first).at(val);
            }

            double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
            return -p * log(p);

        });

        /*
        for (auto &val: all_vals) {
            COUNT_TYPE av_count = 0;

            if (this->av_counts.count(attr) and this->av_counts.at(attr).count(val)){
                av_count += this->av_counts.at(attr).at(val);
            }
            if (other->av_counts.count(attr) and other->av_counts.at(attr).count(val)){
                av_count += other->av_counts.at(attr).at(val);
            }
            if (instance.count(attr) and instance.at(attr).count(val)){
                av_count += instance.at(attr).at(val);
            }

            double p = ((av_count + this->tree->alpha(attr)) / (attr_counts +
                        this->tree->attr_vals.at(attr).size() *
                        this->tree->alpha(attr)));
            info -= p * log(p);
        }
        */

        COUNT_TYPE num_missing = num_vals - all_vals.size();
        if (num_missing > 0 and alpha > 0){
            double p = (alpha / (attr_count + num_vals * alpha));
            info += num_missing * -p * log(p);
        }

        return info;
    });

    // return info;

}

inline MultinomialCobwebNode* MultinomialCobwebNode::get_best_level(const AV_COUNT_TYPE &instance){
    MultinomialCobwebNode* curr = this;
    MultinomialCobwebNode* best = this;
    double best_ll = this->log_prob_class_given_instance(instance);

    while (curr->parent != NULL) {
        curr = curr->parent;
        double curr_ll = curr->log_prob_class_given_instance(instance);

        if (curr_ll > best_ll) {
            best = curr;
            best_ll = curr_ll;
        }
    }

    return best;
}

inline MultinomialCobwebNode* MultinomialCobwebNode::get_basic_level(){
    MultinomialCobwebNode* curr = this;
    MultinomialCobwebNode* best = this;
    double best_mi = this->basic_mi();

    while (curr->parent != NULL) {
        curr = curr->parent;
        double curr_mi = curr->basic_mi();

        if (curr_mi > best_mi) {
            best = curr;
            best_mi = curr_mi;
        }
    }

    return best;
}

inline double MultinomialCobwebNode::original_entropy() {
    double info = 0;

    for (auto &[attr, val_map]: this->av_counts) {
        if (attr[0] == '_') continue;

        COUNT_TYPE attr_count = this->attr_counts.at(attr);
        int num_vals = this->tree->attr_vals.at(attr).size();
        float alpha = this->tree->alpha(attr);
           
        for (auto &[val, cnt]: val_map) {
            double p = ((cnt + alpha) / (attr_count + num_vals * alpha));
            info -= p * log(p);
        }

        COUNT_TYPE num_missing = num_vals - val_map.size();
        if (num_missing > 0 and alpha > 0){
            double p = (alpha / (attr_count + num_vals * alpha));
            info -= num_missing * p * log(p);
        }
        
    }

    return info;
}

inline double MultinomialCobwebNode::entropy() {

    return transform_reduce(PAR this->av_counts.begin(), this->av_counts.end(), 0.0,
                plus<>(), [&](const auto& attr_it){
        if (attr_it.first[0] == '_') return 0.0;

        COUNT_TYPE attr_count = this->attr_counts.at(attr_it.first);
        int num_vals = this->tree->attr_vals.at(attr_it.first).size();
        float alpha = this->tree->alpha(attr_it.first);

        double info = transform_reduce(PAR attr_it.second.begin(), attr_it.second.end(), 0.0,
                plus<>(), [&](const auto& val_it){
            double p = ((val_it.second + alpha) / (attr_count + num_vals * alpha));
            return -p * log(p);
        });

        COUNT_TYPE num_missing = num_vals - attr_it.second.size();
        if (num_missing > 0 and alpha > 0){
            double p = (alpha / (attr_count + num_vals * alpha));
            info += num_missing * -p * log(p);
        }
        
        return info;
    });
}

inline double MultinomialCobwebNode::mutual_information() {
    if (children.empty()) {
        return 0.0;
    }

    double children_entropy = 0.0;
    double info_c = 0.0;

    for (auto &child: children) {
        double p_of_child = (1.0 * child->count) / this->count;
        info_c -= p_of_child * log(p_of_child);
        children_entropy += p_of_child * child->entropy();
    }

    return ((this->entropy() - children_entropy)
            // - info_c);
            / children.size());
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
        operations.push_back(make_tuple(best1Cu,
                    custom_rand(),
                    // 3,
                    "best"));
    }
    if (new_op){
        operations.push_back(make_tuple(mi_for_new_child(instance),
                    custom_rand(),
                    // 1,
                    "new"));
    }
    if (merge_op && children.size() > 2 && best2 != NULL) {
        operations.push_back(make_tuple(mi_for_merge(best1, best2,
                        instance),
                    custom_rand(),
                    // 2,
                    "merge"));
        // for (auto &child1: children) {
        //     for (auto &child2: children) {
        //         if (child1 == child2) continue;
        //         operations.push_back(make_tuple(mi_for_merge(child1, child2,
        //                         instance),
        //                     custom_rand(),
        //                     // 2,
        //                     "merge"));
        //     }
        // }
    }

    if (split_op && best1->children.size() > 0) {
        operations.push_back(make_tuple(mi_for_split(best1),
                    custom_rand(),
                    // 4,
                    "split"));
        // for (auto &child: children) {
        //     operations.push_back(make_tuple(mi_for_split(child),
        //                 custom_rand(),
        //                 // 4,
        //                 "split"));
        // }
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

    /*
    vector<tuple<double, double, double, MultinomialCobwebNode *>> relative_mis(this->children.size());
    transform(std::execution::par_unseq, this->children.begin(), this->children.end(), relative_mis.begin(), [&instance](auto const child){
        // return child->entropy();
        return make_tuple(
            (child->count * child->entropy()) -
            ((child->count + 1) * child->entropy_insert(instance)),
            child->count,
            0, // custom_rand(),
            child);
    });
    */

    vector<tuple<double, double, double, MultinomialCobwebNode *>> relative_mis;
    for (auto &child: this->children) {
        relative_mis.push_back(
            make_tuple(
                (child->count * child->entropy()) -
                ((child->count + 1) * child->entropy_insert(instance)),
                child->count,
                custom_rand(),
                child));
    }

    sort(relative_mis.rbegin(), relative_mis.rend());

    MultinomialCobwebNode *best1 = get<3>(relative_mis[0]);
    double best1_mi = mi_for_insert(best1, instance);
    MultinomialCobwebNode *best2 = relative_mis.size() > 1 ? get<3>(relative_mis[1]) : NULL;
    return make_tuple(best1_mi, best1, best2);
}

inline double MultinomialCobwebNode::mi_for_insert(MultinomialCobwebNode *child, const AV_COUNT_TYPE &instance) {
    double children_entropy = 0.0;
    double info_c = 0.0;

    for (auto &c: this->children) {
        if (c == child) {
            double p_of_child = (c->count + 1.0) / (this->count + 1.0);
            info_c -= p_of_child * log(p_of_child);
            children_entropy += p_of_child * c->entropy_insert(instance);
        }
        else{
            double p_of_child = (1.0 * c->count) / (this->count + 1.0);
            info_c -= p_of_child * log(p_of_child);
            children_entropy += p_of_child * c->entropy();
        }
    }

    return ((this->entropy_insert(instance) - children_entropy)
            // - info_c);
            / this->children.size());
}

inline MultinomialCobwebNode* MultinomialCobwebNode::create_new_child(const AV_COUNT_TYPE &instance) {
    MultinomialCobwebNode *newChild = new MultinomialCobwebNode();
    newChild->parent = this;
    newChild->tree = this->tree;
    newChild->increment_counts(instance);
    this->children.push_back(newChild);
    return newChild;
};

inline double MultinomialCobwebNode::mi_for_new_child(const AV_COUNT_TYPE &instance) {
    double children_entropy = 0.0;
    double info_c = 0.0;

    for (auto &c: this->children) {
        double p_of_child = (1.0 * c->count) / (this->count + 1.0);
        info_c -= p_of_child * log(p_of_child);
        children_entropy += p_of_child * c->entropy();
    }

    MultinomialCobwebNode new_child = MultinomialCobwebNode();
    new_child.parent = this;
    new_child.tree = this->tree;
    new_child.increment_counts(instance);
    double p_of_child = 1.0 / (this->count + 1.0);
    children_entropy += p_of_child * new_child.entropy();
    info_c -= p_of_child * log(p_of_child);

    return ((this->entropy_insert(instance) - children_entropy)
            // - info_c);
            / (children.size()+1));
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

inline double MultinomialCobwebNode::mi_for_merge(MultinomialCobwebNode *best1,
        MultinomialCobwebNode *best2, const AV_COUNT_TYPE &instance) {

    double children_entropy = 0.0;
    double info_c = 0.0;

    for (auto &c: children) {
        if (c == best1 || c == best2){
            continue;
        }

        double p_of_child = (1.0 * c->count) / (this->count + 1.0);
        info_c -= p_of_child * log(p_of_child);
        children_entropy += p_of_child * c->entropy();
    }

    double p_of_child = (best1->count + best2->count + 1.0) / (this->count + 1.0);
    info_c -= p_of_child * log(p_of_child);
    children_entropy += p_of_child * best1->entropy_merge(best2, instance);

    return ((this->entropy_insert(instance) - children_entropy)
            // - info_c);
            / (children.size()-1));
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

inline double MultinomialCobwebNode::mi_for_split(MultinomialCobwebNode *best){
    double children_entropy = 0.0;
    double info_c = 0.0;

    for (auto &c: children) {
        if (c == best) continue;

        double p_of_child = (1.0 * c->count) / this->count;
        info_c -= p_of_child * log(p_of_child);
        children_entropy += p_of_child * c->entropy();
    }

    for (auto &c: best->children) {
        double p_of_child = (1.0 * c->count) / this->count;
        info_c -= p_of_child * log(p_of_child);
        children_entropy += p_of_child * c->entropy();
    }

    return ((this->entropy() - children_entropy)
            // - info_c);
            / (children.size() - 1 + best->children.size()));
}

inline bool MultinomialCobwebNode::is_exact_match(const AV_COUNT_TYPE &instance) {
    unordered_set<ATTR_TYPE> all_attrs;
    for (auto &[attr, tmp]: instance) all_attrs.insert(attr);
    for (auto &[attr, tmp]: this->av_counts) all_attrs.insert(attr);

    for (auto &attr: all_attrs) {
        if (attr[0] == '_') continue;
        if (instance.count(attr) && !this->av_counts.count(attr)) {
            return false;
        }
        if (this->av_counts.count(attr) && !instance.count(attr)) {
            return false;
        }
        if (this->av_counts.count(attr) && instance.count(attr)) {
            double instance_attr_count = 0.0;
            unordered_set<VALUE_TYPE> all_vals;
            for (auto &[val, tmp]: this->av_counts.at(attr)) all_vals.insert(val);
            for (auto &[val, cnt]: instance.at(attr)){
                all_vals.insert(val);
                instance_attr_count += cnt;
            }

            for (auto &val: all_vals) {
                if (instance.at(attr).count(val) && !this->av_counts.at(attr).count(val)) {
                    return false;
                }
                if (this->av_counts.at(attr).count(val) && !instance.at(attr).count(val)) {
                    return false;
                }

                double instance_prob = (1.0 * instance.at(attr).at(val)) / instance_attr_count;
                double concept_prob = (1.0 * this->av_counts.at(attr).at(val)) / this->attr_counts.at(attr);

                if (abs(instance_prob - concept_prob) > 0.00001){
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

    // ret += "\"_expected_guesses\": {\n";
    ret += "\"_entropy\": {\n";
    ret += "\"#ContinuousValue#\": {\n";
    ret += "\"mean\": " + to_string(this->entropy()) + ",\n";
    ret += "\"std\": 1,\n";
    ret += "\"n\": 1,\n";
    ret += "}},\n";

    ret += "\"_mutual_info\": {\n";
    ret += "\"#ContinuousValue#\": {\n";
    ret += "\"mean\": " + to_string(this->mutual_information()) + ",\n";
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

inline string MultinomialCobwebNode::attr_counts_to_json() {
    string ret = "{";

    bool first =true;
    for (auto &[attr, cnt]: this->attr_counts) {
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
    output += "\"attr_counts\": " + this->attr_counts_to_json() + ",\n";
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
    unordered_set<ATTR_TYPE> allAttrs;
    for (auto &[attr, tmp]: this->av_counts) allAttrs.insert(attr);
    for (auto &[attr, tmp]: childLeaf->av_counts) allAttrs.insert(attr);

    double ll = 0;

    for (auto &attr: allAttrs) {
        if (attr[0] == '_') continue;
        unordered_set<VALUE_TYPE> vals;
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

inline double MultinomialCobwebNode::basic_mi(){
    double p_of_c = (1.0 * this->count) / this->tree->root->count;
    return (p_of_c * (this->tree->root->entropy() - this->entropy()));
}


inline double MultinomialCobwebNode::log_prob_class_given_instance(const AV_COUNT_TYPE &instance){
    
    double log_prob = 0;

    for (auto &[attr, vAttr]: instance) {
        bool hidden = attr[0] == '_';
        if (hidden || !this->tree->root->av_counts.count(attr)){
            continue;
        }

        double num_vals = this->tree->attr_vals.at(attr).size();
        // double num_vals = this->tree->root->av_counts.at(attr).size();

        for (auto &[val, cnt]: vAttr){
            if (!this->tree->root->av_counts.at(attr).count(val)){
                continue;
            }
        
            float av_count = this->tree->alpha(attr);
            if (this->av_counts.count(attr) && this->av_counts.at(attr).count(val)){
                av_count += this->av_counts.at(attr).at(val);
                // cout << val << "(" << this->av_counts.at(attr).at(val) << ") ";
            }

            // the cnt here is because we have to compute probability over all context words.
            log_prob += cnt * log((1.0 * av_count) / (this->attr_counts[attr] + num_vals * this->tree->alpha(attr)));
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
        .def("entropy", &MultinomialCobwebNode::entropy)
        .def("original_entropy", &MultinomialCobwebNode::original_entropy)
        .def("mutual_information", &MultinomialCobwebNode::mutual_information)
        .def("__str__", &MultinomialCobwebNode::__str__)
        .def_readonly("count", &MultinomialCobwebNode::count)
        .def_readonly("concept_id", &MultinomialCobwebNode::concept_id)
        .def_readonly("children", &MultinomialCobwebNode::children)
        .def_readonly("parent", &MultinomialCobwebNode::parent)
        .def_readonly("av_counts", &MultinomialCobwebNode::av_counts)
        .def_readonly("attr_counts", &MultinomialCobwebNode::attr_counts)
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
