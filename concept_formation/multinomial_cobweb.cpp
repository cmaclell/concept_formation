#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <queue>
#include <unordered_map>
#include <functional>
#include <random>
#include <tuple>
#include <unordered_set>
#include <chrono>
#include <cmath>

#include "assert.h"
#include "json.hpp"
#include "cached_string.hpp"

namespace py = pybind11;

#define NULL_STRING CachedString("\0")
#define BEST 0
#define NEW 1
#define MERGE 2
#define SPLIT 3

typedef CachedString ATTR_TYPE;
typedef CachedString VALUE_TYPE;
typedef double COUNT_TYPE;
typedef std::unordered_map<std::string, std::unordered_map<std::string, COUNT_TYPE>> INSTANCE_TYPE;
typedef std::unordered_map<VALUE_TYPE, COUNT_TYPE> VAL_COUNT_TYPE;
typedef std::unordered_map<ATTR_TYPE, VAL_COUNT_TYPE> AV_COUNT_TYPE;
typedef std::unordered_map<ATTR_TYPE, std::unordered_set<VALUE_TYPE>> AV_KEY_TYPE;
typedef std::unordered_map<ATTR_TYPE, COUNT_TYPE> ATTR_COUNT_TYPE;
typedef std::pair<double, int> OPERATION_TYPE;

class MultinomialCobwebTree;
class MultinomialCobwebNode;

std::random_device rd;
std::mt19937_64 gen(rd());
std::uniform_real_distribution<double> unif(0, 1);

double custom_rand() {
    return unif(gen);
}

std::string repeat(std::string s, int n) {
    std::string res = "";
    for (int i = 0; i < n; i++) {
        res += s;
    }
    return res;
}

std::string doubleToString(double cnt) {
    std::ostringstream stream;
    // Set stream to output floating point numbers with maximum precision
    stream << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10) << cnt;
    return stream.str();
}

class MultinomialCobwebNode {

    public:
        MultinomialCobwebTree *tree;
        MultinomialCobwebNode *parent;
        std::vector<MultinomialCobwebNode *> children;

        COUNT_TYPE count;
        ATTR_COUNT_TYPE a_counts;
        AV_COUNT_TYPE av_counts;

        MultinomialCobwebNode();
        MultinomialCobwebNode(MultinomialCobwebNode *otherNode);
        void increment_counts(const AV_COUNT_TYPE &instance);
        void update_counts_from_node(MultinomialCobwebNode *node);
        double entropy_insert(const AV_COUNT_TYPE &instance);
        double entropy_merge(MultinomialCobwebNode *other, const AV_COUNT_TYPE
                &instance);
        MultinomialCobwebNode* get_best_level(INSTANCE_TYPE instance);
        MultinomialCobwebNode* get_basic_level();
        double category_utility();
        double cu_given_instance(const AV_COUNT_TYPE &instance, double parent_prob);
        double entropy();
        double partition_utility();
        std::tuple<double, int> get_best_operation(const AV_COUNT_TYPE
                &instance, MultinomialCobwebNode *best1, MultinomialCobwebNode
                *best2, double best1Cu);
        std::tuple<double, MultinomialCobwebNode *, MultinomialCobwebNode *>
            two_best_children(const AV_COUNT_TYPE &instance);
        std::vector<double> prob_children_given_instance(const AV_COUNT_TYPE &instance);
        std::vector<double> prob_children_given_instance_ext(INSTANCE_TYPE instance);
        double log_prob_instance(const AV_COUNT_TYPE &instance);
        double log_prob_instance_ext(INSTANCE_TYPE instance);
        double log_prob_class_given_instance(const AV_COUNT_TYPE &instance,
                bool use_root_counts=false);
        double log_prob_class_given_instance_ext(INSTANCE_TYPE instance,
                bool use_root_counts=false);
        double pu_for_insert(MultinomialCobwebNode *child, const AV_COUNT_TYPE
                &instance);
        double pu_for_new_child(const AV_COUNT_TYPE &instance);
        double pu_for_merge(MultinomialCobwebNode *best1, MultinomialCobwebNode
                *best2, const AV_COUNT_TYPE &instance);
        double pu_for_split(MultinomialCobwebNode *best);
        bool is_exact_match(const AV_COUNT_TYPE &instance);
        size_t _hash();
        std::string __str__();
        std::string concept_hash();
        std::string pretty_print(int depth = 0);
        int depth();
        bool is_parent(MultinomialCobwebNode *otherConcept);
        int num_concepts();
        std::string avcounts_to_json();
        std::string ser_avcounts();
        std::string a_counts_to_json();
        std::string dump_json();
        std::string output_json();
        std::vector<std::tuple<VALUE_TYPE, double>>
            get_weighted_values(ATTR_TYPE attr, bool allowNone = true);
        std::unordered_map<std::string, std::unordered_map<std::string, double>> predict_probs();
        std::unordered_map<std::string, std::unordered_map<std::string, double>> predict_weighted_probs(INSTANCE_TYPE instance);
        std::unordered_map<std::string, std::unordered_map<std::string, double>> predict_weighted_leaves_probs(INSTANCE_TYPE instance);
        VALUE_TYPE predict(ATTR_TYPE attr, std::string choiceFn = "most likely",
                bool allowNone = true);

};


class MultinomialCobwebTree {

    public:
        float alpha;
        bool weight_attr;
        MultinomialCobwebNode *root;
        AV_KEY_TYPE attr_vals;

        MultinomialCobwebTree(float alpha, bool weight_attr) {
            this->alpha = alpha;
            this->weight_attr = weight_attr;

            this->root = new MultinomialCobwebNode();
            this->root->tree = this;
            this->attr_vals = AV_KEY_TYPE();
        }

        std::string __str__(){
            return this->root->__str__();
        }

        MultinomialCobwebNode* load_json_helper(json_object_s* object) {
            MultinomialCobwebNode *new_node = new MultinomialCobwebNode();
            new_node->tree = this;

            // // Get concept_id
            // struct json_object_element_s* concept_id_obj = object->start;
            // unsigned long long concept_id_val = stoull(json_value_as_number(concept_id_obj->value)->number);
            // new_node->concept_id = concept_id_val;
            // new_node->update_counter(concept_id_val);

            // Get count
            struct json_object_element_s* count_obj = object->start;
            // struct json_object_element_s* count_obj = object->start;
            double count_val = atof(json_value_as_number(count_obj->value)->number);
            new_node->count = count_val;

            // Get a_counts
            struct json_object_element_s* a_counts_obj = count_obj->next;
            struct json_object_s* a_counts_dict = json_value_as_object(a_counts_obj->value);
            struct json_object_element_s* a_counts_cursor = a_counts_dict->start;
            while(a_counts_cursor != NULL) {
                // Get attr name
                std::string attr_name = std::string(a_counts_cursor->name->string);

                // A count is stored with each attribute
                double count_value = atof(json_value_as_number(a_counts_cursor->value)->number);
                new_node->a_counts[attr_name] = count_value;

                a_counts_cursor = a_counts_cursor->next;
            }

            // Get av counts
            struct json_object_element_s* av_counts_obj = a_counts_obj->next;
            struct json_object_s* av_counts_dict = json_value_as_object(av_counts_obj->value);
            struct json_object_element_s* av_counts_cursor = av_counts_dict->start;
            while(av_counts_cursor != NULL) {
                // Get attr name
                std::string attr_name = std::string(av_counts_cursor->name->string);

                // The attr val is a dict of strings to ints
                struct json_object_s* attr_val_dict = json_value_as_object(av_counts_cursor->value);
                struct json_object_element_s* inner_counts_cursor = attr_val_dict->start;
                while(inner_counts_cursor != NULL) {
                    // this will be a word
                    std::string val_name = std::string(inner_counts_cursor->name->string);

                    // This will always be a number
                    double attr_val_count = atof(json_value_as_number(inner_counts_cursor->value)->number);
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
            std::vector<MultinomialCobwebNode*> new_children;
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

        std::string dump_json(){
            std::string output = "{";

            output += "\"alpha\": " + doubleToString(this->alpha) + ",\n";
            output += "\"weight_attr\": " + std::to_string(this->weight_attr) + ",\n";
            output += "\"root\": " + this->root->dump_json();
            output += "}\n";

            return output;
            // return this->root->dump_json();
        }

        void load_json(std::string json) {
            struct json_value_s* tree = json_parse(json.c_str(), strlen(json.c_str()));
            struct json_object_s* object = (struct json_object_s*)tree->payload;

            // alpha
            struct json_object_element_s* alpha_obj = object->start;
            double alpha = atof(json_value_as_number(alpha_obj->value)->number);
            this->alpha = alpha;

            // weight_attr
            struct json_object_element_s* weight_attr_obj = alpha_obj->next;
            bool weight_attr = bool(atoi(json_value_as_number(weight_attr_obj->value)->number));
            this->weight_attr = weight_attr;

            // root
            struct json_object_element_s* root_obj = weight_attr_obj->next;
            struct json_object_s* root = json_value_as_object(root_obj->value);

            delete this->root;
            this->root = this->load_json_helper(root);

            for (auto &[attr, val_map]: this->root->av_counts) {
                for (auto &[val, cnt]: val_map) {
                    this->attr_vals[attr].insert(val);
                }
            }
        }

        void clear() {
            delete this->root;
            this->root = new MultinomialCobwebNode();
            this->root->tree = this;
            this->attr_vals = AV_KEY_TYPE();
        }

        MultinomialCobwebNode* ifit_helper(const INSTANCE_TYPE &instance){
            AV_COUNT_TYPE cached_instance;
            for (auto &[attr, val_map]: instance) {
                for (auto &[val, cnt]: val_map) {
                    cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
                }
            }
            return this->cobweb(cached_instance);
        }

        MultinomialCobwebNode* ifit(INSTANCE_TYPE instance) {
            return this->ifit_helper(instance);
        }

        void fit(std::vector<INSTANCE_TYPE> instances, int iterations = 1, bool randomizeFirst = true) {
            for (int i = 0; i < iterations; i++) {
                if (i == 0 && randomizeFirst) {
                    shuffle(instances.begin(), instances.end(), std::default_random_engine());
                }
                for (auto &instance: instances) {
                    this->ifit(instance);
                }
                shuffle(instances.begin(), instances.end(), std::default_random_engine());
            }
        }

        MultinomialCobwebNode* cobweb(const AV_COUNT_TYPE &instance) {
            // std::cout << "cobweb top level" << std::endl;

            for (auto &[attr, val_map]: instance) {
                for (auto &[val, cnt]: val_map) {
                    attr_vals[attr].insert(val);
                }
            }

            MultinomialCobwebNode* current = root;

            while (true) {
                if (current->children.empty() && (current->count == 0 || current->is_exact_match(instance))) {
                    // std::cout << "empty / exact match" << std::endl;
                    current->increment_counts(instance);
                    break;
                } else if (current->children.empty()) {
                    // std::cout << "fringe split" << std::endl;
                    MultinomialCobwebNode* new_node = new MultinomialCobwebNode(current);
                    current->parent = new_node;
                    new_node->children.push_back(current);

                    if (new_node->parent == nullptr) {
                        root = new_node;
                    }
                    else{
                        new_node->parent->children.erase(remove(new_node->parent->children.begin(),
                                    new_node->parent->children.end(), current), new_node->parent->children.end());
                        new_node->parent->children.push_back(new_node);
                    }
                    new_node->increment_counts(instance);

                    current = new MultinomialCobwebNode();
                    current->parent = new_node;
                    current->tree = this;
                    current->increment_counts(instance);
                    new_node->children.push_back(current);
                    break;

                } else {
                    auto[best1_mi, best1, best2] = current->two_best_children(instance);
                    auto[_, bestAction] = current->get_best_operation(instance, best1, best2, best1_mi);

                    if (bestAction == BEST) {
                        // std::cout << "best" << std::endl;
                        current->increment_counts(instance);
                        current = best1;

                    } else if (bestAction == NEW) {
                        // std::cout << "new" << std::endl;
                        current->increment_counts(instance);

                        // current = current->create_new_child(instance);
                        MultinomialCobwebNode *new_child = new MultinomialCobwebNode();
                        new_child->parent = current;
                        new_child->tree = this;
                        new_child->increment_counts(instance);
                        current->children.push_back(new_child);
                        current = new_child;
                        break;

                    } else if (bestAction == MERGE) {
                        // std::cout << "merge" << std::endl;
                        current->increment_counts(instance);
                        // MultinomialCobwebNode* new_child = current->merge(best1, best2);

                        MultinomialCobwebNode *new_child = new MultinomialCobwebNode();
                        new_child->parent = current;
                        new_child->tree = this;

                        new_child->update_counts_from_node(best1);
                        new_child->update_counts_from_node(best2);
                        best1->parent = new_child;
                        best2->parent = new_child;
                        new_child->children.push_back(best1);
                        new_child->children.push_back(best2);
                        current->children.erase(remove(current->children.begin(),
                                    current->children.end(), best1), current->children.end());
                        current->children.erase(remove(current->children.begin(),
                                    current->children.end(), best2), current->children.end());
                        current->children.push_back(new_child);
                        current = new_child;
                        
                    } else if (bestAction == SPLIT) {
                        // std::cout << "split" << std::endl;
                        current->children.erase(remove(current->children.begin(),
                                    current->children.end(), best1), current->children.end());
                        for (auto &c: best1->children) {
                            c->parent = current;
                            c->tree = this;
                            current->children.push_back(c);
                        }
                        delete best1;

                    } else {
                        throw "Best action choice \"" + std::to_string(bestAction) +
                            "\" (best=0, new=1, merge=2, split=3) not a recognized option. This should be impossible...";
                    }
                }
            }
            return current;
        }

        MultinomialCobwebNode* _cobweb_categorize(const AV_COUNT_TYPE &instance) {

            auto current = this->root;

            while (true) {
                if (current->children.empty()) {
                    return current;
                }

                auto parent = current;
                current = nullptr;
                double best_logp;

                for (auto &child: parent->children) {
                    double logp = child->log_prob_class_given_instance(instance, false);
                    if (current == nullptr || logp > best_logp){
                        best_logp = logp;
                        current = child;
                    }
                }
            }
        }

        MultinomialCobwebNode* categorize_helper(const INSTANCE_TYPE &instance){
            AV_COUNT_TYPE cached_instance;
            for (auto &[attr, val_map]: instance) {
                for (auto &[val, cnt]: val_map) {
                    cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
                }
            }
            return this->_cobweb_categorize(cached_instance);
        }

        MultinomialCobwebNode* categorize(const INSTANCE_TYPE instance) {
            return this->categorize_helper(instance);
        }

        std::unordered_map<std::string, std::unordered_map<std::string, double>> predict_probs_mixture_helper(const AV_COUNT_TYPE &instance, double ll_path, int max_nodes, bool greedy, int obj){

            std::unordered_map<std::string, std::unordered_map<std::string, double>> out;
            int nodes_expanded = 0;
            double total_weight = 0.0;

            double root_ll_inst = 0;
            root_ll_inst = this->root->log_prob_instance(instance);

            auto queue = std::priority_queue<
                std::tuple<double, double, MultinomialCobwebNode*>>();

            double score = 0.0;
            if (obj == 0){
                score = 1.0;
            } else if (obj == 1){
                score = exp(root_ll_inst);
            } else if (obj == 2){
                score = this->root->cu_given_instance(instance, 1.0);
            } else if (obj == 3){
                // score = 0.0;
                score = exp(2 * root_ll_inst);
            }

            queue.push(std::make_tuple(score, 0.0, this->root));

            while (queue.size() > 0){
                auto node = queue.top();
                queue.pop();
                nodes_expanded += 1;

                if (greedy){
                    queue = std::priority_queue<
                        std::tuple<double, double, MultinomialCobwebNode*>>();
                }

                auto curr_score = std::get<0>(node);
                auto curr_ll = std::get<1>(node);
                auto curr = std::get<2>(node);

                if (curr_score < 0){
                    curr_score = 0;
                }

                total_weight += curr_score;

                auto curr_preds = curr->predict_probs();

                for (auto &[attr, val_set]: curr_preds) {
                    for (auto &[val, p]: val_set) {
                        out[attr][val] += curr_score * p;
                    }
                }

                if (nodes_expanded >= max_nodes) break;

                // TODO look at missing in computing prob children given instance
                std::vector<double> children_probs = curr->prob_children_given_instance(instance);

                for (size_t i = 0; i < curr->children.size(); ++i) {
                    auto child = curr->children[i];
                    double child_ll_inst = 0;
                    child_ll_inst = child->log_prob_instance(instance);
                    auto child_ll_given_parent = log(children_probs[i]);
                    auto child_ll = child_ll_given_parent + curr_ll;

                    double score = 0;
                    if (obj == 0){
                        score = exp(child_ll);
                    } else if (obj == 1){
                        score = exp(child_ll_inst + child_ll);
                    } else if (obj == 2){
                        // double p_of_c = (child->count * 1.0) / (this->root->count);
                        // score = exp(child_ll) * 1/p_of_c * child->category_utility();
                        score = child->cu_given_instance(instance, exp(child_ll));
                        // score = exp(child_ll) * (exp(child_ll_inst) - exp(root_ll_inst));
                    } else if (obj == 3){
                        // score = child->count / this->root->count * exp(2 * child_ll_inst);
                        score = child->count / this->root->count * exp(2 * child_ll_inst);
                        // score = exp(child_ll) * (child_ll_inst - root_ll_inst);
                    }

                    queue.push(std::make_tuple(score, child_ll, child));
                }
            }

            for (auto &[attr, val_set]: out) {
                for (auto &[val, p]: val_set) {
                    out[attr][val] /= total_weight;
                }
            }

            return out;
        }

        std::unordered_map<std::string, std::unordered_map<std::string, double>> predict_probs_mixture(INSTANCE_TYPE instance, int max_nodes, bool greedy, int obj){
           AV_COUNT_TYPE cached_instance;
            for (auto &[attr, val_map]: instance) {
                for (auto &[val, cnt]: val_map) {
                    cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
                }
            }
            return this->predict_probs_mixture_helper(cached_instance, 0.0,
                    max_nodes, greedy, obj);
        }

};

inline MultinomialCobwebNode::MultinomialCobwebNode() {
    count = 0;
    a_counts = ATTR_COUNT_TYPE();
    parent = nullptr;
    tree = nullptr;
}

inline MultinomialCobwebNode::MultinomialCobwebNode(MultinomialCobwebNode *otherNode) {
    count = 0;
    a_counts = ATTR_COUNT_TYPE();

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
            this->a_counts[attr] += cnt;
            this->av_counts[attr][val] += cnt;
        }
    }
}

inline void MultinomialCobwebNode::update_counts_from_node(MultinomialCobwebNode *node) {
    this->count += node->count;

    for (auto &[attr, val_map]: node->av_counts) {
        this->a_counts[attr] += node->a_counts.at(attr);

        for (auto&[val, cnt]: val_map) {
            this->av_counts[attr][val] += cnt;
        }
    }
}

inline double MultinomialCobwebNode::entropy_insert(const AV_COUNT_TYPE &instance){
    // TODO
    return 0.0;
}

inline double MultinomialCobwebNode::entropy_merge(MultinomialCobwebNode *other,
        const AV_COUNT_TYPE &instance) {

    // TODO
    return 0.0;

}

inline MultinomialCobwebNode* MultinomialCobwebNode::get_best_level(
        INSTANCE_TYPE instance){

    AV_COUNT_TYPE cached_instance;
    for (auto &[attr, val_map]: instance) {
        for (auto &[val, cnt]: val_map) {
            cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
        }
    }

    MultinomialCobwebNode* curr = this;
    MultinomialCobwebNode* best = this;
    double best_ll = this->log_prob_class_given_instance(cached_instance, true);

    while (curr->parent != nullptr) {
        curr = curr->parent;
        double curr_ll = curr->log_prob_class_given_instance(cached_instance, true);

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
    double best_cu = this->category_utility();

    while (curr->parent != nullptr) {
        curr = curr->parent;
        double curr_cu = curr->category_utility();

        if (curr_cu > best_cu) {
            best = curr;
            best_cu = curr_cu;
        }
    }

    return best;
}

inline double MultinomialCobwebNode::entropy() {

    // TODO 
    return 0.0;

    /*
    double info = 0.0;
    for (auto &[attr, inner_av]: this->av_counts){
        if (attr.is_hidden()) continue;
        info += this->entropy_attr(attr);
    }

    return info;
    */
}


inline std::tuple<double, int> MultinomialCobwebNode::get_best_operation(
        const AV_COUNT_TYPE &instance, MultinomialCobwebNode *best1,
        MultinomialCobwebNode *best2, double best1_pu){

    if (best1 == nullptr) {
        throw "Need at least one best child.";
    }
    std::vector<std::tuple<double, double, int>> operations;
    operations.push_back(std::make_tuple(best1_pu,
                custom_rand(),
                BEST));

    operations.push_back(std::make_tuple(pu_for_new_child(instance),
                custom_rand(),
                NEW));
    if (children.size() > 2 && best2 != nullptr) {
        operations.push_back(std::make_tuple(pu_for_merge(best1, best2,
                        instance),
                    custom_rand(),
                    MERGE));
    }

    if (best1->children.size() > 0) {
        operations.push_back(std::make_tuple(pu_for_split(best1),
                    custom_rand(),
                    SPLIT));
    }

    sort(operations.rbegin(), operations.rend());

    OPERATION_TYPE bestOp = std::make_pair(std::get<0>(operations[0]), std::get<2>(operations[0]));
    return bestOp;
}

inline std::tuple<double, MultinomialCobwebNode *, MultinomialCobwebNode *> MultinomialCobwebNode::two_best_children(
        const AV_COUNT_TYPE &instance) {

    if (children.empty()) {
        throw "No children!";
    }

    /*
    // DO RELATIVE PU, requires only B
    std::vector<std::tuple<double, double, double, MultinomialCobwebNode *>> relative_pu;
    for (auto &child: this->children) {
        relative_pu.push_back(
                std::make_tuple(
                    (child->count * child->entropy()) -
                    ((child->count + 1) * child->entropy_insert(instance)),
                    child->count,
                    custom_rand(),
                    child));
    }

    sort(relative_pu.rbegin(), relative_pu.rend());
    MultinomialCobwebNode *best1 = std::get<3>(relative_pu[0]);
    double best1_pu = pu_for_insert(best1, instance);
    MultinomialCobwebNode *best2 = relative_pu.size() > 1 ? std::get<3>(relative_pu[1]) : nullptr;
    */

    // Evaluate each insert, requires B^2 where B is branching factor
    std::vector<std::tuple<double, double, double, MultinomialCobwebNode *>> pus;
    for (auto &child: this->children) {
        pus.push_back(
            std::make_tuple(
                pu_for_insert(child, instance),
                child->count,
                custom_rand(),
                child));
    }
    sort(pus.rbegin(), pus.rend());
    MultinomialCobwebNode *best1 = std::get<3>(pus[0]);
    double best1_pu = std::get<0>(pus[0]);
    MultinomialCobwebNode *best2 = pus.size() > 1 ? std::get<3>(pus[1]) : nullptr;

    return std::make_tuple(best1_pu, best1, best2);
}

inline double MultinomialCobwebNode::partition_utility() {
    if (this->children.size() == 0) return 0.0;

    double score = 0.0;

    for (auto &c: children){
        double p_of_child = (c->count * 1.0) / (this->count);

        for (auto &[attr, vAttr]: this->av_counts) {

            double ratio = 1.0;
            if (this->tree->weight_attr){
                ratio = (1.0 * this->tree->root->a_counts.at(attr)) / (this->tree->root->count);
                // ratio = (1.0 * attr_count) / this->count;
            }

            int num_vals = this->tree->attr_vals.at(attr).size();
            double n_a = this->a_counts.at(attr) + num_vals * this->tree->alpha;
            double n_a_given_c = num_vals * this->tree->alpha;

            if (c->a_counts.count(attr)){
                n_a_given_c += c->a_counts.at(attr);
            }

            for (auto &[val, cnt]: vAttr) {
                double n_av = cnt + this->tree->alpha;
                double n_av_given_c = this->tree->alpha;

                if (c->av_counts.count(attr) and c->av_counts.at(attr).count(val)){
                    n_av_given_c += c->av_counts.at(attr).at(val);
                }

                double p_of_av = n_av / n_a;
                double p_of_av_given_c = n_av_given_c / n_a_given_c;
                // std::cout << "p(av): " << p_of_av << std::endl;
                // std::cout << "p(av|c): " << p_of_av << std::endl;
                // std::cout << "score(av|c): " << p_of_child * p_of_av_given_c * (log(p_of_av_given_c) - log(p_of_av)) << std::endl;
                score += ratio * p_of_child * p_of_av_given_c * (log(p_of_av_given_c) - log(p_of_av));
                // score += ratio * p_of_child * (p_of_av_given_c * log(p_of_av_given_c) - p_of_av * log(p_of_av));
            }
        }
    }

    return score / this->children.size();
}

inline double MultinomialCobwebNode::pu_for_insert(MultinomialCobwebNode *child, const AV_COUNT_TYPE &instance) {
    double score = 0.0;

    for (auto &c: children){
        double p_of_child = (c->count * 1.0) / (this->count + 1.0);
        if (c == child){
            p_of_child = (c->count + 1.0) / (this->count + 1.0);
        }

        for (auto &[attr, vAttr]: this->av_counts) {
            double ratio = 1.0;
            if (this->tree->weight_attr){
                ratio = (1.0 * this->tree->root->a_counts.at(attr)) / (this->tree->root->count);
                // ratio = (1.0 * attr_count) / this->count;
            }

            int num_vals = this->tree->attr_vals.at(attr).size();
            double n_a = this->a_counts.at(attr) + num_vals * this->tree->alpha;
            double n_a_given_c = num_vals * this->tree->alpha;

            if (c->a_counts.count(attr)){
                n_a_given_c += c->a_counts.at(attr);
            }

            for (auto &[val, cnt]: vAttr) {
                double n_av = cnt + this->tree->alpha;
                double n_av_given_c = this->tree->alpha;

                if (c->av_counts.count(attr) and c->av_counts.at(attr).count(val)){
                    n_av_given_c += c->av_counts.at(attr).at(val);
                }

                if (instance.count(attr) and instance.at(attr).count(val)){
                    n_a += instance.at(attr).at(val);
                    n_av += instance.at(attr).at(val);

                    if (c == child){
                        n_a_given_c += instance.at(attr).at(val);
                        n_av_given_c += instance.at(attr).at(val);
                    }
                }

                double p_of_av = n_av / n_a;
                double p_of_av_given_c = n_av_given_c / n_a_given_c;
                score += ratio * p_of_child * p_of_av_given_c * (log(p_of_av_given_c) - log(p_of_av));
                // score += ratio * p_of_child * (p_of_av_given_c * log(p_of_av_given_c) - p_of_av * log(p_of_av));
            }
        }
    }

    return score / this->children.size();
}

inline double MultinomialCobwebNode::pu_for_new_child(const AV_COUNT_TYPE &instance) {

    double score = 0.0;

    for (auto &c: children){
        double p_of_child = (c->count * 1.0) / (this->count + 1.0);

        for (auto &[attr, vAttr]: this->av_counts) {
            double ratio = 1.0;
            if (this->tree->weight_attr){
                ratio = (1.0 * this->tree->root->a_counts.at(attr)) / (this->tree->root->count);
                // ratio = (1.0 * attr_count) / this->count;
            }

            int num_vals = this->tree->attr_vals.at(attr).size();
            double n_a = this->a_counts.at(attr) + num_vals * this->tree->alpha;
            double n_a_given_c = num_vals * this->tree->alpha;

            if (c->a_counts.count(attr)){
                n_a_given_c += c->a_counts.at(attr);
            }

            for (auto &[val, cnt]: vAttr) {
                double n_av = cnt + this->tree->alpha;
                double n_av_given_c = this->tree->alpha;

                if (c->av_counts.count(attr) and c->av_counts.at(attr).count(val)){
                    n_av_given_c += c->av_counts.at(attr).at(val);
                }

                if (instance.count(attr) and instance.at(attr).count(val)){
                    n_a += instance.at(attr).at(val);
                    n_av += instance.at(attr).at(val);
                }

                double p_of_av = n_av / n_a;
                double p_of_av_given_c = n_av_given_c / n_a_given_c;
                score += ratio * p_of_child * p_of_av_given_c * (log(p_of_av_given_c) - log(p_of_av));
                // score += ratio * p_of_child * (p_of_av_given_c * log(p_of_av_given_c) - p_of_av * log(p_of_av));
            }
        }
    }

    // DO NEW
    double p_of_child = 1.0 / (this->count + 1.0);

    for (auto &[attr, vAttr]: this->av_counts) {
        double ratio = 1.0;
        if (this->tree->weight_attr){
            ratio = (1.0 * this->tree->root->a_counts.at(attr)) / (this->tree->root->count);
            // ratio = (1.0 * attr_count) / this->count;
        }
        
        int num_vals = this->tree->attr_vals.at(attr).size();
        double n_a = this->a_counts.at(attr) + num_vals * this->tree->alpha;
        double n_a_given_c = num_vals * this->tree->alpha;

        for (auto &[val, cnt]: vAttr) {
            double n_av = cnt + this->tree->alpha;
            double n_av_given_c = this->tree->alpha;

            if (instance.count(attr) and instance.at(attr).count(val)){
                n_a += instance.at(attr).at(val);
                n_av += instance.at(attr).at(val);
            }

            double p_of_av = n_av / n_a;
            double p_of_av_given_c = n_av_given_c / n_a_given_c;
            score += ratio * p_of_child * p_of_av_given_c * (log(p_of_av_given_c) - log(p_of_av));
            // score += ratio * p_of_child * (p_of_av_given_c * log(p_of_av_given_c) - p_of_av * log(p_of_av));
        }
    }

    return score / (this->children.size() + 1);

    /*
    double p_of_new_child = 1.0 / (this->count + 1.0);

    double parent_entropy = 0.0;
    double children_entropy = 0.0;
    double concept_entropy = -p_of_new_child * log(p_of_new_child);


    for (auto &[attr, val_set]: this->tree->attr_vals) {
        children_entropy += p_of_new_child * new_child.entropy_attr(attr);
        parent_entropy += this->entropy_attr_insert(attr, instance);
    }

    for (auto &child: children) {
        double p_of_child = (1.0 * child->count) / (this->count + 1.0);
        concept_entropy -= p_of_child * log(p_of_child);

        for (auto &[attr, val_set]: this->tree->attr_vals) {
            children_entropy += p_of_child * child->entropy_attr(attr);
        }
    }

    double obj = (parent_entropy - children_entropy);
    if (this->tree->mutual_info == 1){
        obj /= parent_entropy;
    }
    else if (this->tree->mutual_info == 2){
        obj /= (children_entropy + concept_entropy);
    }
    if (this->tree->children_norm){
        obj /= (this->children.size() + 1);
    }
    return obj;
    */
}

inline double MultinomialCobwebNode::pu_for_merge(MultinomialCobwebNode *best1,
        MultinomialCobwebNode *best2, const AV_COUNT_TYPE &instance) {
    double score = 0.0;
    double parent_score = 0.0;
    double children_score = 0.0;

    for (auto &c: children){
        if (c == best2) continue;

        double p_of_child = (c->count * 1.0) / (this->count + 1.0);
        if (c == best1){
            p_of_child = (best1->count + best2->count + 1.0) / (this->count + 1.0);
        }

        for (auto &[attr, vAttr]: this->av_counts) {
            double ratio = 1.0;
            if (this->tree->weight_attr){
                ratio = (1.0 * this->tree->root->a_counts.at(attr)) / (this->tree->root->count);
                // ratio = (1.0 * attr_count) / this->count;
            }
            
            int num_vals = this->tree->attr_vals.at(attr).size();
            double n_a = this->a_counts.at(attr) + num_vals * this->tree->alpha;
            double n_a_given_c = num_vals * this->tree->alpha;

            if (c->a_counts.count(attr)){
                n_a_given_c += c->a_counts.at(attr);
            }

            if (c == best1){
                if (best2->a_counts.count(attr)){
                    n_a_given_c += best2->a_counts.at(attr);
                }
            }

            for (auto &[val, cnt]: vAttr) {
                double n_av = cnt + this->tree->alpha;
                double n_av_given_c = this->tree->alpha;

                if (c->av_counts.count(attr) and c->av_counts.at(attr).count(val)){
                    n_av_given_c += c->av_counts.at(attr).at(val);
                }

                if (c == best1){
                    if (best2->av_counts.count(attr) and best2->av_counts.at(attr).count(val)){
                        n_av_given_c += best2->av_counts.at(attr).at(val);
                    }
                }

                if (instance.count(attr) and instance.at(attr).count(val)){
                    n_a += instance.at(attr).at(val);
                    n_av += instance.at(attr).at(val);

                    if (c == best1){
                        n_a_given_c += instance.at(attr).at(val);
                        n_av_given_c += instance.at(attr).at(val);
                    }
                }

                double p_of_av = n_av / n_a;
                double p_of_av_given_c = n_av_given_c / n_a_given_c;
                score += ratio * p_of_child * p_of_av_given_c * (log(p_of_av_given_c) - log(p_of_av));
                // score += ratio * p_of_child * (p_of_av_given_c * log(p_of_av_given_c) - p_of_av * log(p_of_av));
            }
        }
    }

    return score / (this->children.size() - 1);
    
    /*
    double parent_entropy = 0.0;
    double children_entropy = 0.0;
    double p_of_merged = (best1->count + best2->count + 1.0) / (this->count + 1.0);
    double concept_entropy = -p_of_merged * log(p_of_merged);

    for (auto &[attr, val_set]: this->tree->attr_vals) {
        parent_entropy += this->entropy_attr_insert(attr, instance);
        children_entropy += p_of_merged * best1->entropy_attr_merge(attr, best2, instance);
    }

    for (auto &child: children) {
        if (child == best1 || child == best2){
            continue;
        }
        double p_of_child = (1.0 * child->count) / (this->count + 1.0);
        concept_entropy -= p_of_child * log(p_of_child);

        for (auto &[attr, val_set]: this->tree->attr_vals) {
            children_entropy += p_of_child * child->entropy_attr(attr);
        }
    }

    double obj = (parent_entropy - children_entropy);
    if (this->tree->mutual_info == 1){
        obj /= parent_entropy;
    }
    else if (this->tree->mutual_info == 2){
        obj /= (children_entropy + concept_entropy);
    }
    if (this->tree->children_norm){
        obj /= (this->children.size() - 1);
    }
    return obj;
    */
}

inline double MultinomialCobwebNode::pu_for_split(MultinomialCobwebNode *best){
    double score = 0.0;

    for (auto &c: children){
        if (c == best) continue;

        double p_of_child = (c->count * 1.0) / (this->count);

        for (auto &[attr, vAttr]: this->av_counts) {
            double ratio = 1.0;
            if (this->tree->weight_attr){
                ratio = (1.0 * this->tree->root->a_counts.at(attr)) / (this->tree->root->count);
                // ratio = (1.0 * attr_count) / this->count;
            }

            int num_vals = this->tree->attr_vals.at(attr).size();
            double n_a = this->a_counts.at(attr) + num_vals * this->tree->alpha;
            double n_a_given_c = num_vals * this->tree->alpha;

            if (c->a_counts.count(attr)){
                n_a_given_c += c->a_counts.at(attr);
            }

            for (auto &[val, cnt]: vAttr) {
                double n_av = cnt + this->tree->alpha;
                double n_av_given_c = this->tree->alpha;

                if (c->av_counts.count(attr) and c->av_counts.at(attr).count(val)){
                    n_av_given_c += c->av_counts.at(attr).at(val);
                }

                double p_of_av = n_av / n_a;
                double p_of_av_given_c = n_av_given_c / n_a_given_c;
                score += ratio * p_of_child * p_of_av_given_c * (log(p_of_av_given_c) - log(p_of_av));
                // score += ratio * p_of_child * (p_of_av_given_c * log(p_of_av_given_c) - p_of_av * log(p_of_av));
            }
        }
    }

    for (auto &c: best->children){
        double p_of_child = (c->count * 1.0) / (this->count);

        for (auto &[attr, vAttr]: this->av_counts) {
            double ratio = 1.0;
            if (this->tree->weight_attr){
                ratio = (1.0 * this->tree->root->a_counts.at(attr)) / (this->tree->root->count);
                // ratio = (1.0 * attr_count) / this->count;
            }
            
            int num_vals = this->tree->attr_vals.at(attr).size();
            double n_a = this->a_counts.at(attr) + num_vals * this->tree->alpha;
            double n_a_given_c = num_vals * this->tree->alpha;

            if (c->a_counts.count(attr)){
                n_a_given_c += c->a_counts.at(attr);
            }

            for (auto &[val, cnt]: vAttr) {
                double n_av = cnt + this->tree->alpha;
                double n_av_given_c = this->tree->alpha;

                if (c->av_counts.count(attr) and c->av_counts.at(attr).count(val)){
                    n_av_given_c += c->av_counts.at(attr).at(val);
                }

                double p_of_av = n_av / n_a;
                double p_of_av_given_c = n_av_given_c / n_a_given_c;
                score += ratio * p_of_child * p_of_av_given_c * (log(p_of_av_given_c) - log(p_of_av));
                // score += ratio * p_of_child * (p_of_av_given_c * log(p_of_av_given_c) - p_of_av * log(p_of_av));
            }
        }
    }

    return score / (this->children.size() - 1 + best->children.size());

    /*
    double parent_entropy = 0.0;
    double children_entropy = 0.0;
    double concept_entropy = 0.0;

    for (auto &[attr, val_set]: this->tree->attr_vals) {
        parent_entropy += this->entropy_attr(attr);
    }

    for (auto &child: children) {
        if (child == best) continue;
        double p_of_child = (1.0 * child->count) / this->count;
        concept_entropy -= p_of_child * log(p_of_child);
        for (auto &[attr, val_set]: this->tree->attr_vals) {
            children_entropy += p_of_child * child->entropy_attr(attr);
        }
    }

    for (auto &child: best->children) {
        double p_of_child = (1.0 * child->count) / this->count;
        concept_entropy -= p_of_child * log(p_of_child);
        for (auto &[attr, val_set]: this->tree->attr_vals) {
            children_entropy += p_of_child * child->entropy_attr(attr);
        }
    }

    double obj = (parent_entropy - children_entropy);
    if (this->tree->mutual_info == 1){
        obj /= parent_entropy;
    }
    else if (this->tree->mutual_info == 2){
        obj /= (children_entropy + concept_entropy);
    }
    if (this->tree->children_norm){
        obj /= (this->children.size() - 1 + best->children.size());
    }
    return obj;
    */
}

inline bool MultinomialCobwebNode::is_exact_match(const AV_COUNT_TYPE &instance) {
    std::unordered_set<ATTR_TYPE> all_attrs;
    for (auto &[attr, tmp]: instance) all_attrs.insert(attr);
    for (auto &[attr, tmp]: this->av_counts) all_attrs.insert(attr);

    for (auto &attr: all_attrs) {
        if (attr.is_hidden()) continue;
        if (instance.count(attr) && !this->av_counts.count(attr)) {
            return false;
        }
        if (this->av_counts.count(attr) && !instance.count(attr)) {
            return false;
        }
        if (this->av_counts.count(attr) && instance.count(attr)) {
            double instance_attr_count = 0.0;
            std::unordered_set<VALUE_TYPE> all_vals;
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
                double concept_prob = (1.0 * this->av_counts.at(attr).at(val)) / this->a_counts.at(attr);

                if (abs(instance_prob - concept_prob) > 0.00001){
                    return false;
                }
            }
        }
    }
    return true;
}

inline size_t MultinomialCobwebNode::_hash() {
    return std::hash<uintptr_t>()(reinterpret_cast<uintptr_t>(this));
}

inline std::string MultinomialCobwebNode::__str__(){
    return this->pretty_print();
}

inline std::string MultinomialCobwebNode::concept_hash(){
    return std::to_string(this->_hash());
}

inline std::string MultinomialCobwebNode::pretty_print(int depth) {
    std::string ret = repeat("\t", depth) + "|-" + avcounts_to_json() + "\n";

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
    while (temp != nullptr) {
        if (temp == this) {
            return true;
        }
        try {
            temp = temp->parent;
        } catch (std::string e) {
            std::cout << temp;
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

inline std::string MultinomialCobwebNode::avcounts_to_json() {
    std::string ret = "{";

    // // ret += "\"_expected_guesses\": {\n";
    // ret += "\"_entropy\": {\n";
    // ret += "\"#ContinuousValue#\": {\n";
    // ret += "\"mean\": " + std::to_string(this->entropy()) + ",\n";
    // ret += "\"std\": 1,\n";
    // ret += "\"n\": 1,\n";
    // ret += "}},\n";
    
    ret += "\"_category_utility\": {\n";
    ret += "\"#ContinuousValue#\": {\n";
    ret += "\"mean\": " + std::to_string(this->category_utility()) + ",\n";
    ret += "\"std\": 1,\n";
    ret += "\"n\": 1,\n";
    ret += "}},\n";

    ret += "\"_partition_utility\": {\n";
    ret += "\"#ContinuousValue#\": {\n";
    ret += "\"mean\": " + std::to_string(this->partition_utility()) + ",\n";
    ret += "\"std\": 1,\n";
    ret += "\"n\": 1,\n";
    ret += "}},\n";

    int c = 0;
    for (auto &[attr, vAttr]: av_counts) {
        ret += "\"" + attr.get_string() + "\": {";
        int inner_count = 0;
        for (auto &[val, cnt]: vAttr) {
            ret += "\"" + val.get_string() + "\": " + doubleToString(cnt);
                // std::to_string(cnt);
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

inline std::string MultinomialCobwebNode::ser_avcounts() {
    std::string ret = "{";

    int c = 0;
    for (auto &[attr, vAttr]: av_counts) {
        ret += "\"" + attr.get_string() + "\": {";
        int inner_count = 0;
        for (auto &[val, cnt]: vAttr) {
            ret += "\"" + val.get_string() + "\": " + doubleToString(cnt);
                // std::to_string(cnt);
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

inline std::string MultinomialCobwebNode::a_counts_to_json() {
    std::string ret = "{";

    bool first = true;
    for (auto &[attr, cnt]: this->a_counts) {
        if (!first) ret += ",\n";
        else first = false;
        ret += "\"" + attr.get_string() + "\": " + doubleToString(cnt);
            // std::to_string(cnt);
    }

    ret += "}";
    return ret;
}

inline std::string MultinomialCobwebNode::dump_json() {
    std::string output = "{";

    // output += "\"concept_id\": " + std::to_string(this->_hash()) + ",\n";
    output += "\"count\": " + doubleToString(this->count) + ",\n";
    output += "\"a_counts\": " + this->a_counts_to_json() + ",\n";
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

inline std::string MultinomialCobwebNode::output_json(){
    std::string output = "{";

    output += "\"name\": \"Concept" + std::to_string(this->_hash()) + "\",\n";
    output += "\"size\": " + std::to_string(this->count) + ",\n";
    output += "\"children\": [\n";
    bool first = true;
    for (auto &c: children) {
        if(!first) output += ",";
        else first = false;
        output += c->output_json();
    }
    output += "],\n";

    output += "\"counts\": " + this->avcounts_to_json() + ",\n";
    output += "\"attr_counts\": " + this->a_counts_to_json() + "\n";

    output += "}\n";

    return output;
}

// TODO 
// TODO This should use the path prob, not the node prob.
// TODO
inline std::unordered_map<std::string, std::unordered_map<std::string, double>> MultinomialCobwebNode::predict_weighted_leaves_probs(INSTANCE_TYPE instance){

    AV_COUNT_TYPE cached_instance;
    for (auto &[attr, val_map]: instance) {
        for (auto &[val, cnt]: val_map) {
            cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
        }
    }

    double concept_weights = 0.0;
    std::unordered_map<std::string, std::unordered_map<std::string, double>> out;

    // std::cout << std::endl << "Prob of nodes along path (starting with leaf)" << std::endl;
    auto curr = this;
    while (curr->parent != nullptr) {
        auto prev = curr;
        curr = curr->parent;

        for (auto &child: curr->children) {
            if (child == prev) continue;
            double c_prob = exp(child->log_prob_class_given_instance(cached_instance, true));
            // double c_prob = 1.0;
            // std::cout << c_prob << std::endl;
            concept_weights += c_prob;

            for (auto &[attr, val_set]: this->tree->attr_vals) {
                // std::cout << attr << std::endl;
                int num_vals = this->tree->attr_vals.at(attr).size();
                float alpha = this->tree->alpha;
                COUNT_TYPE attr_count = 0;

                if (child->a_counts.count(attr)){
                    attr_count = child->a_counts.at(attr);
                }

                for (auto val: val_set) {
                    // std::cout << val << std::endl;
                    COUNT_TYPE av_counts = 0;
                    if (child->av_counts.count(attr) and child->av_counts.at(attr).count(val)){
                        av_counts = child->av_counts.at(attr).at(val);
                    }

                    double p = ((av_counts + alpha) / (attr_count + num_vals * alpha));
                    // std::cout << p << std::endl;
                    // if (attr.get_string() == "class"){
                    //     std::cout << val.get_string() << ", " << c_prob << ", " << p << ", " << p * c_prob << " :: ";
                    // }
                    out[attr.get_string()][val.get_string()] += p * c_prob;
                }
            }
        }
        // std::cout << std::endl;

    }

    for (auto &[attr, val_set]: this->tree->attr_vals) {
        for (auto val: val_set) {
            out[attr.get_string()][val.get_string()] /= concept_weights;
        }
    }

    return out;
}

inline std::unordered_map<std::string, std::unordered_map<std::string, double>> MultinomialCobwebNode::predict_weighted_probs(INSTANCE_TYPE instance){

    AV_COUNT_TYPE cached_instance;
    for (auto &[attr, val_map]: instance) {
        for (auto &[val, cnt]: val_map) {
            cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
        }
    }

    double concept_weights = 0.0;
    std::unordered_map<std::string, std::unordered_map<std::string, double>> out;

    // std::cout << std::endl << "Prob of nodes along path (starting with leaf)" << std::endl;
    auto curr = this;
    while (curr != nullptr) {
        double c_prob = exp(curr->log_prob_class_given_instance(cached_instance, true));
        // double c_prob = 1.0;
        // std::cout << c_prob << std::endl;
        concept_weights += c_prob;

        for (auto &[attr, val_set]: this->tree->attr_vals) {
            // std::cout << attr << std::endl;
            int num_vals = this->tree->attr_vals.at(attr).size();
            float alpha = this->tree->alpha;
            COUNT_TYPE attr_count = 0;

            if (curr->a_counts.count(attr)){
                attr_count = curr->a_counts.at(attr);
            }

            for (auto val: val_set) {
                // std::cout << val << std::endl;
                COUNT_TYPE av_counts = 0;
                if (curr->av_counts.count(attr) and curr->av_counts.at(attr).count(val)){
                    av_counts = curr->av_counts.at(attr).at(val);
                }

                double p = ((av_counts + alpha) / (attr_count + num_vals * alpha));
                // std::cout << p << std::endl;
                // if (attr.get_string() == "class"){
                //     std::cout << val.get_string() << ", " << c_prob << ", " << p << ", " << p * c_prob << " :: ";
                // }
                out[attr.get_string()][val.get_string()] += p * c_prob;
            }
        }
        // std::cout << std::endl;

        curr = curr->parent;
    }

    for (auto &[attr, val_set]: this->tree->attr_vals) {
        for (auto val: val_set) {
            out[attr.get_string()][val.get_string()] /= concept_weights;
        }
    }

    return out;
}

inline std::unordered_map<std::string, std::unordered_map<std::string, double>> MultinomialCobwebNode::predict_probs(){
    std::unordered_map<std::string, std::unordered_map<std::string, double>> out;
    for (auto &[attr, val_set]: this->tree->attr_vals) {
        // std::cout << attr << std::endl;
        int num_vals = this->tree->attr_vals.at(attr).size();
        float alpha = this->tree->alpha;
        COUNT_TYPE attr_count = 0;

        if (this->a_counts.count(attr)){
            attr_count = this->a_counts.at(attr);
        }

        for (auto val: val_set) {
            // std::cout << val << std::endl;
            COUNT_TYPE av_counts = 0;
            if (this->av_counts.count(attr) and this->av_counts.at(attr).count(val)){
                av_counts = this->av_counts.at(attr).at(val);
            }

            double p = ((av_counts + alpha) / (attr_count + num_vals * alpha));
            // std::cout << p << std::endl;
            out[attr.get_string()][val.get_string()] += p;
        }
    }

    return out;
}

inline std::vector<std::tuple<VALUE_TYPE, double>> MultinomialCobwebNode::get_weighted_values(
        ATTR_TYPE attr, bool allowNone) {

    std::vector<std::tuple<VALUE_TYPE, double>> choices;
    if (!this->av_counts.count(attr)) {
        choices.push_back(std::make_tuple(NULL_STRING, 1.0));
    }
    double valCount = 0;
    for (auto &[val, tmp]: this->av_counts.at(attr)) {
        COUNT_TYPE count = this->av_counts.at(attr).at(val);
        choices.push_back(std::make_tuple(val, (1.0 * count) / this->count));
        valCount += count;
    }
    if (allowNone) {
        choices.push_back(std::make_tuple(NULL_STRING, ((1.0 * (this->count - valCount)) / this->count)));
    }
    return choices;
}

inline double MultinomialCobwebNode::cu_given_instance(const AV_COUNT_TYPE &instance, double p_of_c){
    double score = 0.0;

    for (auto &[attr, vAttr]: this->tree->root->av_counts) {
        // if (!(attr.get_string() == "Class")) continue;
        // if (instance.count(attr)) continue;
        if (attr.is_hidden()) continue;

        double ratio = 1.0;
        if (this->tree->weight_attr){
            ratio = (1.0 * this->tree->root->a_counts.at(attr)) / (this->tree->root->count);
            // ratio = (1.0 * attr_count) / this->count;
        }

        int num_vals = this->tree->attr_vals.at(attr).size();
        double n_a = this->tree->root->a_counts.at(attr) + num_vals * this->tree->alpha;
        double n_a_given_c = num_vals * this->tree->alpha;

        if (this->a_counts.count(attr)){
            n_a_given_c += this->a_counts.at(attr);
        }

        for (auto &[val, cnt]: vAttr) {
            double n_av = cnt + this->tree->alpha;
            double n_av_given_c = this->tree->alpha;

            if (this->av_counts.count(attr) and this->av_counts.at(attr).count(val)){
                n_av_given_c += this->av_counts.at(attr).at(val);
            }

            double p_of_av = n_av / n_a;
            double p_of_av_given_c = n_av_given_c / n_a_given_c;
            score += ratio * p_of_c * p_of_av_given_c * (log(p_of_av_given_c) - log(p_of_av));
            // score += ratio * p_of_c * (p_of_av_given_c * log(p_of_av_given_c) - p_of_av * log(p_of_av));
        }
    }

    return score;
}

inline double MultinomialCobwebNode::category_utility(){
    double score = 0.0;

    double p_of_c = (this->count * 1.0) / (this->tree->root->count);

    for (auto &[attr, vAttr]: this->tree->root->av_counts) {
        double ratio = 1.0;
        if (this->tree->weight_attr){
            ratio = (1.0 * this->tree->root->a_counts.at(attr)) / (this->tree->root->count);
            // ratio = (1.0 * attr_count) / this->count;
        }
        
        int num_vals = this->tree->attr_vals.at(attr).size();
        double n_a = this->tree->root->a_counts.at(attr) + num_vals * this->tree->alpha;
        double n_a_given_c = num_vals * this->tree->alpha;

        if (this->a_counts.count(attr)){
            n_a_given_c += this->a_counts.at(attr);
        }

        for (auto &[val, cnt]: vAttr) {
            double n_av = cnt + this->tree->alpha;
            double n_av_given_c = this->tree->alpha;

            if (this->av_counts.count(attr) and this->av_counts.at(attr).count(val)){
                n_av_given_c += this->av_counts.at(attr).at(val);
            }

            double p_of_av = n_av / n_a;
            double p_of_av_given_c = n_av_given_c / n_a_given_c;
            score += ratio * p_of_c * p_of_av_given_c * (log(p_of_av_given_c) - log(p_of_av));
            // score += ratio * p_of_c * (p_of_av_given_c * log(p_of_av_given_c) - p_of_av * log(p_of_av));
        }
    }

    return score;
}

inline std::vector<double> MultinomialCobwebNode::prob_children_given_instance_ext(INSTANCE_TYPE instance){
    AV_COUNT_TYPE cached_instance;
    for (auto &[attr, val_map]: instance) {
        for (auto &[val, cnt]: val_map) {
            cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
        }
    }

    return this->prob_children_given_instance(cached_instance);
}

inline std::vector<double> MultinomialCobwebNode::prob_children_given_instance(const AV_COUNT_TYPE &instance){

    double sum_probs = 0;
    std::vector<double> raw_probs = std::vector<double>();
    std::vector<double> norm_probs = std::vector<double>();

    for (auto &child: this->children){
        double p = exp(child->log_prob_class_given_instance(instance, false));
        sum_probs += p;
        raw_probs.push_back(p);
    }

    for (auto p: raw_probs){
        norm_probs.push_back(p/sum_probs);
    }

    return norm_probs;

}

inline double MultinomialCobwebNode::log_prob_class_given_instance_ext(INSTANCE_TYPE instance, bool use_root_counts){
    AV_COUNT_TYPE cached_instance;
    for (auto &[attr, val_map]: instance) {
        for (auto &[val, cnt]: val_map) {
            cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
        }
    }

    return this->log_prob_class_given_instance(cached_instance, use_root_counts);
}

inline double MultinomialCobwebNode::log_prob_class_given_instance(const AV_COUNT_TYPE &instance, bool use_root_counts){

    double log_prob = log_prob_instance(instance);

    if (use_root_counts){
        log_prob += log((1.0 * this->count) / this->tree->root->count);
    }
    else{
        log_prob += log((1.0 * this->count) / this->parent->count);
    }

    // std::cout << "LOB PROB" << std::to_string(log_prob) << std::endl;

    return log_prob;
}

inline double MultinomialCobwebNode::log_prob_instance_ext(INSTANCE_TYPE instance){
    AV_COUNT_TYPE cached_instance;
    for (auto &[attr, val_map]: instance) {
        for (auto &[val, cnt]: val_map) {
            cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
        }
    }

    return this->log_prob_instance(cached_instance);
}

inline double MultinomialCobwebNode::log_prob_instance(const AV_COUNT_TYPE &instance){

    double log_prob = 0;


    for (auto &[attr, vAttr]: instance) {
        bool hidden = attr.is_hidden();
        if (hidden || !this->tree->attr_vals.count(attr)){
            continue;
        }

        double num_vals = this->tree->attr_vals.at(attr).size();

        for (auto &[val, cnt]: vAttr){
            if (!this->tree->attr_vals.at(attr).count(val)){
                continue;
            }

            double alpha = this->tree->alpha;
            double av_counts = alpha;
            if (this->av_counts.count(attr) && this->av_counts.at(attr).count(val)){
                av_counts += this->av_counts.at(attr).at(val);
            }

            // a_counts starts with the alphas over all values (even vals not in
            // current node)
            COUNT_TYPE a_counts = num_vals * alpha;
            if (this->a_counts.count(attr)){
                a_counts += this->a_counts.at(attr);
            }

            // we use cnt here to weight accuracy by counts in the training
            // instance. Usually this is 1, but in multinomial models, it might
            // be something else.
            log_prob += cnt * (log(av_counts) - log(a_counts));

        }

    }

    return log_prob;
}

PYBIND11_MODULE(multinomial_cobweb, m) {
    m.doc() = "concept_formation.multinomial_cobweb plugin"; // optional module docstring

    py::class_<MultinomialCobwebNode>(m, "MultinomialCobwebNode")
        .def(py::init<>())
        .def("pretty_print", &MultinomialCobwebNode::pretty_print)
        .def("output_json", &MultinomialCobwebNode::output_json)
        .def("predict_probs", &MultinomialCobwebNode::predict_probs)
        .def("predict_weighted_probs", &MultinomialCobwebNode::predict_weighted_probs)
        .def("predict_weighted_leaves_probs", &MultinomialCobwebNode::predict_weighted_leaves_probs)
        .def("get_best_level", &MultinomialCobwebNode::get_best_level, py::return_value_policy::reference)
        .def("get_basic_level", &MultinomialCobwebNode::get_basic_level, py::return_value_policy::reference)
        .def("log_prob_class_given_instance", &MultinomialCobwebNode::log_prob_class_given_instance_ext) 
        .def("log_prob_instance", &MultinomialCobwebNode::log_prob_instance_ext) 
        .def("prob_children_given_instance", &MultinomialCobwebNode::prob_children_given_instance_ext)
        .def("entropy", &MultinomialCobwebNode::entropy)
        .def("category_utility", &MultinomialCobwebNode::category_utility)
        .def("partition_utility", &MultinomialCobwebNode::partition_utility)
        .def("__str__", &MultinomialCobwebNode::__str__)
        .def("concept_hash", &MultinomialCobwebNode::concept_hash)
        .def_readonly("count", &MultinomialCobwebNode::count)
        .def_readonly("children", &MultinomialCobwebNode::children, py::return_value_policy::reference)
        .def_readonly("parent", &MultinomialCobwebNode::parent, py::return_value_policy::reference)
        .def_readonly("av_counts", &MultinomialCobwebNode::av_counts, py::return_value_policy::reference)
        .def_readonly("a_counts", &MultinomialCobwebNode::a_counts, py::return_value_policy::reference)
        .def_readonly("tree", &MultinomialCobwebNode::tree, py::return_value_policy::reference);

    py::class_<MultinomialCobwebTree>(m, "MultinomialCobwebTree")
        .def(py::init<float, bool>(),
                py::arg("alpha") = 1.0,
                py::arg("weight_attr") = false)
        .def("ifit", &MultinomialCobwebTree::ifit, py::return_value_policy::reference)
        .def("fit", &MultinomialCobwebTree::fit,
                py::arg("instances") = std::vector<AV_COUNT_TYPE>(),
                py::arg("iterations") = 1,
                py::arg("randomizeFirst") = true)
        .def("categorize", &MultinomialCobwebTree::categorize,
                py::arg("instance") = std::vector<AV_COUNT_TYPE>(),
                // py::arg("get_best_concept") = false,
                py::return_value_policy::reference)
        .def("predict_probs_mixture", &MultinomialCobwebTree::predict_probs_mixture)
        .def("clear", &MultinomialCobwebTree::clear)
        .def("__str__", &MultinomialCobwebTree::__str__)
        .def("dump_json", &MultinomialCobwebTree::dump_json)
        .def("load_json", &MultinomialCobwebTree::load_json)
        .def_readonly("root", &MultinomialCobwebTree::root, py::return_value_policy::reference);
}
