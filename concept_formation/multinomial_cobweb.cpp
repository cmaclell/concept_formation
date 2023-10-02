#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
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
typedef float COUNT_TYPE;
typedef std::unordered_map<std::string, std::unordered_map<std::string, COUNT_TYPE>> INSTANCE_TYPE;
typedef std::unordered_map<VALUE_TYPE, COUNT_TYPE> VAL_COUNT_TYPE;
typedef std::unordered_map<ATTR_TYPE, VAL_COUNT_TYPE> AV_COUNT_TYPE;
typedef std::unordered_map<ATTR_TYPE, std::unordered_set<VALUE_TYPE>> AV_KEY_TYPE;
typedef std::unordered_map<ATTR_TYPE, COUNT_TYPE> VAL_COUNTS_TYPE;
typedef std::pair<double, int> OPERATION_TYPE;

class MultinomialCobwebTree;
class MultinomialCobwebNode;

std::random_device rd;
std::mt19937_64 gen(rd());
std::uniform_real_distribution<double> unif(0, 1);

std::unordered_map<int, double> lgammaCache;
std::unordered_map<int, std::unordered_map<int, int>> binomialCache;
std::unordered_map<int, std::unordered_map<double, double>> entropy_k_cache;

double lgamma_cached(int n){
    auto it = lgammaCache.find(n);
    if (it != lgammaCache.end()) return it->second;

    double result = std::lgamma(n);
    lgammaCache[n] = result;
    return result;

}

int nChoosek(int n, int k) {
    if (k > n) return 0;
    if (k * 2 > n) k = n - k;
    if (k == 0) return 1;

    // Check if the value is in the cache
    auto it_n = binomialCache.find(n);
    if (it_n != binomialCache.end()){
        auto it_k = it_n->second.find(k);
        if (it_k != it_n->second.end()) return it_k->second;
    }

    int result = n;
    for (int i = 2; i <= k; ++i) {
        result *= (n - i + 1);
        result /= i;
    }

    // Store the computed value in the cache
    binomialCache[n][k] = result;
    // std::cout << n << "!" << k << " : " << result << std::endl;

    return result;
}

double entropy_component_k(int n, double p){
    if (p == 0.0 || p == 1.0){
        return 0.0;
    }

    auto it_n = entropy_k_cache.find(n);
    if (it_n != entropy_k_cache.end()){
        auto it_p = it_n->second.find(p);
        if (it_p != it_n->second.end()) return it_p->second;
    }

    double precision = 1e-10;
    double info = -n * p * log(p);

    // This is where we'll see the highest entropy
    int mid = std::ceil(n * p);

    for (int xi = mid; xi > 2; xi--){
        double v = nChoosek(n, xi) * std::pow(p, xi) * std::pow((1-p), (n-xi)) * lgamma_cached(xi+1);
        if (v < precision) break;
        info += v;
    }

    for (int xi = mid+1; xi <= n; xi++){
        double v = nChoosek(n, xi) * std::pow(p, xi) * std::pow((1-p), (n-xi)) * lgamma_cached(xi+1);
        if (v < precision) break;
        info += v;
    }

    entropy_k_cache[n][p] = info;

    return info;
}

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

VALUE_TYPE most_likely_choice(std::vector<std::tuple<VALUE_TYPE, double>> choices) {
    std::vector<std::tuple<double, double, VALUE_TYPE>> vals;

    for (auto &[val, prob]: choices){
        if (prob < 0){
            std::cout << "most_likely_choice: all weights must be greater than or equal to 0" << std::endl;
        }
        vals.push_back(std::make_tuple(prob, custom_rand(), val));
    }
    sort(vals.rbegin(), vals.rend());

    return std::get<2>(vals[0]);
}

VALUE_TYPE weighted_choice(std::vector<std::tuple<VALUE_TYPE, double>> choices) {
    std::cout << "weighted_choice: Not implemented yet" << std::endl;
    return std::get<0>(choices[0]);
}


class MultinomialCobwebNode {

    public:
        COUNT_TYPE count;
        std::unordered_map<ATTR_TYPE, COUNT_TYPE> attr_counts;
        std::vector<MultinomialCobwebNode *> children;
        MultinomialCobwebNode *parent;
        MultinomialCobwebTree *tree;
        AV_COUNT_TYPE av_counts;
        MultinomialCobwebNode();
        MultinomialCobwebNode(MultinomialCobwebNode *otherNode);

        void increment_counts(const AV_COUNT_TYPE &instance);
        void update_counts_from_node(MultinomialCobwebNode *node);
        double score_insert(const AV_COUNT_TYPE &instance);
        double score_merge(MultinomialCobwebNode *other, const AV_COUNT_TYPE
                &instance);
        MultinomialCobwebNode* get_best_level(INSTANCE_TYPE instance);
        MultinomialCobwebNode* get_basic_level();
        double category_utility();
        double score();
        double partition_utility();
        std::tuple<double, int> get_best_operation(const AV_COUNT_TYPE
                &instance, MultinomialCobwebNode *best1, MultinomialCobwebNode
                *best2, double best1Cu);
        std::tuple<double, MultinomialCobwebNode *, MultinomialCobwebNode *>
            two_best_children(const AV_COUNT_TYPE &instance);
        double log_prob_class_given_instance(const AV_COUNT_TYPE &instance,
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
        std::string attr_counts_to_json();
        std::string dump_json();
        std::string output_json();
        std::vector<std::tuple<VALUE_TYPE, double>>
            get_weighted_values(ATTR_TYPE attr, bool allowNone = true);
        std::unordered_map<std::string, std::unordered_map<std::string, double>> predict_probs();
        VALUE_TYPE predict(ATTR_TYPE attr, std::string choiceFn = "most likely",
                bool allowNone = true);
        double probability(ATTR_TYPE attr, VALUE_TYPE val);

};


class MultinomialCobwebTree {

    public:
        bool use_mutual_info;
        float alpha_weight;
        bool dynamic_alpha;
        bool weight_attr;
        MultinomialCobwebNode *root;
        AV_KEY_TYPE attr_vals;

        MultinomialCobwebTree(bool use_mutual_info, float alpha_weight, bool
                dynamic_alpha, bool weight_attr) {
            this->use_mutual_info = use_mutual_info;
            this->alpha_weight = alpha_weight;
            this->dynamic_alpha = dynamic_alpha;
            this->weight_attr = weight_attr;

            this->root = new MultinomialCobwebNode();
            this->root->tree = this;
            this->attr_vals = AV_KEY_TYPE();
        }

        std::string __str__(){
            return this->root->__str__();
        }

        float alpha(int n_vals){
            if (!this->dynamic_alpha){
                return this->alpha_weight;
            }

            if (n_vals == 0){
                return this->alpha_weight;
            } else {
                return this->alpha_weight / n_vals;
            }
        }

        MultinomialCobwebNode* load_json_helper(json_object_s* object) {
            MultinomialCobwebNode *new_node = new MultinomialCobwebNode();
            new_node->tree = this;

            // // Get concept_id
            struct json_object_element_s* concept_id_obj = object->start;
            // unsigned long long concept_id_val = stoull(json_value_as_number(concept_id_obj->value)->number);
            // new_node->concept_id = concept_id_val;
            // new_node->update_counter(concept_id_val);

            // Get count
            struct json_object_element_s* count_obj = concept_id_obj->next;
            // struct json_object_element_s* count_obj = object->start;
            float count_val = atof(json_value_as_number(count_obj->value)->number);
            new_node->count = count_val;

            // Get attr_counts
            struct json_object_element_s* attr_counts_obj = count_obj->next;
            struct json_object_s* attr_counts_dict = json_value_as_object(attr_counts_obj->value);
            struct json_object_element_s* attr_counts_cursor = attr_counts_dict->start;
            while(attr_counts_cursor != NULL) {
                // Get attr name
                std::string attr_name = std::string(attr_counts_cursor->name->string);

                // A count is stored with each attribute
                float count_value = atof(json_value_as_number(attr_counts_cursor->value)->number);
                new_node->attr_counts[attr_name] = count_value;

                attr_counts_cursor = attr_counts_cursor->next;
            }

            // Get av counts
            struct json_object_element_s* av_counts_obj = attr_counts_obj->next;
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
                    float attr_val_count = atof(json_value_as_number(inner_counts_cursor->value)->number);

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
            return this->root->dump_json();
        }

        void load_json(std::string json) {
            struct json_value_s* root = json_parse(json.c_str(), strlen(json.c_str()));
            struct json_object_s* object = (struct json_object_s*)root->payload;
            delete this->root;
            this->root = this->load_json_helper(object);

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

};

inline MultinomialCobwebNode::MultinomialCobwebNode() {
    count = 0;
    attr_counts = std::unordered_map<ATTR_TYPE, COUNT_TYPE>();
    parent = nullptr;
    tree = nullptr;
}

inline MultinomialCobwebNode::MultinomialCobwebNode(MultinomialCobwebNode *otherNode) {
    count = 0;
    attr_counts = std::unordered_map<ATTR_TYPE, COUNT_TYPE>();

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

inline double MultinomialCobwebNode::score_insert(const AV_COUNT_TYPE &instance){

    double info = 0.0;

    for (auto &[attr, av_inner]: this->av_counts){
        if (attr.is_hidden()) continue;
        bool instance_has_attr = instance.count(attr);

        int num_vals = this->tree->attr_vals.at(attr).size();
        float alpha = this->tree->alpha(num_vals);

        COUNT_TYPE attr_count = this->attr_counts.at(attr);
        if (instance_has_attr){
            for (auto &[val, cnt]: instance.at(attr)){
                attr_count += cnt;
            }
        }

        double ratio = 1.0;

        if (this->tree->weight_attr){
            ratio = (1.0 * attr_count) / (this->count + 1);
        }

        int n = std::ceil(ratio);
        if (this->tree->use_mutual_info) info -= lgamma_cached(n+1);

        int vals_processed = 0;
        for (auto &[val, cnt]: av_inner){
            vals_processed += 1;
            COUNT_TYPE av_count = cnt;
            if (instance_has_attr and instance.at(attr).count(val)){
                av_count += instance.at(attr).at(val);
            }

            double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
            if (this->tree->use_mutual_info){
                info += entropy_component_k(n, p);
            } else{
                info += ratio * -p * p;
            }
        }
        if (instance_has_attr){
            for (auto &[val, cnt]: instance.at(attr)){
                if (av_inner.count(val)) continue;
                vals_processed += 1;
                COUNT_TYPE av_count = cnt;

                double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
                if (this->tree->use_mutual_info){
                    info += entropy_component_k(n, p);
                } else{
                    info += ratio * -p * p;
                }
            }
        }

        COUNT_TYPE num_missing = num_vals - vals_processed;
        if (num_missing > 0 and alpha > 0){
            double p = (alpha / (attr_count + num_vals * alpha));
            if (this->tree->use_mutual_info){
                info += num_missing * entropy_component_k(n, p);
            } else {
                info += num_missing * ratio * -p * p;
            }
        }
    }

    // iterate over attr in instance not in av_counts
    for (auto &[attr, av_inner]: instance){
        if (attr.is_hidden()) continue;
        if (this->av_counts.count(attr)) continue;
        COUNT_TYPE attr_count = 0;
        int num_vals = this->tree->attr_vals.at(attr).size();
        float alpha = this->tree->alpha(num_vals);
        for (auto &[val, cnt]: av_inner){
            attr_count += cnt;
        }

        double ratio = 1.0;

        if (this->tree->weight_attr){
            ratio = (1.0 * attr_count) / (this->count + 1);
        }
        int n = std::ceil(ratio);
        if (this->tree->use_mutual_info) info -= lgamma_cached(n+1);

        int vals_processed = 0;
        for (auto &[val, av_count]: av_inner){
            vals_processed += 1;
            double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
            if (this->tree->use_mutual_info){
                info += entropy_component_k(n, p);
            } else{
                info += ratio * -p * p;
            }
        }

        COUNT_TYPE num_missing = num_vals - vals_processed;
        if (num_missing > 0 and alpha > 0){
            double p = (alpha / (attr_count + num_vals * alpha));
            if (this->tree->use_mutual_info){
                info += num_missing * entropy_component_k(n, p);
            } else {
                info += num_missing * ratio * -p * p;
            }
        }
    }

    return info;
}

inline double MultinomialCobwebNode::score_merge(MultinomialCobwebNode *other,
        const AV_COUNT_TYPE &instance) {

    double info = 0.0;

    for (auto &[attr, inner_vals]: this->av_counts){
        if (attr.is_hidden()) continue;
        int num_vals = this->tree->attr_vals.at(attr).size();
        float alpha = this->tree->alpha(num_vals);

        bool other_has_attr = other->av_counts.count(attr);
        bool instance_has_attr = instance.count(attr);

        COUNT_TYPE attr_count = this->attr_counts.at(attr);

        if (other_has_attr){
            attr_count += other->attr_counts.at(attr);
        }
        if (instance_has_attr){
            for (auto &[val, cnt]: instance.at(attr)){
                attr_count += cnt;
            }
        }

        double ratio = 1.0;

        if (this->tree->weight_attr){
            ratio = (1.0 * attr_count) / (this->count + other->count + 1);
        }

        int n = std::ceil(ratio);
        if (this->tree->use_mutual_info) info -= lgamma_cached(n+1);

        int vals_processed = 0;
        for (auto &[val, cnt]: inner_vals){
            vals_processed += 1;
            COUNT_TYPE av_count = cnt;

            if (other->av_counts.count(attr) and other->av_counts.at(attr).count(val)){
                av_count += other->av_counts.at(attr).at(val);
            }
            if (instance.count(attr) and instance.at(attr).count(val)){
                av_count += instance.at(attr).at(val);
            }

            double p = ((av_count + alpha) / (attr_count + num_vals * alpha));

            if (this->tree->use_mutual_info){
                info += entropy_component_k(n, p);
            } else {
                info += ratio * -p * p;
            }
        }

        if (other_has_attr){
            for (auto &[val, cnt]: other->av_counts.at(attr)){
                if (inner_vals.count(val)) continue;
                vals_processed += 1;
                COUNT_TYPE av_count = cnt;

                if (instance.count(attr) and instance.at(attr).count(val)){
                    av_count += instance.at(attr).at(val);
                }

                double p = ((av_count + alpha) / (attr_count + num_vals * alpha));

                if (this->tree->use_mutual_info){
                    info += entropy_component_k(n, p);
                } else {
                    info += ratio * -p * p;
                }
            }
        }

        if (instance_has_attr){
            for (auto &[val, cnt]: instance.at(attr)){
                if (inner_vals.count(val)) continue;
                if (other_has_attr && other->av_counts.at(attr).count(val)) continue;
                vals_processed += 1;
                COUNT_TYPE av_count = cnt;

                double p = ((av_count + alpha) / (attr_count + num_vals * alpha));

                if (this->tree->use_mutual_info){
                    info += entropy_component_k(n, p);
                } else {
                    info += ratio * -p * p;
                }
            }
        }

        COUNT_TYPE num_missing = num_vals - vals_processed;
        if (num_missing > 0 and alpha > 0){
            double p = (alpha / (attr_count + num_vals * alpha));
            if (this->tree->use_mutual_info){
                info += num_missing * entropy_component_k(n, p);
            } else {
                info += num_missing * ratio * -p * p;
            }
        }
    }

    for (auto &[attr, inner_vals]: other->av_counts){
        if (attr.is_hidden()) continue;
        if (this->av_counts.count(attr)) continue;
        COUNT_TYPE attr_count = other->attr_counts.at(attr);
        int num_vals = this->tree->attr_vals.at(attr).size();
        float alpha = this->tree->alpha(num_vals);

        bool instance_has_attr = instance.count(attr);

        if (instance_has_attr){
            for (auto &[val, cnt]: instance.at(attr)){
                attr_count += cnt;
            }
        }

        double ratio = 1.0;

        if (this->tree->weight_attr){
            ratio = (1.0 * attr_count) / (this->count + other->count + 1);
        }

        int n = std::ceil(ratio);
        if (this->tree->use_mutual_info) info -= lgamma_cached(n+1);

        int vals_processed = 0;
        for (auto &[val, cnt]: inner_vals){
            vals_processed += 1;
            COUNT_TYPE av_count = cnt;

            if (instance_has_attr and instance.at(attr).count(val)){
                av_count += instance.at(attr).at(val);
            }

            double p = ((av_count + alpha) / (attr_count + num_vals * alpha));

            if (this->tree->use_mutual_info){
                info += entropy_component_k(n, p);
            } else {
                info += ratio * -p * p;
            }
        }

        if (instance_has_attr){
            for (auto &[val, cnt]: instance.at(attr)){
                if (inner_vals.count(val)) continue;
                vals_processed += 1;
                COUNT_TYPE av_count = cnt;

                double p = ((av_count + alpha) / (attr_count + num_vals * alpha));

                if (this->tree->use_mutual_info){
                    info += entropy_component_k(n, p);
                } else {
                    info += ratio * -p * p;
                }
            }
        }

        COUNT_TYPE num_missing = num_vals - vals_processed;
        if (num_missing > 0 and alpha > 0){
            double p = (alpha / (attr_count + num_vals * alpha));
            if (this->tree->use_mutual_info){
                info += num_missing * entropy_component_k(n, p);
            } else {
                info += num_missing * ratio * -p * p;
            }
        }
    }

    for (auto &[attr, inner_vals]: instance){
        if (attr.is_hidden()) continue;
        if (this->av_counts.count(attr)) continue;
        if (other->av_counts.count(attr)) continue;

        int num_vals = this->tree->attr_vals.at(attr).size();
        float alpha = this->tree->alpha(num_vals);

        COUNT_TYPE attr_count = 0;
        for (auto &[val, cnt]: instance.at(attr)){
            attr_count += cnt;
        }

        double ratio = 1.0;

        if (this->tree->weight_attr){
            ratio = (1.0 * attr_count) / (this->count + other->count + 1);
        }

        int n = std::ceil(ratio);
        if (this->tree->use_mutual_info) info -= lgamma_cached(n+1);

        int vals_processed = 0;
        for (auto &[val, av_count]: inner_vals){
            vals_processed += 1;

            double p = ((av_count + alpha) / (attr_count + num_vals * alpha));

            if (this->tree->use_mutual_info){
                info += entropy_component_k(n, p);
            } else {
                info += ratio * -p * p;
            }
        }

        COUNT_TYPE num_missing = num_vals - vals_processed;
        if (num_missing > 0 and alpha > 0){
            double p = (alpha / (attr_count + num_vals * alpha));
            if (this->tree->use_mutual_info){
                info += num_missing * entropy_component_k(n, p);
            } else {
                info += num_missing * ratio * -p * p;
            }
        }
    }

    return info;
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

inline double MultinomialCobwebNode::score() {

    double info = 0.0;
    for (auto &[attr, inner_av]: this->av_counts){
        if (attr.is_hidden()) continue;

        COUNT_TYPE attr_count = this->attr_counts.at(attr);
        int num_vals = this->tree->attr_vals.at(attr).size();
        float alpha = this->tree->alpha(num_vals);

        double ratio = 1.0;

        if (this->tree->weight_attr){
            ratio = (1.0 * attr_count) / this->count;
        }

        int n = std::ceil(ratio);
        if (this->tree->use_mutual_info) info -= lgamma_cached(n+1);

        for (auto &[val, cnt]: inner_av){
            double p = ((cnt + alpha) / (attr_count + num_vals * alpha));
            if (this->tree->use_mutual_info){
                info += entropy_component_k(n, p);
            } else {
                info += ratio * -p * p;
            }
        }

        COUNT_TYPE num_missing = num_vals - inner_av.size();
        if (num_missing > 0 and alpha > 0){
            double p = (alpha / (attr_count + num_vals * alpha));
            if (this->tree->use_mutual_info){
                info += num_missing * entropy_component_k(n, p);
            } else {
                info += num_missing * ratio * -p * p;
            }
        }
    }

    return info;
}

inline double MultinomialCobwebNode::partition_utility() {
    if (children.empty()) {
        return 0.0;
    }

    double children_score = 0.0;

    for (auto &child: children) {
        double p_of_child = (1.0 * child->count) / this->count;
        children_score += p_of_child * child->score();
    }

    return ((this->score() - children_score) / children.size());

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

    std::vector<std::tuple<double, double, double, MultinomialCobwebNode *>> relative_pu;
    for (auto &child: this->children) {
        relative_pu.push_back(
                std::make_tuple(
                    (child->count * child->score()) -
                    ((child->count + 1) * child->score_insert(instance)),
                    child->count,
                    custom_rand(),
                    child));
    }

    sort(relative_pu.rbegin(), relative_pu.rend());

    MultinomialCobwebNode *best1 = std::get<3>(relative_pu[0]);
    double best1_pu = pu_for_insert(best1, instance);
    MultinomialCobwebNode *best2 = relative_pu.size() > 1 ? std::get<3>(relative_pu[1]) : nullptr;
    return std::make_tuple(best1_pu, best1, best2);
}

inline double MultinomialCobwebNode::pu_for_insert(MultinomialCobwebNode *child, const AV_COUNT_TYPE &instance) {
    double children_score = 0.0;

    for (auto &c: this->children) {
        if (c == child) {
            double p_of_child = (c->count + 1.0) / (this->count + 1.0);
            children_score += p_of_child * c->score_insert(instance);
        }
        else{
            double p_of_child = (1.0 * c->count) / (this->count + 1.0);
            children_score += p_of_child * c->score();
        }
    }

    return ((this->score_insert(instance) - children_score) / this->children.size());
}

inline double MultinomialCobwebNode::pu_for_new_child(const AV_COUNT_TYPE &instance) {
    double children_score = 0.0;

    for (auto &c: this->children) {
        double p_of_child = (1.0 * c->count) / (this->count + 1.0);
        children_score += p_of_child * c->score();
    }

    // TODO modify so that we can evaluate new child without copying instance.
    MultinomialCobwebNode new_child = MultinomialCobwebNode();
    new_child.parent = this;
    new_child.tree = this->tree;
    new_child.increment_counts(instance);
    double p_of_child = 1.0 / (this->count + 1.0);
    children_score += p_of_child * new_child.score();

    return ((this->score_insert(instance) - children_score) /
            (children.size()+1));
}

inline double MultinomialCobwebNode::pu_for_merge(MultinomialCobwebNode *best1,
        MultinomialCobwebNode *best2, const AV_COUNT_TYPE &instance) {

    double children_score = 0.0;

    for (auto &c: children) {
        if (c == best1 || c == best2){
            continue;
        }

        double p_of_child = (1.0 * c->count) / (this->count + 1.0);
        children_score += p_of_child * c->score();
    }

    double p_of_child = (best1->count + best2->count + 1.0) / (this->count + 1.0);
    children_score += p_of_child * best1->score_merge(best2, instance);

    return ((this->score_insert(instance) - children_score) / (children.size()-1));
}

inline double MultinomialCobwebNode::pu_for_split(MultinomialCobwebNode *best){
    double children_score = 0.0;

    for (auto &c: children) {
        if (c == best) continue;

        double p_of_child = (1.0 * c->count) / this->count;
        children_score += p_of_child * c->score();
    }

    for (auto &c: best->children) {
        double p_of_child = (1.0 * c->count) / this->count;
        children_score += p_of_child * c->score();
    }

    double pu = ((this->score() - children_score) / (children.size() - 1 + best->children.size()));

    return pu;
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
                double concept_prob = (1.0 * this->av_counts.at(attr).at(val)) / this->attr_counts.at(attr);

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
    // ret += "\"mean\": " + std::to_string(this->score()) + ",\n";
    // ret += "\"std\": 1,\n";
    // ret += "\"n\": 1,\n";
    // ret += "}},\n";

    // ret += "\"_mutual_info\": {\n";
    // ret += "\"#ContinuousValue#\": {\n";
    // ret += "\"mean\": " + std::to_string(this->mutual_information()) + ",\n";
    // ret += "\"std\": 1,\n";
    // ret += "\"n\": 1,\n";
    // ret += "}},\n";

    int c = 0;
    for (auto &[attr, vAttr]: av_counts) {
        ret += "\"" + attr.get_string() + "\": {";
        int inner_count = 0;
        for (auto &[val, cnt]: vAttr) {
            ret += "\"" + val.get_string() + "\": " + std::to_string(cnt);
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
            ret += "\"" + val.get_string() + "\": " + std::to_string(cnt);
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

inline std::string MultinomialCobwebNode::attr_counts_to_json() {
    std::string ret = "{";

    bool first = true;
    for (auto &[attr, cnt]: this->attr_counts) {
        if (!first) ret += ",\n";
        else first = false;
        ret += "\"" + attr.get_string() + "\": " + std::to_string(cnt);
    }

    ret += "}";
    return ret;
}

inline std::string MultinomialCobwebNode::dump_json() {
    std::string output = "{";

    output += "\"concept_id\": " + std::to_string(this->_hash()) + ",\n";
    output += "\"count\": " + std::to_string(this->count) + ",\n";
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
    output += "\"attr_counts\": " + this->attr_counts_to_json() + "\n";

    output += "}\n";

    return output;
}

inline std::unordered_map<std::string, std::unordered_map<std::string, double>> MultinomialCobwebNode::predict_probs(){
    std::unordered_map<std::string, std::unordered_map<std::string, double>> out;
    for (auto &[attr, val_set]: this->tree->attr_vals) {
        // std::cout << attr << std::endl;
        int num_vals = this->tree->attr_vals.at(attr).size();
        float alpha = this->tree->alpha(num_vals);
        COUNT_TYPE attr_count = 0;

        if (this->attr_counts.count(attr)){
            attr_count = this->attr_counts.at(attr);
        }

        for (auto val: val_set) {
            // std::cout << val << std::endl;
            COUNT_TYPE av_count = 0;
            if (this->av_counts.count(attr) and this->av_counts.at(attr).count(val)){
                av_count = this->av_counts.at(attr).at(val);
            }

            double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
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

inline VALUE_TYPE MultinomialCobwebNode::predict(ATTR_TYPE attr, std::string choiceFn, bool allowNone) {
    std::function<ATTR_TYPE(std::vector<std::tuple<VALUE_TYPE, double>>)> choose;
    if (choiceFn == "most likely" || choiceFn == "m") {
        choose = most_likely_choice;
    } else if (choiceFn == "sampled" || choiceFn == "s") {
        choose = weighted_choice;
    } else throw "Unknown choice_fn";
    if (!this->av_counts.count(attr)) {
        return NULL_STRING;
    }
    std::vector<std::tuple<VALUE_TYPE, double>> choices = this->get_weighted_values(attr, allowNone);
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

inline double MultinomialCobwebNode::category_utility(){
    double p_of_c = (1.0 * this->count) / this->tree->root->count;
    return (p_of_c * (this->tree->root->score() - this->score()));
}

inline double MultinomialCobwebNode::log_prob_class_given_instance(const AV_COUNT_TYPE &instance, bool use_root_counts){

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

            float alpha = this->tree->alpha(num_vals);
            float av_count = alpha;
            if (this->av_counts.count(attr) && this->av_counts.at(attr).count(val)){
                av_count += this->av_counts.at(attr).at(val);
                // std::cout << val << "(" << this->av_counts.at(attr).at(val) << ") ";
            }

            // the cnt here is because we have to compute probability over all context words.
            log_prob += cnt * log((1.0 * av_count) / (this->attr_counts[attr] + num_vals * alpha));
        }

        // std::cout << std::endl;
        // std::cout << "denom: " << std::to_string(this->counts[attr] + num_vals * this->tree->alpha) << std::endl;
        // std::cout << "denom (no alpha): " << std::to_string(this->counts[attr]) << std::endl;
        // std::cout << "node count: " << std::to_string(this->count) << std::endl;
        // std::cout << "num vals: " << std::to_string(num_vals) << std::endl;
    }

    // std::cout << std::endl;

    if (use_root_counts){
        log_prob += log((1.0 * this->count) / this->tree->root->count);
    }
    else{
        log_prob += log((1.0 * this->count) / this->parent->count);
    }

    return log_prob;
}



int main(int argc, char* argv[]) {
    std::vector<AV_COUNT_TYPE> instances;
    std::vector<MultinomialCobwebNode*> cs;
    auto tree = MultinomialCobwebTree(true, 1.0, true, true);

    for (int i = 0; i < 1000; i++){
        INSTANCE_TYPE inst;
        inst["anchor"]["word" + std::to_string(i)] = 1;
        inst["anchor2"]["word" + std::to_string(i % 10)] = 1;
        inst["anchor3"]["word" + std::to_string(i % 20)] = 1;
        inst["anchor4"]["word" + std::to_string(i % 100)] = 1;
        cs.push_back(tree.ifit(inst));
    }

    return 0;
}


PYBIND11_MODULE(multinomial_cobweb, m) {
    m.doc() = "concept_formation.multinomial_cobweb plugin"; // optional module docstring

    py::class_<MultinomialCobwebNode>(m, "MultinomialCobwebNode")
        .def(py::init<>())
        .def("pretty_print", &MultinomialCobwebNode::pretty_print)
        .def("output_json", &MultinomialCobwebNode::output_json)
        .def("predict_probs", &MultinomialCobwebNode::predict_probs)
        .def("predict", &MultinomialCobwebNode::predict, py::arg("attr") = "",
                py::arg("choiceFn") = "most likely",
                py::arg("allowNone") = true )
        .def("get_best_level", &MultinomialCobwebNode::get_best_level, py::return_value_policy::reference)
        .def("get_basic_level", &MultinomialCobwebNode::get_basic_level, py::return_value_policy::reference)
        .def("log_prob_class_given_instance", &MultinomialCobwebNode::log_prob_class_given_instance)
        .def("score", &MultinomialCobwebNode::score)
        .def("category_utility", &MultinomialCobwebNode::category_utility)
        .def("partition_utility", &MultinomialCobwebNode::partition_utility)
        .def("__str__", &MultinomialCobwebNode::__str__)
        .def("concept_hash", &MultinomialCobwebNode::concept_hash)
        .def_readonly("count", &MultinomialCobwebNode::count)
        .def_readonly("children", &MultinomialCobwebNode::children, py::return_value_policy::reference)
        .def_readonly("parent", &MultinomialCobwebNode::parent, py::return_value_policy::reference)
        .def_readonly("av_counts", &MultinomialCobwebNode::av_counts, py::return_value_policy::reference)
        .def_readonly("attr_counts", &MultinomialCobwebNode::attr_counts, py::return_value_policy::reference)
        .def_readonly("tree", &MultinomialCobwebNode::tree, py::return_value_policy::reference);

    py::class_<MultinomialCobwebTree>(m, "MultinomialCobwebTree")
        .def(py::init<bool, float, bool, bool>(),
                py::arg("use_mutual_info") = true,
                py::arg("alpha_weight") = 1.0,
                py::arg("dynamic_alpha") = true,
                py::arg("weight_attr") = true)
        .def("ifit", &MultinomialCobwebTree::ifit, py::return_value_policy::reference)
        .def("fit", &MultinomialCobwebTree::fit,
                py::arg("instances") = std::vector<AV_COUNT_TYPE>(),
                py::arg("iterations") = 1,
                py::arg("randomizeFirst") = true)
        .def("categorize", &MultinomialCobwebTree::categorize,
                py::arg("instance") = std::vector<AV_COUNT_TYPE>(),
                // py::arg("get_best_concept") = false,
                py::return_value_policy::reference)
        .def("clear", &MultinomialCobwebTree::clear)
        .def("__str__", &MultinomialCobwebTree::__str__)
        .def("dump_json", &MultinomialCobwebTree::dump_json)
        .def("load_json", &MultinomialCobwebTree::load_json)
        .def_readonly("root", &MultinomialCobwebTree::root, py::return_value_policy::reference);
}
