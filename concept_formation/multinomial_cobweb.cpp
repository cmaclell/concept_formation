#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <sstream>
#include <iomanip>
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
        ATTR_COUNT_TYPE a_count;
        ATTR_COUNT_TYPE sum_n_logn;
        AV_COUNT_TYPE av_count;

        MultinomialCobwebNode();
        MultinomialCobwebNode(MultinomialCobwebNode *otherNode);
        void increment_counts(const AV_COUNT_TYPE &instance);
        void update_counts_from_node(MultinomialCobwebNode *node);
        double entropy_attr_insert(ATTR_TYPE attr, const AV_COUNT_TYPE &instance);
        double entropy_insert(const AV_COUNT_TYPE &instance);
        double entropy_attr_merge(ATTR_TYPE attr, MultinomialCobwebNode *other, const AV_COUNT_TYPE
                &instance);
        double entropy_merge(MultinomialCobwebNode *other, const AV_COUNT_TYPE
                &instance);
        MultinomialCobwebNode* get_best_level(INSTANCE_TYPE instance);
        MultinomialCobwebNode* get_basic_level();
        double category_utility();
        double entropy_attr(ATTR_TYPE attr);
        double entropy();
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
        std::string a_count_to_json();
        std::string sum_n_logn_to_json();
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
        float alpha;
        bool weight_attr;
        int objective;
        bool children_norm;
        bool norm_attributes;
        MultinomialCobwebNode *root;
        AV_KEY_TYPE attr_vals;

        MultinomialCobwebTree(float alpha, bool weight_attr, int objective, bool children_norm, bool norm_attributes) {
            this->alpha = alpha;
            this->weight_attr = weight_attr;
            this->objective = objective;
            this->children_norm = children_norm;
            this->norm_attributes = norm_attributes;

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

            // Get a_count
            struct json_object_element_s* a_count_obj = count_obj->next;
            struct json_object_s* a_count_dict = json_value_as_object(a_count_obj->value);
            struct json_object_element_s* a_count_cursor = a_count_dict->start;
            while(a_count_cursor != NULL) {
                // Get attr name
                std::string attr_name = std::string(a_count_cursor->name->string);

                // A count is stored with each attribute
                double count_value = atof(json_value_as_number(a_count_cursor->value)->number);
                new_node->a_count[attr_name] = count_value;

                a_count_cursor = a_count_cursor->next;
            }

            // Get sum_n_logn
            struct json_object_element_s* sum_n_logn_obj = a_count_obj->next;
            struct json_object_s* sum_n_logn_dict = json_value_as_object(sum_n_logn_obj->value);
            struct json_object_element_s* sum_n_logn_cursor = sum_n_logn_dict->start;
            while(sum_n_logn_cursor != NULL) {
                // Get attr name
                std::string attr_name = std::string(sum_n_logn_cursor->name->string);

                // A count is stored with each attribute
                double count_value = atof(json_value_as_number(sum_n_logn_cursor->value)->number);
                new_node->sum_n_logn[attr_name] = count_value;
                sum_n_logn_cursor = sum_n_logn_cursor->next;
            }

            // Get av counts
            struct json_object_element_s* av_count_obj = sum_n_logn_obj->next;
            struct json_object_s* av_count_dict = json_value_as_object(av_count_obj->value);
            struct json_object_element_s* av_count_cursor = av_count_dict->start;
            while(av_count_cursor != NULL) {
                // Get attr name
                std::string attr_name = std::string(av_count_cursor->name->string);

                // The attr val is a dict of strings to ints
                struct json_object_s* attr_val_dict = json_value_as_object(av_count_cursor->value);
                struct json_object_element_s* inner_counts_cursor = attr_val_dict->start;
                while(inner_counts_cursor != NULL) {
                    // this will be a word
                    std::string val_name = std::string(inner_counts_cursor->name->string);

                    // This will always be a number
                    double attr_val_count = atof(json_value_as_number(inner_counts_cursor->value)->number);
                    // Update the new node's counts
                    new_node->av_count[attr_name][val_name] = attr_val_count;

                    inner_counts_cursor = inner_counts_cursor->next;
                }

                av_count_cursor = av_count_cursor->next;
            }

            // At this point in the coding, I am supremely annoyed at
            // myself for choosing this approach.

            // Get children
            struct json_object_element_s* children_obj = av_count_obj->next;
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
            output += "\"objective\": " + std::to_string(this->objective) + ",\n";
            output += "\"children_norm\": " + std::to_string(this->children_norm) + ",\n";
            output += "\"norm_attributes\": " + std::to_string(this->norm_attributes) + ",\n";
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
            
            // objective
            struct json_object_element_s* objective_obj = weight_attr_obj->next;
            int objective = atoi(json_value_as_number(objective_obj->value)->number);
            this->objective = objective;

            // children_norm
            struct json_object_element_s* children_norm_obj = objective_obj->next;
            bool children_norm = bool(atoi(json_value_as_number(children_norm_obj->value)->number));
            this->children_norm = children_norm;
            
            // norm_attributes
            struct json_object_element_s* norm_attributes_obj = children_norm_obj->next;
            bool norm_attributes = bool(atoi(json_value_as_number(norm_attributes_obj->value)->number));
            this->norm_attributes = norm_attributes;

            // root
            struct json_object_element_s* root_obj = norm_attributes_obj->next;
            struct json_object_s* root = json_value_as_object(root_obj->value);

            delete this->root;
            this->root = this->load_json_helper(root);

            for (auto &[attr, val_map]: this->root->av_count) {
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
    sum_n_logn = ATTR_COUNT_TYPE();
    a_count = ATTR_COUNT_TYPE();
    parent = nullptr;
    tree = nullptr;
}

inline MultinomialCobwebNode::MultinomialCobwebNode(MultinomialCobwebNode *otherNode) {
    count = 0;
    sum_n_logn = ATTR_COUNT_TYPE();
    a_count = ATTR_COUNT_TYPE();

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
            this->a_count[attr] += cnt;

            if (!attr.is_hidden()){
                if(this->av_count.count(attr) && this->av_count.at(attr).count(val)){
                    double tf = this->av_count.at(attr).at(val) + this->tree->alpha;
                    this->sum_n_logn[attr] -= tf * log(tf);
                }
            }

            this->av_count[attr][val] += cnt;

            if (!attr.is_hidden()){
                double tf = this->av_count.at(attr).at(val) + this->tree->alpha;
                this->sum_n_logn[attr] += tf * log(tf);
                // std::cout << "av_count for [" << attr.get_string() << "] = [" << val.get_string() << "]: " << this->av_count[attr][val] << std::endl;
                // std::cout << "updated sum nlogn for [" << attr.get_string() << "]: " << this->sum_n_logn[attr] << std::endl;
            }
        }
    }
}

inline void MultinomialCobwebNode::update_counts_from_node(MultinomialCobwebNode *node) {
    this->count += node->count;

    for (auto &[attr, val_map]: node->av_count) {
        this->a_count[attr] += node->a_count.at(attr);

        for (auto&[val, cnt]: val_map) {
            if (!attr.is_hidden()){
                if(this->av_count.count(attr) && this->av_count.at(attr).count(val)){
                    double tf = this->av_count.at(attr).at(val) + this->tree->alpha;
                    this->sum_n_logn[attr] -= tf * log(tf);
                }
            }

            this->av_count[attr][val] += cnt;

            if (!attr.is_hidden()){
                double tf = this->av_count.at(attr).at(val) + this->tree->alpha;
                this->sum_n_logn[attr] += tf * log(tf);
            }
        }
    }
}

inline double MultinomialCobwebNode::entropy_attr_insert(ATTR_TYPE attr, const AV_COUNT_TYPE &instance){
    if (attr.is_hidden()) return 0.0;

    float alpha = this->tree->alpha;
    int num_vals_total = this->tree->attr_vals.at(attr).size();
    int num_vals_in_c = 0;
    COUNT_TYPE attr_count = 0;

    double ratio = 1.0;
    if (this->tree->weight_attr){
        ratio = (1.0 * this->tree->root->a_count.at(attr)) / (this->tree->root->count);
        // ratio = (1.0 * attr_count) / this->count;
    }
    // ratio = std::ceil(ratio);

    if (this->av_count.count(attr)){
        attr_count = this->a_count.at(attr);
        num_vals_in_c = this->av_count.at(attr).size();
    }

    double sum_n_logn = 0.0;
    if (this->sum_n_logn.count(attr)){
        sum_n_logn = this->sum_n_logn.at(attr);
    }

    if (instance.count(attr)){
        for (auto &[val, cnt]: instance.at(attr)){
            attr_count += cnt;
            COUNT_TYPE prior_av_count = 0.0;
            if (this->av_count.count(attr) && this->av_count.at(attr).count(val)){
                prior_av_count = this->av_count.at(attr).at(val);
                COUNT_TYPE tf = prior_av_count + this->tree->alpha;
                sum_n_logn -= tf * log(tf);
            }
            else{
                num_vals_in_c += 1;
            }
            COUNT_TYPE tf = prior_av_count + cnt + this->tree->alpha;
            sum_n_logn += (tf) * log(tf);
        }
    }

    int n0 = num_vals_total - num_vals_in_c;
    double info = -ratio * ((1 / (attr_count + num_vals_total * alpha)) *
            (sum_n_logn + n0 * alpha * log(alpha)) - log(attr_count +
                num_vals_total * alpha));
    return info;
}

inline double MultinomialCobwebNode::entropy_insert(const AV_COUNT_TYPE &instance){

    double info = 0.0;

    for (auto &[attr, av_inner]: this->av_count){
        if (attr.is_hidden()) continue;
        info += this->entropy_attr_insert(attr, instance);
     }

    // iterate over attr in instance not in av_count
    for (auto &[attr, av_inner]: instance){
        if (attr.is_hidden()) continue;
        if (this->av_count.count(attr)) continue;
        info += this->entropy_attr_insert(attr, instance);
     }

    return info;
}

inline double MultinomialCobwebNode::entropy_attr_merge(ATTR_TYPE attr,
        MultinomialCobwebNode *other, const AV_COUNT_TYPE &instance) {

    if (attr.is_hidden()) return 0.0;

    float alpha = this->tree->alpha;
    int num_vals_total = this->tree->attr_vals.at(attr).size();
    int num_vals_in_c = 0;
    COUNT_TYPE attr_count = 0;

    double ratio = 1.0;
    if (this->tree->weight_attr){
        ratio = (1.0 * this->tree->root->a_count.at(attr)) / (this->tree->root->count);
        // ratio = (1.0 * attr_count) / this->count;
    }
    // ratio = std::ceil(ratio);

    if (this->av_count.count(attr)){
        attr_count = this->a_count.at(attr);
        num_vals_in_c = this->av_count.at(attr).size();
    }

    double sum_n_logn = 0.0;
    if (this->sum_n_logn.count(attr)){
        sum_n_logn = this->sum_n_logn.at(attr);
    }

    if (other->av_count.count(attr)){
        for (auto &[val, other_av_count]: other->av_count.at(attr)){
            COUNT_TYPE instance_av_count = 0.0;

            if (instance.count(attr) && instance.at(attr).count(val)){
                instance_av_count = instance.at(attr).at(val);
            }

            attr_count += other_av_count + instance_av_count;
            COUNT_TYPE prior_av_count = 0.0;
            if (this->av_count.count(attr) && this->av_count.at(attr).count(val)){
                prior_av_count = this->av_count.at(attr).at(val);
                COUNT_TYPE tf = prior_av_count + alpha;
                sum_n_logn -= tf * log(tf);
            }
            else{
                num_vals_in_c += 1;
            }

            COUNT_TYPE new_tf = prior_av_count + other_av_count + instance_av_count + alpha;
            sum_n_logn += (new_tf) * log(new_tf);
        }
    }

    if (instance.count(attr)){
        for (auto &[val, instance_av_count]: instance.at(attr)){
            if (other->av_count.count(attr) && other->av_count.at(attr).count(val)){
                continue;
            }
            COUNT_TYPE other_av_count = 0.0;

            attr_count += other_av_count + instance_av_count;
            COUNT_TYPE prior_av_count = 0.0;
            if (this->av_count.count(attr) && this->av_count.at(attr).count(val)){
                prior_av_count = this->av_count.at(attr).at(val);
                COUNT_TYPE tf = prior_av_count + alpha;
                sum_n_logn -= tf * log(tf);
            }
            else{
                num_vals_in_c += 1;
            }

            COUNT_TYPE new_tf = prior_av_count + other_av_count + instance_av_count + alpha;
            sum_n_logn += (new_tf) * log(new_tf);
        }
    }

    int n0 = num_vals_total - num_vals_in_c;
    double info = -ratio * ((1 / (attr_count + num_vals_total * alpha)) *
            (sum_n_logn + n0 * alpha * log(alpha)) - log(attr_count +
                num_vals_total * alpha));
    return info;
}



inline double MultinomialCobwebNode::entropy_merge(MultinomialCobwebNode *other,
        const AV_COUNT_TYPE &instance) {

    double info = 0.0;

    for (auto &[attr, inner_vals]: this->av_count){
        if (attr.is_hidden()) continue;
        info += this->entropy_attr_merge(attr, other, instance);
    }

    for (auto &[attr, inner_vals]: other->av_count){
        if (attr.is_hidden()) continue;
        if (this->av_count.count(attr)) continue;
        info += this->entropy_attr_merge(attr, other, instance);
    }

    for (auto &[attr, inner_vals]: instance){
        if (attr.is_hidden()) continue;
        if (this->av_count.count(attr)) continue;
        if (other->av_count.count(attr)) continue;
        info += entropy_attr_merge(attr, other, instance);
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

inline double MultinomialCobwebNode::entropy_attr(ATTR_TYPE attr){
    if (attr.is_hidden()) return 0.0;

    float alpha = this->tree->alpha;
    int num_vals_total = this->tree->attr_vals.at(attr).size();
    int num_vals_in_c = 0;
    COUNT_TYPE attr_count = 0;

    if (this->av_count.count(attr)){
        attr_count = this->a_count.at(attr);
        num_vals_in_c = this->av_count.at(attr).size();
    }

    double ratio = 1.0;
    if (this->tree->weight_attr){
        ratio = (1.0 * this->tree->root->a_count.at(attr)) / (this->tree->root->count);
        // ratio = (1.0 * attr_count) / this->count;
    }
    // ratio = std::ceil(ratio);

    double sum_n_logn = 0.0;
    if (this->sum_n_logn.count(attr)){
        sum_n_logn = this->sum_n_logn.at(attr);
    }


    int n0 = num_vals_total - num_vals_in_c;
    // std::cout << "sum n logn: " << sum_n_logn << std::endl;
    // std::cout << "n0: " << n0 << std::endl;
    // std::cout << "alpha: " << alpha << std::endl;
    // std::cout << "attr_count: " << attr_count << std::endl;
    double info = -ratio * ((1 / (attr_count + num_vals_total * alpha)) *
            (sum_n_logn + n0 * alpha * log(alpha)) - log(attr_count +
                num_vals_total * alpha));
    return info;

    /* 
    int n = std::ceil(ratio);
    info -= lgamma_cached(n+1);

    for (auto &[val, cnt]: inner_av){
        double p = ((cnt + alpha) / (attr_count + num_vals * alpha));
        info += entropy_component_k(n, p);
    }

    COUNT_TYPE num_missing = num_vals - inner_av.size();
    if (num_missing > 0 and alpha > 0){
        double p = (alpha / (attr_count + num_vals * alpha));
        info += num_missing * entropy_component_k(n, p);
    }

    return info;
    */

}

inline double MultinomialCobwebNode::entropy() {

    double info = 0.0;
    for (auto &[attr, inner_av]: this->av_count){
        if (attr.is_hidden()) continue;
        info += this->entropy_attr(attr);
    }

    return info;
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
    if (children.empty()) {
        return 0.0;
    }

    // BEGIN INDIVIDUAL
    if (!this->tree->norm_attributes){
        double parent_entropy = 0.0;
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &[attr, val_set]: this->tree->attr_vals) {
            parent_entropy += this->entropy_attr(attr);
        }

        for (auto &child: children) {
            double p_of_child = (1.0 * child->count) / this->count;
            concept_entropy -= p_of_child * log(p_of_child);

            for (auto &[attr, val_set]: this->tree->attr_vals) {
                children_entropy += p_of_child * child->entropy_attr(attr);
            }
        }

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1){
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2){
            obj /= (children_entropy + concept_entropy);
        }
        if (this->tree->children_norm){
            obj /= this->children.size();
        }
        return obj;
    }
    // END INDIVIDUAL

    double entropy = 0.0;

    for (auto &[attr, val_set]: this->tree->attr_vals) {
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &child: children) {
            double p_of_child = (1.0 * child->count) / this->count;
            children_entropy += p_of_child * child->entropy_attr(attr);
            concept_entropy -= p_of_child * log(p_of_child);
        }

        double parent_entropy = this->entropy_attr(attr);

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1){
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2){
            obj /= (children_entropy + concept_entropy);
        }

        if (this->tree->children_norm){
            obj /= this->children.size();
        }
        entropy += obj;

        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy) / this->children.size(); 
        // entropy += (parent_entropy - children_entropy) / parent_entropy;
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy);
    }

    return entropy;

}

inline double MultinomialCobwebNode::pu_for_insert(MultinomialCobwebNode *child, const AV_COUNT_TYPE &instance) {

    // BEGIN INDIVIDUAL
    if (!this->tree->norm_attributes){
        double parent_entropy = 0.0;
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &[attr, val_set]: this->tree->attr_vals) {
            parent_entropy += this->entropy_attr_insert(attr, instance);
        }

        for (auto &c: children) {
            if (c == child) {
                double p_of_child = (c->count + 1.0) / (this->count + 1.0);
                concept_entropy -= p_of_child * log(p_of_child);
                for (auto &[attr, val_set]: this->tree->attr_vals) {
                    children_entropy += p_of_child * c->entropy_attr_insert(attr, instance);
                }
            }
            else{
                double p_of_child = (1.0 * c->count) / (this->count + 1.0);
                concept_entropy -= p_of_child * log(p_of_child);

                for (auto &[attr, val_set]: this->tree->attr_vals) {
                    children_entropy += p_of_child * c->entropy_attr(attr);
                }
            }
        }

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1){
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2){
            obj /= (children_entropy + concept_entropy);
        }
        if (this->tree->children_norm){
            obj /= this->children.size();
        }
        return obj;
    }
    // END INDIVIDUAL

    double entropy = 0.0;

    for (auto &[attr, val_set]: this->tree->attr_vals) {
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &c: this->children) {
            if (c == child) {
                double p_of_child = (c->count + 1.0) / (this->count + 1.0);
                children_entropy += p_of_child * c->entropy_attr_insert(attr, instance);
                concept_entropy -= p_of_child * log(p_of_child);
            }
            else{
                double p_of_child = (1.0 * c->count) / (this->count + 1.0);
                children_entropy += p_of_child * c->entropy_attr(attr);
                concept_entropy -= p_of_child * log(p_of_child);
            }
        }

        double parent_entropy = this->entropy_attr_insert(attr, instance);

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1){
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2){
            obj /= (children_entropy + concept_entropy);
        }

        if (this->tree->children_norm){
            obj /= this->children.size();
        }
        entropy += obj;

        // entropy += (parent_entropy - children_entropy) / this->children.size();
        // entropy += (parent_entropy - children_entropy) / parent_entropy / this->children.size(); 
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy) / this->children.size(); 
        // entropy += (parent_entropy - children_entropy) / parent_entropy;
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy); 

    }

    return entropy;
}

inline double MultinomialCobwebNode::pu_for_new_child(const AV_COUNT_TYPE &instance) {
    

    // TODO maybe modify so that we can evaluate new child without copying
    // instance.
    MultinomialCobwebNode new_child = MultinomialCobwebNode();
    new_child.parent = this;
    new_child.tree = this->tree;
    new_child.increment_counts(instance);
    double p_of_new_child = 1.0 / (this->count + 1.0);

    // BEGIN INDIVIDUAL
    if (!this->tree->norm_attributes){
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
        if (this->tree->objective == 1){
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2){
            obj /= (children_entropy + concept_entropy);
        }
        if (this->tree->children_norm){
            obj /= (this->children.size() + 1);
        }
        return obj;
    }
    // END INDIVIDUAL

    double entropy = 0.0;

    for (auto &[attr, val_set]: this->tree->attr_vals) {
        double children_entropy = p_of_new_child * new_child.entropy_attr(attr);
        double concept_entropy = -p_of_new_child * log(p_of_new_child);

        for (auto &c: this->children) {
            double p_of_child = (1.0 * c->count) / (this->count + 1.0);
            children_entropy += p_of_child * c->entropy_attr(attr);
            concept_entropy -= p_of_child * log(p_of_child);
        }

        double parent_entropy = this->entropy_attr_insert(attr, instance);

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1){
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2){
            obj /= (children_entropy + concept_entropy);
        }

        if (this->tree->children_norm){
            obj /= (this->children.size() + 1);
        }
        entropy += obj;

        // entropy += (parent_entropy - children_entropy) / (this->children.size() + 1);
        // entropy += (parent_entropy - children_entropy) / parent_entropy / (this->children.size() + 1);
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy) / (this->children.size() + 1);
        // entropy += (parent_entropy - children_entropy) / parent_entropy;
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy); 
    }

    return entropy;
}

inline double MultinomialCobwebNode::pu_for_merge(MultinomialCobwebNode *best1,
        MultinomialCobwebNode *best2, const AV_COUNT_TYPE &instance) {
    
    // BEGIN INDIVIDUAL
    if (!this->tree->norm_attributes){
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
        if (this->tree->objective == 1){
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2){
            obj /= (children_entropy + concept_entropy);
        }
        if (this->tree->children_norm){
            obj /= (this->children.size() - 1);
        }
        return obj;
    }
    // END INDIVIDUAL

    double entropy = 0.0;

    for (auto &[attr, val_set]: this->tree->attr_vals) {
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &c: children) {
            if (c == best1 || c == best2){
                continue;
            }

            double p_of_child = (1.0 * c->count) / (this->count + 1.0);
            children_entropy += p_of_child * c->entropy_attr(attr);
            concept_entropy -= p_of_child * log(p_of_child);
        }

        double p_of_child = (best1->count + best2->count + 1.0) / (this->count + 1.0);
        children_entropy += p_of_child * best1->entropy_attr_merge(attr, best2, instance);
        concept_entropy -= p_of_child * log(p_of_child);

        double parent_entropy = this->entropy_attr_insert(attr, instance);

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1){
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2){
            obj /= (children_entropy + concept_entropy);
        }

        if (this->tree->children_norm){
            obj /= (this->children.size() - 1);
        }
        entropy += obj;

        // entropy += (parent_entropy - children_entropy) / (this->children.size() - 1);
        // entropy += (parent_entropy - children_entropy) / parent_entropy / (this->children.size() - 1);
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy) / (this->children.size() - 1);
        // entropy += (parent_entropy - children_entropy) / parent_entropy;
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy); 
    }

    return entropy;
}

inline double MultinomialCobwebNode::pu_for_split(MultinomialCobwebNode *best){

    // BEGIN INDIVIDUAL
    if (!this->tree->norm_attributes){
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
        if (this->tree->objective == 1){
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2){
            obj /= (children_entropy + concept_entropy);
        }
        if (this->tree->children_norm){
            obj /= (this->children.size() - 1 + best->children.size());
        }
        return obj;
    }
    // END INDIVIDUAL

    double entropy = 0.0;

    for (auto &[attr, val_set]: this->tree->attr_vals) {

        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &c: children) {
            if (c == best) continue;
            double p_of_child = (1.0 * c->count) / this->count;
            children_entropy += p_of_child * c->entropy_attr(attr);
            concept_entropy -= p_of_child * log(p_of_child);
        }

        for (auto &c: best->children) {
            double p_of_child = (1.0 * c->count) / this->count;
            children_entropy += p_of_child * c->entropy_attr(attr);
            concept_entropy -= p_of_child * log(p_of_child);
        }

        double parent_entropy = this->entropy_attr(attr);

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1){
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2){
            obj /= (children_entropy + concept_entropy);
        }

        if (this->tree->children_norm){
            obj /= (this->children.size() - 1 + best->children.size());
        }
        entropy += obj;

        // entropy += (parent_entropy - children_entropy) / (this->children.size() - 1 + best->children.size());
        // entropy += (parent_entropy - children_entropy) / parent_entropy / (this->children.size() - 1 + best->children.size());
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy) / (this->children.size() - 1 + best->children.size());
        // entropy += (parent_entropy - children_entropy) / parent_entropy;
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy); 

    }

    return entropy;
}

inline bool MultinomialCobwebNode::is_exact_match(const AV_COUNT_TYPE &instance) {
    std::unordered_set<ATTR_TYPE> all_attrs;
    for (auto &[attr, tmp]: instance) all_attrs.insert(attr);
    for (auto &[attr, tmp]: this->av_count) all_attrs.insert(attr);

    for (auto &attr: all_attrs) {
        if (attr.is_hidden()) continue;
        if (instance.count(attr) && !this->av_count.count(attr)) {
            return false;
        }
        if (this->av_count.count(attr) && !instance.count(attr)) {
            return false;
        }
        if (this->av_count.count(attr) && instance.count(attr)) {
            double instance_attr_count = 0.0;
            std::unordered_set<VALUE_TYPE> all_vals;
            for (auto &[val, tmp]: this->av_count.at(attr)) all_vals.insert(val);
            for (auto &[val, cnt]: instance.at(attr)){
                all_vals.insert(val);
                instance_attr_count += cnt;
            }

            for (auto &val: all_vals) {
                if (instance.at(attr).count(val) && !this->av_count.at(attr).count(val)) {
                    return false;
                }
                if (this->av_count.at(attr).count(val) && !instance.at(attr).count(val)) {
                    return false;
                }

                double instance_prob = (1.0 * instance.at(attr).at(val)) / instance_attr_count;
                double concept_prob = (1.0 * this->av_count.at(attr).at(val)) / this->a_count.at(attr);

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

    // ret += "\"_mutual_info\": {\n";
    // ret += "\"#ContinuousValue#\": {\n";
    // ret += "\"mean\": " + std::to_string(this->mutual_information()) + ",\n";
    // ret += "\"std\": 1,\n";
    // ret += "\"n\": 1,\n";
    // ret += "}},\n";

    int c = 0;
    for (auto &[attr, vAttr]: av_count) {
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

        if (c != int(av_count.size())-1){
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
    for (auto &[attr, vAttr]: av_count) {
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

        if (c != int(av_count.size())-1){
            ret += ", ";
        }
        c++;
    }
    ret += "}";
    return ret;
}

inline std::string MultinomialCobwebNode::a_count_to_json() {
    std::string ret = "{";

    bool first = true;
    for (auto &[attr, cnt]: this->a_count) {
        if (!first) ret += ",\n";
        else first = false;
        ret += "\"" + attr.get_string() + "\": " + doubleToString(cnt);
            // std::to_string(cnt);
    }

    ret += "}";
    return ret;
}

inline std::string MultinomialCobwebNode::sum_n_logn_to_json() {
    std::string ret = "{";

    bool first = true;
    for (auto &[attr, cnt]: this->sum_n_logn) {
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
    output += "\"a_count\": " + this->a_count_to_json() + ",\n";
    output += "\"sum_n_logn\": " + this->sum_n_logn_to_json() + ",\n";
    output += "\"av_count\": " + this->ser_avcounts() + ",\n";

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
    output += "\"attr_counts\": " + this->a_count_to_json() + "\n";

    output += "}\n";

    return output;
}

inline std::unordered_map<std::string, std::unordered_map<std::string, double>> MultinomialCobwebNode::predict_probs(){
    std::unordered_map<std::string, std::unordered_map<std::string, double>> out;
    for (auto &[attr, val_set]: this->tree->attr_vals) {
        // std::cout << attr << std::endl;
        int num_vals = this->tree->attr_vals.at(attr).size();
        float alpha = this->tree->alpha;
        COUNT_TYPE attr_count = 0;

        if (this->a_count.count(attr)){
            attr_count = this->a_count.at(attr);
        }

        for (auto val: val_set) {
            // std::cout << val << std::endl;
            COUNT_TYPE av_count = 0;
            if (this->av_count.count(attr) and this->av_count.at(attr).count(val)){
                av_count = this->av_count.at(attr).at(val);
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
    if (!this->av_count.count(attr)) {
        choices.push_back(std::make_tuple(NULL_STRING, 1.0));
    }
    double valCount = 0;
    for (auto &[val, tmp]: this->av_count.at(attr)) {
        COUNT_TYPE count = this->av_count.at(attr).at(val);
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
    if (!this->av_count.count(attr)) {
        return NULL_STRING;
    }
    std::vector<std::tuple<VALUE_TYPE, double>> choices = this->get_weighted_values(attr, allowNone);
    return choose(choices);
}

inline double MultinomialCobwebNode::probability(ATTR_TYPE attr, VALUE_TYPE val) {
    if (val == NULL_STRING) {
        double c = 0.0;
        if (this->av_count.count(attr)) {
            for (auto &[attr, vAttr]: this->av_count) {
                for (auto&[val, cnt]: vAttr) {
                    c += cnt;
                }
            }
            return (1.0 * (this->count - c)) / this->count;
        }
    }
    if (this->av_count.count(attr) && this->av_count.at(attr).count(val)) {
        return (1.0 * this->av_count.at(attr).at(val)) / this->count;
    }
    return 0.0;
}

inline double MultinomialCobwebNode::category_utility(){
    // double p_of_c = (1.0 * this->count) / this->tree->root->count;
    // return (p_of_c * (this->tree->root->entropy() - this->entropy()));

    double root_entropy = 0.0;
    double child_entropy = 0.0;

    double p_of_child = (1.0 * this->count) / this->tree->root->count;
    for (auto &[attr, val_set]: this->tree->attr_vals) {
        root_entropy += this->tree->root->entropy_attr(attr);
        child_entropy += this->entropy_attr(attr);
    }

    return p_of_child * (root_entropy - child_entropy);

}

inline double MultinomialCobwebNode::log_prob_class_given_instance(const AV_COUNT_TYPE &instance, bool use_root_counts){

    double log_prob = 0;

    // std::cout << std::endl;

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
            double av_count = alpha;
            if (this->av_count.count(attr) && this->av_count.at(attr).count(val)){
                av_count += this->av_count.at(attr).at(val);
                // std::cout << val << "(" << this->av_count.at(attr).at(val) << ") ";
            }

            // the cnt here is because we have to compute probability over all context words.
            COUNT_TYPE a_count = 0.0;
            if (this->a_count.count(attr)){
                a_count = this->a_count.at(attr);
            }
            log_prob += cnt * (log(av_count) - log(a_count + num_vals * alpha));

            /*
            if (av_count > 1.0){
                double tmp = cnt * (log(av_count) - log(this->a_count[attr] + num_vals * alpha));
                std::cout << "Depth: " << this->depth() << " Value: " << val.get_string() << ": " << std::to_string(tmp) << " (" << std::to_string(cnt) << " x " << std::to_string(av_count) << "/" << std::to_string(this->a_count[attr] + num_vals * alpha) << ")" << std::endl;
            }
            */
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

    // std::cout << "LOB PROB" << std::to_string(log_prob) << std::endl;

    return log_prob;
}



int main(int argc, char* argv[]) {
    std::vector<AV_COUNT_TYPE> instances;
    std::vector<MultinomialCobwebNode*> cs;
    auto tree = MultinomialCobwebTree(0.01, false, 2, true, true);

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
        .def("entropy", &MultinomialCobwebNode::entropy)
        .def("category_utility", &MultinomialCobwebNode::category_utility)
        .def("partition_utility", &MultinomialCobwebNode::partition_utility)
        .def("__str__", &MultinomialCobwebNode::__str__)
        .def("concept_hash", &MultinomialCobwebNode::concept_hash)
        .def_readonly("count", &MultinomialCobwebNode::count)
        .def_readonly("children", &MultinomialCobwebNode::children, py::return_value_policy::reference)
        .def_readonly("parent", &MultinomialCobwebNode::parent, py::return_value_policy::reference)
        .def_readonly("av_count", &MultinomialCobwebNode::av_count, py::return_value_policy::reference)
        .def_readonly("a_count", &MultinomialCobwebNode::a_count, py::return_value_policy::reference)
        .def_readonly("tree", &MultinomialCobwebNode::tree, py::return_value_policy::reference);

    py::class_<MultinomialCobwebTree>(m, "MultinomialCobwebTree")
        .def(py::init<float, bool, int, bool, bool>(),
                py::arg("alpha") = 1.0,
                py::arg("weight_attr") = false,
                py::arg("objective") = 2,
                py::arg("children_norm") = true,
                py::arg("norm_attributes") = true)
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
