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
#define CATEGORY_UTILITY 1
#define MUTUAL_INFORMATION 2
#define NORMALIZED_MUTUAL_INFORMATION 3

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
    vector<tuple<double, double, VALUE_TYPE>> vals;

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
        COUNT_TYPE count;
        unordered_map<ATTR_TYPE, COUNT_TYPE> attr_counts;
        vector<MultinomialCobwebNode *> children;
        MultinomialCobwebNode *parent;
        MultinomialCobwebTree *tree;
        AV_COUNT_TYPE av_counts;

        MultinomialCobwebNode();
        MultinomialCobwebNode(MultinomialCobwebNode *otherNode);

        void increment_counts(const AV_COUNT_TYPE &instance);
        void decrement_counts(const AV_COUNT_TYPE &instance);
        void update_counts_from_node(MultinomialCobwebNode *node);
        void remove_counts_from_node(MultinomialCobwebNode *node);
        MultinomialCobwebNode* get_best_level(const AV_COUNT_TYPE &instance);
        double category_utility();
        double mutual_information();
        double expected_correct_guesses(ATTR_TYPE attribute);
        double entropy(ATTR_TYPE attribute);
        double joint_entropy(ATTR_TYPE attribute);
        double normalized_mutual_information();
        double score();
        tuple<double, string> get_best_operation(const AV_COUNT_TYPE &instance, MultinomialCobwebNode *best1,
                MultinomialCobwebNode *best2, double best1_score,
                bool best_op=true, bool new_op=true,
                bool merge_op=true, bool split_op=true);
        tuple<double, MultinomialCobwebNode*, MultinomialCobwebNode*> two_best_children(const AV_COUNT_TYPE &instance);
        double log_prob_class_given_instance(const AV_COUNT_TYPE &instance);
        double score_for_insert(MultinomialCobwebNode *child, const AV_COUNT_TYPE &instance);
        MultinomialCobwebNode *create_new_child(const AV_COUNT_TYPE &instance);
        void delete_child(MultinomialCobwebNode* child);
        double score_for_new_child(const AV_COUNT_TYPE &instance);
        MultinomialCobwebNode* merge(const vector<MultinomialCobwebNode*>& nodes);
        double score_for_merge(const vector<MultinomialCobwebNode*>& nodes, const AV_COUNT_TYPE &instance);
        vector<MultinomialCobwebNode*> split(MultinomialCobwebNode *best);
        double score_for_split(MultinomialCobwebNode *best);
        bool is_exact_match(const AV_COUNT_TYPE &instance);
        size_t _hash();
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
        int objective;
        float alpha_weight;
        bool dynamic_alpha;
        bool weight_attr;
        bool cat_basic;
        bool predict_mixture;

        MultinomialCobwebTree(int objective, float alpha_weight,
                bool dynamic_alpha, bool weight_attr, bool
                cat_basic, bool predict_mixture) {
            this->objective = objective;
            this->alpha_weight = alpha_weight;
            this->dynamic_alpha = dynamic_alpha;
            this->weight_attr = weight_attr;
            this->cat_basic = cat_basic;
            this->predict_mixture = predict_mixture;

            this->root = new MultinomialCobwebNode();
            this->root->tree = this;
            this->attr_vals = AV_KEY_TYPE();
        }

        string __str__(){
            return this->root->__str__();
        }

        float alpha(ATTR_TYPE attr){
            if (!this->dynamic_alpha){
                return this->alpha_weight;
            }

            COUNT_TYPE n_vals = this->attr_vals.at(attr).size();

            if (n_vals == 0){
                return this->alpha_weight;
            } else {
                return this->alpha_weight / n_vals;
            }
        }

        float attr_weight(ATTR_TYPE attr){
            return (1.0 * this->root->attr_counts.at(attr)) / this->root->count;
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
            // struct json_object_element_s* count_obj = concept_id_obj->next;
            struct json_object_element_s* count_obj = object->start;
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

            for (auto &[attr, val_map]: this->root->av_counts) {
                // if (attr[0] == '_') continue;
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

        MultinomialCobwebNode *ifit(AV_COUNT_TYPE instance) {
            for (auto &[attr, val_map]: instance) {
                // if (attr[0] == '_') continue;
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
                    auto[best1_score, best1, best2] = current->two_best_children(instance);
                    auto[_, bestAction] = current->get_best_operation(instance, best1, best2, best1_score);
                    if (bestAction == "best") {
                        current->increment_counts(instance);
                        current = best1;
                    } else if (bestAction == "new") {
                        current->increment_counts(instance);
                        current = current->create_new_child(instance);
                        break;
                    } else if (bestAction == "merge") {
                        current->increment_counts(instance);
                        MultinomialCobwebNode *newChild = current->merge({best1, best2});
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

            auto best_concept = current;
            double best_score = best_concept->log_prob_class_given_instance(instance);

            while (true) {
                if (current->children.empty()) {
                    if (this->cat_basic) return best_concept;
                    return current;
                }

                auto parent = current;
                current = NULL;
                double best_logp = 0.0;

                for (auto &child: parent->children) {
                    double logp = child->log_prob_class_given_instance(instance);
                    if (current == NULL || logp > best_logp){
                        best_logp = logp;
                        current = child;

                        double score = 0.0;
                        score = logp;

                        if (score > best_score){
                            best_score = score;
                            best_concept = current;
                        }
                    }
                }
            }
        }

        MultinomialCobwebNode *categorize(const AV_COUNT_TYPE &instance) {
            return this->_cobweb_categorize(instance);
        }

        unordered_map<string, unordered_map<string, double>> predict(const AV_COUNT_TYPE &instance){
            unordered_map<string, unordered_map<string, double>> out = unordered_map<string, unordered_map<string, double>>();

            if (this->root->children.empty()){
                return out;
            }

            if (this->predict_mixture){
                for (auto &child: this->root->children) {
                    double log_p_class = child->log_prob_class_given_instance(instance);
                    double p_class = exp(log_p_class);

                    for (auto &[attr, val_set]: this->attr_vals) {
                        float alpha = this->alpha(attr);
                        COUNT_TYPE attr_count = 0;
                        int num_vals = this->attr_vals.at(attr).size();

                        if (child->attr_counts.count(attr)){
                            attr_count = child->attr_counts.at(attr);
                        }

                        for (auto val: val_set) {
                            COUNT_TYPE av_count = 0;
                            if (child->av_counts.count(attr) and child->av_counts.at(attr).count(val)){
                                av_count = child->av_counts.at(attr).at(val);
                            }

                            double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
                            out[attr][val] += p * p_class;
                        }
                    }
                }

                for (auto &[attr, val_set]: this->attr_vals) {
                    double attr_sum = 0.0;
                    for (auto val: val_set) {
                        attr_sum += out[attr][val];
                    }
                    for (auto val: val_set) {
                        out[attr][val] /= attr_sum;
                    }
                }

            } else {
                auto best = this->categorize(instance);

                for (auto &[attr, val_set]: this->attr_vals) {
                    float alpha = this->alpha(attr);
                    int num_vals = this->attr_vals.at(attr).size();
                    COUNT_TYPE attr_count = 0;

                    if (best->attr_counts.count(attr)){
                        attr_count = best->attr_counts.at(attr);
                    }

                    for (auto val: val_set) {
                        COUNT_TYPE av_count = 0;
                        if (best->av_counts.count(attr) and best->av_counts.at(attr).count(val)){
                            av_count = best->av_counts.at(attr).at(val);
                        }

                        double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
                        out[attr][val] += p;
                    }
                }
            }

            return out;

        }

};


inline MultinomialCobwebNode::MultinomialCobwebNode() {
    count = 0;
    attr_counts = unordered_map<ATTR_TYPE, COUNT_TYPE>();
    parent = NULL;
    tree = NULL;
}

inline MultinomialCobwebNode::MultinomialCobwebNode(MultinomialCobwebNode *otherNode) {
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

inline void MultinomialCobwebNode::decrement_counts(const AV_COUNT_TYPE &instance) {
    this->count -= 1;
    for (const auto &[attr, val_map]: instance) {
        for (const auto &[val, cnt]: val_map) {
            this->attr_counts[attr] -= cnt;
            this->av_counts[attr][val] -= cnt;

            if (this->av_counts[attr][val] == 0){
                this->av_counts[attr].erase(val);
            }
        }

        if (this->attr_counts[attr] == 0){
            this->attr_counts.erase(attr);
            this->av_counts.erase(attr);
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

inline void MultinomialCobwebNode::remove_counts_from_node(MultinomialCobwebNode *node) {
    this->count -= node->count;

    for (auto &[attr, val_map]: node->av_counts) {
        this->attr_counts[attr] -= node->attr_counts.at(attr);

        for (auto&[val, cnt]: val_map) {
            this->av_counts[attr][val] -= cnt;

            if (this->av_counts[attr][val] == 0){
                this->av_counts[attr].erase(val);
            }
        }

        if (this->attr_counts[attr] == 0){
            this->attr_counts.erase(attr);
            this->av_counts.erase(attr);
        }
    }
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

inline double MultinomialCobwebNode::expected_correct_guesses(ATTR_TYPE attribute) {
    int num_vals = this->tree->attr_vals.at(attribute).size();
    float alpha = this->tree->alpha(attribute);

    float attr_weight = 1.0;
    if (this->tree->weight_attr){
        attr_weight = this->tree->attr_weight(attribute);
    }

    COUNT_TYPE attr_count = 0;
    double info = 0.0;

    if (this->attr_counts.count(attribute)){
        attr_count = this->attr_counts.at(attribute);
    }

    if (this->av_counts.count(attribute) && (attr_count + num_vals * alpha) > 0){
        info += transform_reduce(PAR this->av_counts.at(attribute).begin(),
            this->av_counts.at(attribute).end(), 0.0, plus<>(), [&](const auto&
                val_it){
            double p_av_given_c = ((val_it.second + alpha) / (attr_count + num_vals * alpha));
            return p_av_given_c * p_av_given_c;
        });
    }

    COUNT_TYPE num_missing = num_vals;

    if (this->av_counts.count(attribute)){
        num_missing -= this->av_counts.at(attribute).size();
    }

    if (num_missing > 0 and alpha > 0){
        double p_av_given_c = (alpha / (attr_count + num_vals * alpha));
        info += num_missing * p_av_given_c * p_av_given_c;
    }

    return attr_weight * info;
    
}

inline double MultinomialCobwebNode::entropy(ATTR_TYPE attribute) {
    int num_vals = this->tree->attr_vals.at(attribute).size();
    float alpha = this->tree->alpha(attribute);

    float attr_weight = 1.0;
    if (this->tree->weight_attr){
        attr_weight = this->tree->attr_weight(attribute);
    }

    COUNT_TYPE attr_count = 0;
    double info = 0.0;

    if (this->attr_counts.count(attribute)){
        attr_count = this->attr_counts.at(attribute);
    }

    if (this->av_counts.count(attribute) && (attr_count + num_vals * alpha) > 0){
        attr_count = this->attr_counts.at(attribute);
        info += transform_reduce(PAR this->av_counts.at(attribute).begin(),
                this->av_counts.at(attribute).end(), 0.0, plus<>(), [&](const auto&
                    val_it){
                double p_av_given_c = ((val_it.second + alpha) / (attr_count + num_vals * alpha));
                return p_av_given_c * log(p_av_given_c);
        });
    }

    COUNT_TYPE num_missing = num_vals;

    if (this->av_counts.count(attribute)){
        num_missing -= this->av_counts.at(attribute).size();
    }

    if (num_missing > 0 and alpha > 0){
        double p_av_given_c = (alpha / (attr_count + num_vals * alpha));
        info += num_missing * p_av_given_c * log(p_av_given_c);
    }

    return -attr_weight * info;
    
}

inline double MultinomialCobwebNode::joint_entropy(ATTR_TYPE attribute) {
    COUNT_TYPE attr_count = this->attr_counts.at(attribute);

    if (this->attr_counts.count(attribute)){
        attr_count = this->attr_counts.at(attribute);
    }

    int num_vals = this->tree->attr_vals.at(attribute).size();
    float alpha = this->tree->alpha(attribute);

    float attr_weight = 1.0;
    if (this->tree->weight_attr){
        attr_weight = this->tree->attr_weight(attribute);
    }

    return -attr_weight * transform_reduce(PAR this->children.begin(),
            this->children.end(), 0.0, plus<>(), [&](const auto& child){

            double info = 0.0;
            double p_of_c = (1.0 * child->count) / (this->count);

            if (this->av_counts.count(attribute) && (attr_count + num_vals * alpha) > 0){
                info += transform_reduce(PAR this->av_counts.at(attribute).begin(),
                    this->av_counts.at(attribute).end(), 0.0, plus<>(), [&](const auto&
                        val_it){
                    double p_av_given_c = ((val_it.second + alpha) / (attr_count + num_vals * alpha));
                    double joint_p = p_av_given_c * p_of_c;
                    return joint_p * log(joint_p);
                });
            }

            COUNT_TYPE num_missing = num_vals;

            if (this->av_counts.count(attribute)){
                num_missing -= this->av_counts.at(attribute).size();
            }

            if (num_missing > 0 and alpha > 0){
                double p_av_given_c = (alpha / (attr_count + num_vals * alpha));
                double joint_p = p_av_given_c * p_of_c;
                info += num_missing * joint_p * log(joint_p);
            }

            return info;
    });
}

inline double MultinomialCobwebNode::category_utility() {
    if (children.empty()) {
        return 0.0;
    }

    float ec_gain = transform_reduce(PAR this->av_counts.begin(), this->av_counts.end(), 0.0,
            plus<>(), [&](const auto& attr_it){
            if (attr_it.first[0] == '_') return 0.0;

            double children_ec = transform_reduce(PAR this->children.begin(),
                this->children.end(), 0.0, plus<>(), [&](const auto& child){
                float p_of_child = (1.0 * child->count) / this->count;
                return p_of_child * child->expected_correct_guesses(attr_it.first);
            });

            return (children_ec - this->expected_correct_guesses(attr_it.first));
    });
                
    return ec_gain / children.size();

}

inline double MultinomialCobwebNode::mutual_information() {
    if (children.empty()) {
        return 0.0;
    }

    float info_gain = transform_reduce(PAR this->av_counts.begin(), this->av_counts.end(), 0.0,
            plus<>(), [&](const auto& attr_it){
            if (attr_it.first[0] == '_') return 0.0;

            double children_info = transform_reduce(PAR this->children.begin(),
                this->children.end(), 0.0, plus<>(), [&](const auto& child){
                float p_of_child = (1.0 * child->count) / this->count;
                return p_of_child * child->entropy(attr_it.first);
            });

            return (this->entropy(attr_it.first) - children_info);
    });
                
    return info_gain / children.size();

}

inline double MultinomialCobwebNode::normalized_mutual_information() {
    if (children.empty()) {
        return 0.0;
    }

    return transform_reduce(PAR this->av_counts.begin(),
            this->av_counts.end(), 0.0, plus<>(), [&](const auto& attr_it){
            if (attr_it.first[0] == '_') return 0.0;

            double children_info = transform_reduce(PAR this->children.begin(),
            this->children.end(), 0.0, plus<>(), [&](const auto& child){
                float p_of_child = (1.0 * child->count) / this->count;
                return p_of_child * child->entropy(attr_it.first);
            });

            return ((this->entropy(attr_it.first) - children_info) /
                    this->joint_entropy(attr_it.first));
    });

}

inline double MultinomialCobwebNode::score() {
    if (this->tree->objective == CATEGORY_UTILITY){
        return this->category_utility();
    }
    else if (this->tree->objective == MUTUAL_INFORMATION){
        return this->mutual_information();
    }
    else if (this->tree->objective == NORMALIZED_MUTUAL_INFORMATION){
        return this->normalized_mutual_information();
    }
    else{
        throw "Unknown objective function.";
    }
}

inline tuple<double, string> MultinomialCobwebNode::get_best_operation(
        const AV_COUNT_TYPE &instance, MultinomialCobwebNode *best1,
        MultinomialCobwebNode *best2, double best1_score,
        bool best_op, bool new_op,
        bool merge_op, bool split_op) {

    if (best1 == NULL) {
        throw "Need at least one best child.";
    }
    vector<tuple<double, double, string>> operations;
    if (best_op){
        operations.push_back(make_tuple(best1_score,
                    custom_rand(),
                    // 3,
                    "best"));
    }
    if (new_op){
        operations.push_back(make_tuple(score_for_new_child(instance),
                    custom_rand(),
                    // 1,
                    "new"));
    }
    if (merge_op && children.size() > 2 && best2 != NULL) {
        operations.push_back(make_tuple(score_for_merge({best1, best2},
                        instance),
                    custom_rand(),
                    // 2,
                    "merge"));
    }

    if (split_op && best1->children.size() > 0) {
        operations.push_back(make_tuple(score_for_split(best1),
                    custom_rand(),
                    // 4,
                    "split"));
    }
    sort(operations.rbegin(), operations.rend());

    OPERATION_TYPE bestOp = make_pair(get<0>(operations[0]), get<2>(operations[0]));
    return bestOp;
}

inline tuple<double, MultinomialCobwebNode *, MultinomialCobwebNode *> MultinomialCobwebNode::two_best_children(const AV_COUNT_TYPE &instance) {

    if (children.empty()) {
        throw "No children!";
    }

    vector<tuple<double, double, double, MultinomialCobwebNode *>> c_scores;

    for (auto &child: this->children) {
        c_scores.push_back(
                make_tuple(
                    score_for_insert(child, instance),
                    // (child->count * child->entropy()) -
                    // ((child->count + 1) * child->entropy_insert(instance)),
                    child->count,
                    custom_rand(),
                    child));
    }

    sort(c_scores.rbegin(), c_scores.rend());

    MultinomialCobwebNode *best1 = get<3>(c_scores[0]);
    double best1_score = get<0>(c_scores[0]);
    // double best1_score = score_for_insert(best1, instance);
    MultinomialCobwebNode *best2 = c_scores.size() > 1 ? get<3>(c_scores[1]) : NULL;
    return make_tuple(best1_score, best1, best2);
}

inline double MultinomialCobwebNode::score_for_insert(MultinomialCobwebNode *child, const AV_COUNT_TYPE &instance) {
    this->increment_counts(instance);
    child->increment_counts(instance);

    double score = this->score();

    child->decrement_counts(instance);
    this->decrement_counts(instance);


    return score;
}

inline MultinomialCobwebNode* MultinomialCobwebNode::create_new_child(const AV_COUNT_TYPE &instance) {
    MultinomialCobwebNode *newChild = new MultinomialCobwebNode();
    newChild->parent = this;
    newChild->tree = this->tree;
    newChild->increment_counts(instance);
    this->children.push_back(newChild);
    return newChild;
}

inline void MultinomialCobwebNode::delete_child(MultinomialCobwebNode* child) {
    this->children.erase(remove(this->children.begin(),
                this->children.end(), child), this->children.end());
    delete child;
}

inline double MultinomialCobwebNode::score_for_new_child(const AV_COUNT_TYPE &instance) {

    this->increment_counts(instance);
    MultinomialCobwebNode* new_c = this->create_new_child(instance);

    double score = this->score();

    this->delete_child(new_c);
    this->decrement_counts(instance);

    return score;
    
}

inline MultinomialCobwebNode* MultinomialCobwebNode::merge(
        const vector<MultinomialCobwebNode*>& nodes) {
    MultinomialCobwebNode *new_child = new MultinomialCobwebNode();
    new_child->parent = this;
    new_child->tree = this->tree;

    for (MultinomialCobwebNode* node : nodes){
        new_child->update_counts_from_node(node);
        node->parent = new_child;
        new_child->children.push_back(node);
        this->children.erase(remove(this->children.begin(),
                    this->children.end(), node), this->children.end());
    }

    children.push_back(new_child);

    return new_child;
}

inline double MultinomialCobwebNode::score_for_merge(
        const vector<MultinomialCobwebNode*>& nodes,
        const AV_COUNT_TYPE &instance) {
    
    this->increment_counts(instance);
    MultinomialCobwebNode* new_c = this->merge(nodes);
    new_c->increment_counts(instance);

    double score = this->score();

    this->split(new_c);
    this->decrement_counts(instance);

    return score;    

}

inline vector<MultinomialCobwebNode*> MultinomialCobwebNode::split(MultinomialCobwebNode *best) {
    vector<MultinomialCobwebNode*> split_children = vector<MultinomialCobwebNode*>();

    for (auto &c: best->children) {
        c->parent = this;
        c->tree = this->tree;
        this->children.push_back(c);
        split_children.push_back(c);
    }

    this->delete_child(best);

    return split_children;
}

inline double MultinomialCobwebNode::score_for_split(MultinomialCobwebNode *best){

    this->children.erase(remove(this->children.begin(),
        this->children.end(), best), this->children.end());

    for (auto &c: best->children) {
        this->children.push_back(c);
    }

    double score = this->score();

    this->children.erase(this->children.end() - best->children.size(),
            this->children.end());
    this->children.push_back(best);

    return score;
    
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

inline size_t MultinomialCobwebNode::_hash() {
    return hash<uintptr_t>()(reinterpret_cast<uintptr_t>(this));
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

    // // ret += "\"_expected_guesses\": {\n";
    // ret += "\"_entropy\": {\n";
    // ret += "\"#ContinuousValue#\": {\n";
    // ret += "\"mean\": " + to_string(this->score()) + ",\n";
    // ret += "\"std\": 1,\n";
    // ret += "\"n\": 1,\n";
    // ret += "}},\n";

    // ret += "\"_mutual_info\": {\n";
    // ret += "\"#ContinuousValue#\": {\n";
    // ret += "\"mean\": " + to_string(this->mutual_information()) + ",\n";
    // ret += "\"std\": 1,\n";
    // ret += "\"n\": 1,\n";
    // ret += "}},\n";

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

    output += "\"concept_id\": " + to_string(this->_hash()) + ",\n";
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

    output += "\"name\": \"Concept" + to_string(this->_hash()) + "\",\n";
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


PYBIND11_MODULE(multinomial_cobweb, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::class_<MultinomialCobwebNode>(m, "MultinomialCobwebNode")
        .def(py::init<>())
        .def("pretty_print", &MultinomialCobwebNode::pretty_print)
        .def("output_json", &MultinomialCobwebNode::output_json)
        .def("predict", &MultinomialCobwebNode::predict, py::arg("attr") = "",
                py::arg("choiceFn") = "most likely",
                py::arg("allowNone") = true )
        .def("get_best_level", &MultinomialCobwebNode::get_best_level, py::return_value_policy::copy)
        .def("log_prob_class_given_instance", &MultinomialCobwebNode::log_prob_class_given_instance)
        .def("category_utility", &MultinomialCobwebNode::category_utility)
        .def("mutual_information", &MultinomialCobwebNode::mutual_information)
        .def("joint_entropy", &MultinomialCobwebNode::joint_entropy)
        .def("normalized_mutual_information", &MultinomialCobwebNode::normalized_mutual_information)
        .def("score", &MultinomialCobwebNode::score)
        .def("score_for_new_child", &MultinomialCobwebNode::score_for_new_child)
        .def("score_for_insert", &MultinomialCobwebNode::score_for_insert)
        .def("score_for_merge", &MultinomialCobwebNode::score_for_merge)
        .def("score_for_split", &MultinomialCobwebNode::score_for_split)
        .def("__str__", &MultinomialCobwebNode::__str__)
        .def_readonly("count", &MultinomialCobwebNode::count)
        // .def_readonly("concept_id", &MultinomialCobwebNode::concept_id)
        .def_readonly("children", &MultinomialCobwebNode::children)
        .def_readonly("parent", &MultinomialCobwebNode::parent)
        .def_readonly("av_counts", &MultinomialCobwebNode::av_counts)
        .def_readonly("attr_counts", &MultinomialCobwebNode::attr_counts)
        .def_readonly("tree", &MultinomialCobwebNode::tree);

    py::class_<MultinomialCobwebTree>(m, "MultinomialCobwebTree")
        // .def(py::init())
        .def(py::init<int, float, bool, bool, bool, bool>())
        .def("ifit", &MultinomialCobwebTree::ifit, py::return_value_policy::copy)
        .def("fit", &MultinomialCobwebTree::fit, py::arg("instances") = vector<AV_COUNT_TYPE>(), py::arg("iterations") = 1, py::arg("randomizeFirst") = true)
        .def("categorize", &MultinomialCobwebTree::categorize, py::return_value_policy::copy)
        .def("predict", &MultinomialCobwebTree::predict, py::return_value_policy::copy)
        .def("clear", &MultinomialCobwebTree::clear)
        .def("__str__", &MultinomialCobwebTree::__str__)
        .def("dump_json", &MultinomialCobwebTree::dump_json)
        .def("load_json", &MultinomialCobwebTree::load_json)
        .def_readonly("root", &MultinomialCobwebTree::root);
    //         .def_readonly("meanSq", &ContinuousValue::meanSq);
}
