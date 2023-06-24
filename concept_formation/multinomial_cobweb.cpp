#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <random>
#include <tuple>
#include <unordered_set>
#include <shared_mutex>
#include <future>

#include "assert.h"
#include "json.cpp"

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

#define NULL_STRING "\0"

typedef std::string ATTR_TYPE;
typedef std::string VALUE_TYPE;
typedef int COUNT_TYPE;
typedef std::unordered_map<VALUE_TYPE, COUNT_TYPE> VAL_COUNT_TYPE;
typedef std::unordered_map<ATTR_TYPE, VAL_COUNT_TYPE> AV_COUNT_TYPE;
typedef std::unordered_map<ATTR_TYPE, std::unordered_set<VALUE_TYPE>> AV_KEY_TYPE;
typedef std::pair<double, std::string> OPERATION_TYPE;

class MultinomialCobwebTree;
class MultinomialCobwebNode;

std::random_device rd;
std::mt19937_64 gen(rd());
std::uniform_real_distribution<double> unif(0, 1);
std::mutex cout_mutex;

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
    private:
        std::shared_mutex node_mtx;

    public:
        COUNT_TYPE count;
        std::unordered_map<ATTR_TYPE, COUNT_TYPE> attr_counts;
        std::vector<MultinomialCobwebNode *> children;
        MultinomialCobwebNode *parent;
        MultinomialCobwebTree *tree;
        AV_COUNT_TYPE av_counts;

        MultinomialCobwebNode();
        MultinomialCobwebNode(MultinomialCobwebNode *otherNode);

        void read_lock();
        void read_unlock();
        void write_lock();
        void write_unlock();
        void increment_counts(const AV_COUNT_TYPE &instance);
        void update_counts_from_node(MultinomialCobwebNode *node);
        double score_insert(const AV_COUNT_TYPE &instance);
        double score_merge(MultinomialCobwebNode *other, const AV_COUNT_TYPE &instance);
        MultinomialCobwebNode* get_best_level(const AV_COUNT_TYPE &instance);
        MultinomialCobwebNode* get_basic_level();
        double category_utility();
        double score();
        double partition_utility();
        std::tuple<double, std::string> get_best_operation(const AV_COUNT_TYPE &instance, MultinomialCobwebNode *best1,
                MultinomialCobwebNode *best2, double best1Cu,
                bool best_op=true, bool new_op=true,
                bool merge_op=true, bool split_op=true);
        std::tuple<double, MultinomialCobwebNode *, MultinomialCobwebNode *> two_best_children(const AV_COUNT_TYPE &instance);
        double log_prob_class_given_instance(const AV_COUNT_TYPE &instance);
        double pu_for_insert(MultinomialCobwebNode *child, const AV_COUNT_TYPE &instance);
        // MultinomialCobwebNode *create_new_child(const AV_COUNT_TYPE &instance);
        double pu_for_new_child(const AV_COUNT_TYPE &instance);
        // MultinomialCobwebNode *merge(MultinomialCobwebNode *best1, MultinomialCobwebNode *best2);
        double pu_for_merge(MultinomialCobwebNode *best1, MultinomialCobwebNode *best2, const AV_COUNT_TYPE &instance);
        // void split(MultinomialCobwebNode *best);
        double pu_for_split(MultinomialCobwebNode *best);
        bool is_exact_match(const AV_COUNT_TYPE &instance);
        size_t _hash();
        std::string __str__();
        std::string pretty_print(int depth = 0);
        int depth();
        bool is_parent(MultinomialCobwebNode *otherConcept);
        int num_concepts();
        std::string avcounts_to_json();
        std::string ser_avcounts();
        std::string attr_counts_to_json();
        std::string dump_json();
        std::string output_json();
        std::vector<std::tuple<VALUE_TYPE, double>> get_weighted_values(ATTR_TYPE attr, bool allowNone = true);
        VALUE_TYPE predict(ATTR_TYPE attr, std::string choiceFn = "most likely", bool allowNone = true);
        double probability(ATTR_TYPE attr, VALUE_TYPE val);
        double log_likelihood(MultinomialCobwebNode *childLeaf);

};


class CategorizationFuture {
    private:
        std::future<MultinomialCobwebNode*> leaf_future;
        MultinomialCobwebNode* leaf;
        AV_COUNT_TYPE instance;
    public:
        CategorizationFuture(std::future<MultinomialCobwebNode*> leaf_future, const AV_COUNT_TYPE &instance);
        CategorizationFuture(MultinomialCobwebNode* leaf_node, const AV_COUNT_TYPE &instance);
        std::unordered_map<ATTR_TYPE, std::unordered_map<VALUE_TYPE, double>> predict();

};


class MultinomialCobwebTree {
    private:
        std::shared_mutex av_keys_mtx;
        std::shared_mutex root_ptr_mtx;
        AV_KEY_TYPE attr_vals;

    public:
        bool use_mutual_info;
        float alpha_weight;
        bool dynamic_alpha;
        bool weight_attr;
        MultinomialCobwebNode *root;

        MultinomialCobwebTree(bool use_mutual_info, float alpha_weight, bool
                dynamic_alpha, bool weight_attr) {
            this->use_mutual_info = use_mutual_info;
            this->alpha_weight = alpha_weight;
            this->dynamic_alpha = dynamic_alpha;
            this->weight_attr = weight_attr;

            // TODO do we need to worry about thread safety here?
            this->root = new MultinomialCobwebNode();
            this->root->tree = this;
            this->attr_vals = AV_KEY_TYPE();
        }

        std::string __str__(){
            return this->root->__str__();
        }

        float alpha(std::string attr){
            if (!this->dynamic_alpha){
                return this->alpha_weight;
            }

            COUNT_TYPE n_vals = num_vals_for_attr(attr);

            if (n_vals == 0){
                return this->alpha_weight;
            } else {
                return this->alpha_weight / n_vals;
            }
        }

        float attr_weight(ATTR_TYPE attr){
            return (1.0 * get_root()->attr_counts.at(attr)) / get_root()->count;
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
            int count_val = atoi(json_value_as_number(count_obj->value)->number);
            new_node->count = count_val;

            // Get attr_counts
            struct json_object_element_s* attr_counts_obj = count_obj->next;
            struct json_object_s* attr_counts_dict = json_value_as_object(attr_counts_obj->value);
            struct json_object_element_s* attr_counts_cursor = attr_counts_dict->start;
            while(attr_counts_cursor != NULL) {
                // Get attr name
                std::string attr_name = std::string(attr_counts_cursor->name->string);

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
                std::string attr_name = std::string(av_counts_cursor->name->string);

                // The attr val is a dict of strings to ints
                struct json_object_s* attr_val_dict = json_value_as_object(av_counts_cursor->value);
                struct json_object_element_s* inner_counts_cursor = attr_val_dict->start;
                while(inner_counts_cursor != NULL) {
                    // this will be a word
                    std::string val_name = std::string(inner_counts_cursor->name->string);
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
            return get_root()->dump_json();
        }

        // TODO, does this need to be made thread safe?
        void load_json(std::string json) {
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

        // TODO thread safety?
        void clear() {
            delete this->root;
            this->root = new MultinomialCobwebNode();
            this->root->tree = this;
            this->attr_vals = AV_KEY_TYPE();
        }

        CategorizationFuture* async_ifit(AV_COUNT_TYPE &instance) {
            auto fut_result = std::async(std::launch::async, [this, instance]() {
                auto* result = this->ifit_helper(instance);
                return result;
            });

            return new CategorizationFuture(std::move(fut_result), instance);
        }

        bool has_attr(ATTR_TYPE attr){
            std::shared_lock<std::shared_mutex> lock(av_keys_mtx);
            return attr_vals.count(attr);
        };

        bool has_attr_val(ATTR_TYPE attr, VALUE_TYPE val){
            std::shared_lock<std::shared_mutex> lock(av_keys_mtx);
            return attr_vals.count(attr) && attr_vals.at(attr).count(val);
        };

        void add_attr_val(ATTR_TYPE attr, VALUE_TYPE val){
            std::unique_lock<std::shared_mutex> lock(av_keys_mtx);
            attr_vals[attr].insert(val);
        }

        int num_vals_for_attr(ATTR_TYPE attr){
            std::shared_lock<std::shared_mutex> lock(av_keys_mtx);
            return attr_vals.at(attr).size();        
        }

        void copy_attr_vals(AV_KEY_TYPE& dest){
            std::shared_lock<std::shared_mutex> lock(av_keys_mtx);
            for (auto &[attr, val_set]: attr_vals) {
                for (auto val: val_set) {
                    dest[attr].insert(val);
                }
            }
        }

        MultinomialCobwebNode* get_root(){
            // std::shared_lock<std::shared_mutex> lock(root_ptr_mtx);
            return root; 
        }

        void set_root(MultinomialCobwebNode* node){
            // std::unique_lock<std::shared_mutex> lock(root_ptr_mtx);
            root = node; 
        }

        MultinomialCobwebNode* ifit_helper(const AV_COUNT_TYPE &instance){
            for (auto &[attr, val_map]: instance) {
                // if (attr[0] == '_') continue;
                for (auto &[val, cnt]: val_map) {
                    if (!has_attr_val(attr, val)){
                        add_attr_val(attr, val);
                    }
                }
            }
            return this->cobweb(instance);
        }

        CategorizationFuture* ifit(AV_COUNT_TYPE instance) {
            return new CategorizationFuture(this->ifit_helper(instance), instance);
        }

        void fit(std::vector<AV_COUNT_TYPE> instances, int iterations = 1, bool randomizeFirst = true) {
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

        MultinomialCobwebNode *cobweb(const AV_COUNT_TYPE &instance) {
            std::cout << "cobweb top level" << std::endl;
            root_ptr_mtx.lock();
            std::cout << "locked root ptr" << std::endl;

            MultinomialCobwebNode* current = root;
            std::cout << "locking root concept" << std::endl;
            current->write_lock();

            std::cout << "cobweb entering loop" << std::endl;

            while (true) {
                // each loop starts with a write lock on current and
                // current->parent (in the case of root, root_ptr_mtx is write
                // locked instead of current->parent).
                if (current->children.empty() && (current->is_exact_match(instance) || current->count == 0)) {
                    std::cout << "empty / exact match" << std::endl;
                    if (current->parent == nullptr) {
                        root_ptr_mtx.unlock();
                        std::cout << "unlocked root ptr" << std::endl;
                    } else {
                        current->parent->write_unlock();
                    }
                    current->increment_counts(instance);
                    current->write_unlock();
                    break;
                } else if (current->children.empty()) {
                    std::cout << "fringe split" << std::endl;
                    MultinomialCobwebNode* new_node = new MultinomialCobwebNode(current);
                    new_node->write_lock();
                    current->parent = new_node;
                    new_node->children.push_back(current);

                    if (new_node->parent == nullptr) {
                        root = new_node;
                        root_ptr_mtx.unlock();
                        std::cout << "unlocked root ptr" << std::endl;
                    }
                    else{
                        new_node->parent->children.erase(remove(new_node->parent->children.begin(),
                            new_node->parent->children.end(), current), new_node->parent->children.end());
                        new_node->parent->children.push_back(new_node);
                        new_node->parent->write_unlock();
                    }
                    current->write_unlock();
                    new_node->increment_counts(instance);
                    
                    current = new MultinomialCobwebNode();
                    current->write_lock();
                    current->parent = new_node;
                    current->tree = this;
                    current->increment_counts(instance);
                    new_node->children.push_back(current);
                    new_node->write_unlock();
                    current->write_unlock();
                    break;

                } else {

                    // read lock children
                    for (auto &c: current->children) {
                        c->read_lock();
                    }
                    auto[best1_mi, best1, best2] = current->two_best_children(instance);

                    // lock best1's children for evaluating split
                    for (auto &c: best1->children){
                        c->read_lock();
                    }
                    auto[_, bestAction] = current->get_best_operation(instance, best1, best2, best1_mi);

                    for (auto &c: best1->children){
                        c->read_unlock();
                    }
                    for (auto &c: current->children) {
                        c->read_unlock();
                    }

                    if (bestAction != "split"){
                        if (current->parent == nullptr){
                            root_ptr_mtx.unlock();
                            std::cout << "unlocked root ptr" << std::endl;
                        }
                        else{
                            current->parent->write_unlock();
                        } 
                    }

                    if (bestAction == "best") {
                        std::cout << "best" << std::endl;
                        current->increment_counts(instance);
                        // TODO should explore an "upgrade lock" on best1
                        best1->write_lock();
                        current = best1;
                    } else if (bestAction == "new") {
                        std::cout << "new" << std::endl;
                        current->increment_counts(instance);

                        // current = current->create_new_child(instance);
                        MultinomialCobwebNode *new_child = new MultinomialCobwebNode();
                        new_child->parent = current;
                        new_child->tree = this;
                        new_child->increment_counts(instance);
                        current->children.push_back(new_child);
                        current->write_unlock();
                        current = new_child;
                        break;
                    } else if (bestAction == "merge") {
                        std::cout << "merge" << std::endl;
                        current->increment_counts(instance);
                        // MultinomialCobwebNode* new_child = current->merge(best1, best2);

                        MultinomialCobwebNode *new_child = new MultinomialCobwebNode();
                        new_child->write_lock();
                        new_child->parent = current;
                        new_child->tree = this;

                        // TODO should explore an "upgrade lock" on best1 and best2.
                        best1->write_lock();
                        best2->write_lock();
                        new_child->update_counts_from_node(best1);
                        new_child->update_counts_from_node(best2);
                        best1->parent = new_child;
                        best2->parent = new_child;
                        best1->write_unlock();
                        best2->write_unlock();
                        new_child->children.push_back(best1);
                        new_child->children.push_back(best2);
                        current->children.erase(remove(current->children.begin(),
                                    current->children.end(), best1), current->children.end());
                        current->children.erase(remove(current->children.begin(),
                                    current->children.end(), best2), current->children.end());
                        current->children.push_back(new_child);
                        current = new_child;
                    } else if (bestAction == "split") {
                        std::cout << "split" << std::endl;
                        // current->split(best1);
                        current->children.erase(remove(current->children.begin(),
                            current->children.end(), best1), current->children.end());
                        for (auto &c: best1->children) {
                            c->write_lock();
                            c->parent = current;
                            c->tree = this;
                            current->children.push_back(c);
                            c->write_unlock();
                        }
                        best1->write_lock();
                        delete best1;

                    } else {
                        throw "Best action choice \"" + bestAction +
                            "\" not a recognized option. This should be impossible...";
                    }
                }
            }
            return current;
        }

        // NOT modifying so only need to lock one node up rather than two; this is because
        // split might bump the node up higher, but when getting considered for splitting 
        // there will be no changes to the node as a result of categorize, so we're good.
        MultinomialCobwebNode* _cobweb_categorize(const AV_COUNT_TYPE &instance, bool get_best_concept) {
            auto current = get_root();

            // TODO the best node might get deleted, not threadsafe!!!
            auto best_concept = current;
            double best_score = best_concept->log_prob_class_given_instance(instance);

            while (true) {
                if (current->children.empty()) {
                    if (get_best_concept) return best_concept;
                    return current;
                }

                auto parent = current;
                current = nullptr;
                double best_logp = 0.0;

                for (auto &child: parent->children) {
                    double logp = child->log_prob_class_given_instance(instance);
                    if (current == nullptr || logp > best_logp){
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

        MultinomialCobwebNode* categorize(const AV_COUNT_TYPE &instance, bool get_best_concept) {
            return this->_cobweb_categorize(instance, get_best_concept);
        }

};

inline CategorizationFuture::CategorizationFuture(std::future<MultinomialCobwebNode*> leaf_future, const AV_COUNT_TYPE &instance): 
    leaf_future(std::move(leaf_future)), leaf(nullptr), instance(instance) {}

inline CategorizationFuture::CategorizationFuture(MultinomialCobwebNode* leaf_node, const AV_COUNT_TYPE &instance):
    leaf_future(), leaf(leaf_node), instance(instance) {}

inline std::unordered_map<ATTR_TYPE, std::unordered_map<VALUE_TYPE, double>> CategorizationFuture::predict(){
    std::unordered_map<ATTR_TYPE, std::unordered_map<VALUE_TYPE, double>> out;

    if (leaf == nullptr){
        leaf = leaf_future.get();
    }

    leaf->read_lock();

    AV_KEY_TYPE attr_vals;
    leaf->tree->copy_attr_vals(attr_vals);

    for (auto &[attr, val_set]: attr_vals) {
        // std::cout << attr << std::endl;
        float alpha = leaf->tree->alpha(attr);
        int num_vals = leaf->tree->num_vals_for_attr(attr);
        COUNT_TYPE attr_count = 0;

        if (leaf->attr_counts.count(attr)){
            attr_count = leaf->attr_counts.at(attr);
        }

        for (auto val: val_set) {
            // std::cout << val << std::endl;
            COUNT_TYPE av_count = 0;
            if (leaf->av_counts.count(attr) and leaf->av_counts.at(attr).count(val)){
                av_count = leaf->av_counts.at(attr).at(val);
            }

            double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
            // std::cout << p << std::endl;
            out[attr][val] += p;
        }
    }

    leaf->read_unlock();
    return out;
}


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

inline void MultinomialCobwebNode::read_lock(){
    node_mtx.lock_shared();
    std::cout << "read locked: " << this << std::endl;
}

inline void MultinomialCobwebNode::read_unlock(){
    node_mtx.unlock_shared();
    std::cout << "read unlocked: " << this << std::endl;
}

inline void MultinomialCobwebNode::write_lock(){
    node_mtx.lock();
    std::cout << "write locked: " << this << std::endl;
}

inline void MultinomialCobwebNode::write_unlock(){
    node_mtx.unlock();
    std::cout << "write unlocked: " << this << std::endl;
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
    std::unordered_set<ATTR_TYPE> all_attrs;
    for (auto &[attr, tmp]: instance){
        if (attr[0] == '_') continue;
        all_attrs.insert(attr);
    }
    for (auto &[attr, tmp]: this->av_counts){
        if (attr[0] == '_') continue;
        all_attrs.insert(attr);
    }

    return transform_reduce(PAR all_attrs.begin(), all_attrs.end(), 0.0,
            std::plus<>(), [&](const auto& attr_it){
                COUNT_TYPE attr_count = 0;
                std::unordered_set<VALUE_TYPE> all_vals;
                float alpha = this->tree->alpha(attr_it);
                int num_vals = this->tree->num_vals_for_attr(attr_it);

                if (this->av_counts.count(attr_it)){
                    attr_count += this->attr_counts.at(attr_it);
                    for (auto &[val, cnt]: this->av_counts.at(attr_it)) all_vals.insert(val);
                }
                if (instance.count(attr_it)){
                    for (auto &[val, cnt]: instance.at(attr_it)){
                    attr_count += cnt;
                    all_vals.insert(val);
                    }
                }

                double ratio = 1.0;

                if (this->tree->weight_attr){
                    ratio = (1.0 * attr_count) / (this->count + 1);
                }

                double info = transform_reduce(PAR all_vals.begin(), all_vals.end(), 0.0,
                        std::plus<>(), [&](const auto &val){
                            COUNT_TYPE av_count = 0;

                            if (this->av_counts.count(attr_it) and this->av_counts.at(attr_it).count(val)){
                                av_count += this->av_counts.at(attr_it).at(val);
                            }
                            if (instance.count(attr_it) and instance.at(attr_it).count(val)){
                                av_count += instance.at(attr_it).at(val);
                            }

                            double p = ((av_count + alpha) / (attr_count + num_vals * alpha));

                            if (this->tree->use_mutual_info){
                                return ratio * -p * log(p);
                            } else{
                                return ratio * -p * p;
                            }
                        });

                COUNT_TYPE num_missing = num_vals - all_vals.size();
                if (num_missing > 0 and alpha > 0){
                    double p = (alpha / (attr_count + num_vals * alpha));
                    if (this->tree->use_mutual_info){
                        info += num_missing * ratio * -p * log(p);
                    } else {
                        info += num_missing * ratio * -p * p;
                    }
                }

                return info;
            });
}

inline double MultinomialCobwebNode::score_merge(MultinomialCobwebNode *other, const AV_COUNT_TYPE &instance) {

    std::unordered_set<ATTR_TYPE> all_attrs;
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

    return transform_reduce(PAR all_attrs.begin(), all_attrs.end(), 0.0,
            std::plus<>(), [&](const auto& attr_it){
                COUNT_TYPE attr_count = 0;
                std::unordered_set<ATTR_TYPE> all_vals;
                float alpha = this->tree->alpha(attr_it);
                int num_vals = this->tree->num_vals_for_attr(attr_it);

                if (this->av_counts.count(attr_it)){
                    attr_count += this->attr_counts.at(attr_it);
                    for (auto &[val, cnt]: this->av_counts.at(attr_it)) all_vals.insert(val);
                }
                if (other->av_counts.count(attr_it)){
                    attr_count += other->attr_counts.at(attr_it);
                    for (auto &[val, cnt]: other->av_counts.at(attr_it)) all_vals.insert(val);
                }
                if (instance.count(attr_it)){
                    for (auto &[val, cnt]: instance.at(attr_it)){
                        attr_count += cnt;
                        all_vals.insert(val);
                    }
                }

                double ratio = 1.0;

                if (this->tree->weight_attr){
                    ratio = (1.0 * attr_count) / (this->count + other->count + 1);
                }

                double info = transform_reduce(PAR all_vals.begin(), all_vals.end(), 0.0,
                        std::plus<>(), [&](const auto &val){
                            COUNT_TYPE av_count = 0;

                            if (this->av_counts.count(attr_it) and this->av_counts.at(attr_it).count(val)){
                                av_count += this->av_counts.at(attr_it).at(val);
                            }
                            if (other->av_counts.count(attr_it) and other->av_counts.at(attr_it).count(val)){
                                av_count += other->av_counts.at(attr_it).at(val);
                            }
                            if (instance.count(attr_it) and instance.at(attr_it).count(val)){
                                av_count += instance.at(attr_it).at(val);
                            }

                            double p = ((av_count + alpha) / (attr_count + num_vals * alpha));

                            if (this->tree->use_mutual_info){
                                return ratio * -p * log(p);
                            } else {
                                return ratio * -p * p;
                            }
                        });

                COUNT_TYPE num_missing = num_vals - all_vals.size();
                if (num_missing > 0 and alpha > 0){
                    double p = (alpha / (attr_count + num_vals * alpha));
                    if (this->tree->use_mutual_info){
                        info += num_missing * ratio * -p * log(p);
                    } else {
                        info += num_missing * ratio * -p * p;
                    }
                }

                return info;
            });
}

inline MultinomialCobwebNode* MultinomialCobwebNode::get_best_level(const AV_COUNT_TYPE &instance){
    MultinomialCobwebNode* curr = this;
    MultinomialCobwebNode* best = this;
    double best_ll = this->log_prob_class_given_instance(instance);

    while (curr->parent != nullptr) {
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

    return transform_reduce(PAR this->av_counts.begin(), this->av_counts.end(), 0.0,
            std::plus<>(), [&](const auto& attr_it){
                if (attr_it.first[0] == '_') return 0.0;

                COUNT_TYPE attr_count = this->attr_counts.at(attr_it.first);
                int num_vals = this->tree->num_vals_for_attr(attr_it.first);
                float alpha = this->tree->alpha(attr_it.first);

                double ratio = 1.0;

                if (this->tree->weight_attr){
                    ratio = (1.0 * attr_count) / this->count;
                }

                double info = transform_reduce(PAR attr_it.second.begin(), attr_it.second.end(), 0.0,
                        std::plus<>(), [&](const auto& val_it){
                            double p = ((val_it.second + alpha) / (attr_count + num_vals * alpha));
                            if (this->tree->use_mutual_info){
                                return ratio * -p * log(p);
                            } else {
                                return ratio * -p * p;
                            }
                        });

                COUNT_TYPE num_missing = num_vals - attr_it.second.size();
                if (num_missing > 0 and alpha > 0){
                    double p = (alpha / (attr_count + num_vals * alpha));
                    if (this->tree->use_mutual_info){
                        info += num_missing * ratio * -p * log(p);
                    } else {
                        info += num_missing * ratio * -p * p;
                    }
                }

                return info;
            });
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

inline std::tuple<double, std::string> MultinomialCobwebNode::get_best_operation(
        const AV_COUNT_TYPE &instance, MultinomialCobwebNode *best1,
        MultinomialCobwebNode *best2, double best1_pu,
        bool best_op, bool new_op,
        bool merge_op, bool split_op) {

    if (best1 == nullptr) {
        throw "Need at least one best child.";
    }
    std::vector<std::tuple<double, double, std::string>> operations;
    if (best_op){
        operations.push_back(std::make_tuple(best1_pu,
                    custom_rand(),
                    "best"));
    }
    if (new_op){
        operations.push_back(std::make_tuple(pu_for_new_child(instance),
                    custom_rand(),
                    "new"));
    }
    if (merge_op && children.size() > 2 && best2 != nullptr) {
        operations.push_back(std::make_tuple(pu_for_merge(best1, best2,
                        instance),
                    custom_rand(),
                    "merge"));
    }

    if (split_op && best1->children.size() > 0) {
        operations.push_back(std::make_tuple(pu_for_split(best1),
                    custom_rand(),
                    "split"));
    }
    sort(operations.rbegin(), operations.rend());

    OPERATION_TYPE bestOp = make_pair(std::get<0>(operations[0]), std::get<2>(operations[0]));
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

/*
inline MultinomialCobwebNode* MultinomialCobwebNode::create_new_child(const AV_COUNT_TYPE &instance) {
    MultinomialCobwebNode *new_child = new MultinomialCobwebNode();
    new_child->parent = this;
    new_child->tree = this->tree;
    new_child->increment_counts(instance);
    this->children.push_back(new_child);
    return new_child;
};
*/

inline double MultinomialCobwebNode::pu_for_new_child(const AV_COUNT_TYPE &instance) {
    double children_score = 0.0;

    for (auto &c: this->children) {
        double p_of_child = (1.0 * c->count) / (this->count + 1.0);
        children_score += p_of_child * c->score();
    }

    MultinomialCobwebNode new_child = MultinomialCobwebNode();
    new_child.parent = this;
    new_child.tree = this->tree;
    new_child.increment_counts(instance);
    double p_of_child = 1.0 / (this->count + 1.0);
    children_score += p_of_child * new_child.score();

    return ((this->score_insert(instance) - children_score) /
            (children.size()+1));
}

/*
inline MultinomialCobwebNode* MultinomialCobwebNode::merge(MultinomialCobwebNode *best1, MultinomialCobwebNode *best2) {
    MultinomialCobwebNode *new_child = new MultinomialCobwebNode();
    new_child->write_lock();
    new_child->parent = this;
    new_child->tree = this->tree;

    best1->write_lock();
    best2->write_lock();
    new_child->update_counts_from_node(best1);
    new_child->update_counts_from_node(best2);
    best1->parent = new_child;
    best2->parent = new_child;
    new_child->children.push_back(best1);
    new_child->children.push_back(best2);
    children.erase(remove(this->children.begin(), this->children.end(), best1), children.end());
    children.erase(remove(this->children.begin(), this->children.end(), best2), children.end());
    children.push_back(new_child);
    best1->write_unlock();
    best2->write_unlock();

    return new_child;
}
*/

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

/*
inline void MultinomialCobwebNode::split(MultinomialCobwebNode *best) {
    children.erase(remove(children.begin(), children.end(), best), children.end());
    for (auto &c: best->children) {
        c->parent = this;
        c->tree = this->tree;
        children.push_back(c);
    }
    delete best;
}
*/

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
        if (attr[0] == '_') continue;
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
        ret += "\"" + attr + "\": {";
        int inner_count = 0;
        for (auto &[val, cnt]: vAttr) {
            ret += "\"" + val + "\": " + std::to_string(cnt);
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
        ret += "\"" + attr + "\": {";
        int inner_count = 0;
        for (auto &[val, cnt]: vAttr) {
            ret += "\"" + val + "\": " + std::to_string(cnt);
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
        ret += "\"" + attr + "\": " + std::to_string(cnt);
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

    for (auto &c: children) {
        output += c->output_json() + ",";
    }

    output += "],\n";
    output += "\"counts\": " + this->avcounts_to_json() + "\n";

    output += "}\n";

    return output;
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

inline double MultinomialCobwebNode::log_likelihood(MultinomialCobwebNode *childLeaf) {
    std::unordered_set<ATTR_TYPE> allAttrs;
    for (auto &[attr, tmp]: this->av_counts) allAttrs.insert(attr);
    for (auto &[attr, tmp]: childLeaf->av_counts) allAttrs.insert(attr);

    double ll = 0;

    for (auto &attr: allAttrs) {
        if (attr[0] == '_') continue;
        std::unordered_set<VALUE_TYPE> vals;
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

inline double MultinomialCobwebNode::category_utility(){
    double p_of_c = (1.0 * this->count) / this->tree->get_root()->count;
    return (p_of_c * (this->tree->get_root()->score() - this->score()));
}

// TODO need to make thread safe...
inline double MultinomialCobwebNode::log_prob_class_given_instance(const AV_COUNT_TYPE &instance){

    double log_prob = 0;

    for (auto &[attr, vAttr]: instance) {
        bool hidden = attr[0] == '_';
        // if (hidden || !this->tree->root->av_counts.count(attr)){
        if (hidden || !this->tree->has_attr(attr)){
            continue;
        }

        double num_vals = this->tree->num_vals_for_attr(attr);
        // double num_vals = this->tree->root->av_counts.at(attr).size();

        for (auto &[val, cnt]: vAttr){
            // if (!this->tree->root->av_counts.at(attr).count(val)){
            if (!this->tree->has_attr_val(attr, val)){
                continue;
            }

            float av_count = this->tree->alpha(attr);
            if (this->av_counts.count(attr) && this->av_counts.at(attr).count(val)){
                av_count += this->av_counts.at(attr).at(val);
                // std::cout << val << "(" << this->av_counts.at(attr).at(val) << ") ";
            }

            // the cnt here is because we have to compute probability over all context words.
            log_prob += cnt * log((1.0 * av_count) / (this->attr_counts[attr] + num_vals * this->tree->alpha(attr)));
        }

        // std::cout << std::endl;
        // std::cout << "denom: " << std::to_string(this->counts[attr] + num_vals * this->tree->alpha) << std::endl;
        // std::cout << "denom (no alpha): " << std::to_string(this->counts[attr]) << std::endl;
        // std::cout << "node count: " << std::to_string(this->count) << std::endl;
        // std::cout << "num vals: " << std::to_string(num_vals) << std::endl;
    }

    // std::cout << std::endl;

    log_prob += log((1.0 * this->count) / this->tree->get_root()->count);

    return log_prob;
}


PYBIND11_MODULE(multinomial_cobweb, m) {
    m.doc() = "concept_formation.multinomial_cobweb plugin"; // optional module docstring

    py::class_<CategorizationFuture>(m, "CategorizationFuture")
        .def("predict", &CategorizationFuture::predict, py::call_guard<py::gil_scoped_release>());

    py::class_<MultinomialCobwebNode>(m, "MultinomialCobwebNode")
        .def(py::init<>())
        .def("pretty_print", &MultinomialCobwebNode::pretty_print)
        .def("output_json", &MultinomialCobwebNode::output_json)
        .def("predict", &MultinomialCobwebNode::predict, py::arg("attr") = "",
                py::arg("choiceFn") = "most likely",
                py::arg("allowNone") = true )
        .def("get_best_level", &MultinomialCobwebNode::get_best_level)
        .def("get_basic_level", &MultinomialCobwebNode::get_basic_level)
        .def("log_prob_class_given_instance", &MultinomialCobwebNode::log_prob_class_given_instance)
        .def("score", &MultinomialCobwebNode::score)
        .def("category_utility", &MultinomialCobwebNode::category_utility)
        .def("partition_utility", &MultinomialCobwebNode::partition_utility)
        .def("__str__", &MultinomialCobwebNode::__str__)
        .def_readonly("count", &MultinomialCobwebNode::count)
        .def_readonly("children", &MultinomialCobwebNode::children)
        .def_readonly("parent", &MultinomialCobwebNode::parent)
        .def_readonly("av_counts", &MultinomialCobwebNode::av_counts)
        .def_readonly("attr_counts", &MultinomialCobwebNode::attr_counts)
        .def_readonly("tree", &MultinomialCobwebNode::tree);

    py::class_<MultinomialCobwebTree>(m, "MultinomialCobwebTree")
        .def(py::init<bool, float, bool, bool>(), py::arg("use_mutual_info") = true, 
                py::arg("alpha_weight") = 1.0,
                py::arg("dynamic_alpha") = true,
                py::arg("weight_attr") = true)
        .def_readonly("root", &MultinomialCobwebTree::root, py::return_value_policy::reference)
        .def("async_ifit", &MultinomialCobwebTree::async_ifit, py::call_guard<py::gil_scoped_release>())
        .def("ifit", &MultinomialCobwebTree::ifit)
        .def("fit", &MultinomialCobwebTree::fit,
                py::arg("instances") = std::vector<AV_COUNT_TYPE>(),
                py::arg("iterations") = 1,
                py::arg("randomizeFirst") = true)
        .def("categorize", &MultinomialCobwebTree::categorize,
                py::arg("instance") = std::vector<AV_COUNT_TYPE>(),
                py::arg("get_best_concept") = true, py::return_value_policy::reference)
        .def("clear", &MultinomialCobwebTree::clear)
        .def("__str__", &MultinomialCobwebTree::__str__)
        .def("dump_json", &MultinomialCobwebTree::dump_json)
        .def("load_json", &MultinomialCobwebTree::load_json);
}
