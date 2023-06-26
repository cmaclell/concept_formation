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
#include <chrono>

#include "assert.h"
#include "json.hpp"
#include "BS_thread_pool.hpp"
#include "cached_string.hpp"

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

#define NULL_STRING CachedString("\0")

typedef CachedString ATTR_TYPE;
typedef CachedString VALUE_TYPE;
typedef int COUNT_TYPE;
typedef std::unordered_map<std::string, std::unordered_map<std::string, COUNT_TYPE>> INSTANCE_TYPE;
typedef std::unordered_map<VALUE_TYPE, COUNT_TYPE> VAL_COUNT_TYPE;
typedef std::unordered_map<ATTR_TYPE, VAL_COUNT_TYPE> AV_COUNT_TYPE;
typedef std::unordered_map<ATTR_TYPE, std::unordered_set<VALUE_TYPE>> AV_KEY_TYPE;
typedef std::unordered_map<ATTR_TYPE, int> VAL_COUNTS_TYPE;
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
        double read_wait_time = 0.0;
        double write_wait_time = 0.0;

        MultinomialCobwebNode();
        MultinomialCobwebNode(MultinomialCobwebNode *otherNode);

        void read_lock();
        void read_unlock();
        void write_lock();
        void write_unlock();
        void increment_counts(const AV_COUNT_TYPE &instance);
        void update_counts_from_node(MultinomialCobwebNode *node);
        double score_insert(const AV_COUNT_TYPE &instance, const VAL_COUNTS_TYPE &val_counts);
        double score_merge(MultinomialCobwebNode *other, const AV_COUNT_TYPE &instance, const VAL_COUNTS_TYPE &val_counts);
        MultinomialCobwebNode* get_best_level(const AV_COUNT_TYPE &instance, const AV_KEY_TYPE &av_keys);
        MultinomialCobwebNode* get_basic_level(const VAL_COUNTS_TYPE &val_counts);
        double category_utility(const VAL_COUNTS_TYPE &val_counts);
        double score(const VAL_COUNTS_TYPE &val_counts);
        double partition_utility(const VAL_COUNTS_TYPE &val_counts);
        std::tuple<double, std::string> get_best_operation(const AV_COUNT_TYPE &instance, MultinomialCobwebNode *best1,
                MultinomialCobwebNode *best2, double best1Cu, const VAL_COUNTS_TYPE &val_counts);
        std::tuple<double, MultinomialCobwebNode *, MultinomialCobwebNode *> two_best_children(const AV_COUNT_TYPE &instance,
                const VAL_COUNTS_TYPE &val_counts);
        double log_prob_class_given_instance(const AV_COUNT_TYPE &instance, const AV_KEY_TYPE &av_keys);
        double pu_for_insert(MultinomialCobwebNode *child, const AV_COUNT_TYPE &instance, const VAL_COUNTS_TYPE &val_counts);
        // MultinomialCobwebNode *create_new_child(const AV_COUNT_TYPE &instance);
        double pu_for_new_child(const AV_COUNT_TYPE &instance, const VAL_COUNTS_TYPE &val_counts);
        // MultinomialCobwebNode *merge(MultinomialCobwebNode *best1, MultinomialCobwebNode *best2);
        double pu_for_merge(MultinomialCobwebNode *best1, MultinomialCobwebNode *best2, const AV_COUNT_TYPE &instance, 
                const VAL_COUNTS_TYPE &val_counts);
        // void split(MultinomialCobwebNode *best);
        double pu_for_split(MultinomialCobwebNode *best, const VAL_COUNTS_TYPE &val_counts);
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
    public:
        CategorizationFuture(std::future<MultinomialCobwebNode*> leaf_future);
        CategorizationFuture(MultinomialCobwebNode* leaf_node);
        void wait();
        std::unordered_map<ATTR_TYPE, std::unordered_map<VALUE_TYPE, double>> predict();

};


class MultinomialCobwebTree {
    private:
        std::shared_mutex tree_mtx;
        std::shared_mutex av_key_mtx;
        BS::thread_pool pool{(std::thread::hardware_concurrency() - 1)};
        // BS::thread_pool pool{3};

    public:
        bool use_mutual_info;
        float alpha_weight;
        bool dynamic_alpha;
        bool weight_attr;
        MultinomialCobwebNode *root;
        AV_KEY_TYPE attr_vals;
        double av_key_wait_time = 0.0;
        double write_wait_time = 0.0;

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

        void read_lock_av_key(){
            av_key_mtx.lock_shared();
        }

        void read_unlock_av_key(){
            av_key_mtx.unlock_shared();
        }

        void write_lock_av_key(){
            auto start_time = std::chrono::high_resolution_clock::now();
            av_key_mtx.lock();
            auto stop_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
            av_key_wait_time += duration;
        }

        void write_unlock_av_key(){
            av_key_mtx.unlock();
        }

        void read_lock_tree_ptr(){
            tree_mtx.lock_shared();
        }

        void read_unlock_tree_ptr(){
            tree_mtx.unlock_shared();
        }

        void write_lock_tree_ptr(){
            auto start_time = std::chrono::high_resolution_clock::now();
            tree_mtx.lock(); 
            // std::cout << "write tree locked" << std::endl;
            auto stop_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
            write_wait_time += duration;
        }

        void write_unlock_tree_ptr(){
            tree_mtx.unlock();
            // std::cout << "write tree unlocked" << std::endl;
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

        /*
        float attr_weight(ATTR_TYPE attr){
            return (1.0 * get_root()->attr_counts.at(attr)) / get_root()->count;
        }
        */

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

        MultinomialCobwebNode* get_root(){
            return root; 
        }

        void set_root(MultinomialCobwebNode* node){
            root = node; 
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

        CategorizationFuture* ifit(INSTANCE_TYPE instance) {
            return new CategorizationFuture(this->ifit_helper(instance));
        }

        CategorizationFuture* async_ifit(INSTANCE_TYPE &instance) {
            auto fut_result = pool.submit([this, instance]() {
                auto* result = this->ifit_helper(instance);
                return result;
            });

            return new CategorizationFuture(std::move(fut_result));
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

        MultinomialCobwebNode *cobweb(const AV_COUNT_TYPE &instance) {
            // std::cout << "cobweb top level" << std::endl;

            for (auto &[attr, val_map]: instance) {
                // if (attr[0] == '_') continue;
                for (auto &[val, cnt]: val_map) {
                    this->read_lock_av_key();
                    bool needs_write = !attr_vals.count(attr) or !attr_vals.at(attr).count(val);
                    this->read_unlock_av_key();

                    if (needs_write){
                        this->write_lock_av_key();
                        attr_vals[attr].insert(val);
                        this->write_unlock_av_key();
                    }
                }
            }

            this->write_lock_tree_ptr();

            this->read_lock_av_key();
            VAL_COUNTS_TYPE val_counts;
            for (auto &[attr, val_map]: attr_vals) {
                val_counts[attr] = val_map.size();
            }
            this->read_unlock_av_key();

            MultinomialCobwebNode* current = root;
            current->write_lock();

            // look ahead to detect fringe split, this is the only case we need to retain parent lock.
            if (!current->children.empty() || current->count == 0 || current->is_exact_match(instance)) {
                this->write_unlock_tree_ptr(); 
            }

            while (true) {
                // each loop starts with a write lock on current and
                // current->parent (in the case of root, root_ptr_mtx is write
                // locked instead of current->parent).
                if (current->children.empty() && (current->count == 0 || current->is_exact_match(instance))) {
                    // std::cout << "empty / exact match" << std::endl;
                    current->increment_counts(instance);
                    current->write_unlock();
                    break;
                } else if (current->children.empty()) {
                    //for this case both current and its parent/root is locked.
                    // std::cout << "fringe split" << std::endl;
                    MultinomialCobwebNode* new_node = new MultinomialCobwebNode(current);
                    new_node->write_lock();
                    current->parent = new_node;
                    current->write_unlock();
                    new_node->children.push_back(current);

                    if (new_node->parent == nullptr) {
                        root = new_node;
                        this->write_unlock_tree_ptr();
                    }
                    else{
                        new_node->parent->children.erase(remove(new_node->parent->children.begin(),
                            new_node->parent->children.end(), current), new_node->parent->children.end());
                        new_node->parent->children.push_back(new_node);
                        new_node->parent->write_unlock();
                    }
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
                    auto[best1_mi, best1, best2] = current->two_best_children(instance, val_counts);

                    // lock best1's children for evaluating split
                    for (auto &c: best1->children){
                        c->read_lock();
                    }
                    auto[_, bestAction] = current->get_best_operation(instance, best1, best2, best1_mi, val_counts);

                    for (auto &c: best1->children){
                        c->read_unlock();
                    }
                    for (auto &c: current->children) {
                        c->read_unlock();
                    }

                    if (bestAction == "best") {
                        // std::cout << "best" << std::endl;
                        current->increment_counts(instance);
                        // TODO should explore an "upgrade lock" on best1
                        best1->write_lock();

                        // look ahead to detect fringe split, which is the only case we need to retain parent lock.
                        if (!best1->children.empty() || best1->count == 0 || best1->is_exact_match(instance)) {
                            current->write_unlock(); 
                        }

                        current = best1;
                    } else if (bestAction == "new") {
                        // std::cout << "new" << std::endl;
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
                        // std::cout << "merge" << std::endl;
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
                        current->write_unlock();
                        current = new_child;
                    } else if (bestAction == "split") {
                        // std::cout << "split" << std::endl;
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

            this->read_lock_tree_ptr();

            AV_KEY_TYPE av_keys = attr_vals;

            auto current = get_root();
            current->read_lock();

            // TODO the best node might get deleted, not threadsafe!!!
            auto best_concept = current;
            double best_score = best_concept->log_prob_class_given_instance(instance, av_keys);

            while (true) {
                if (current->children.empty()) {
                    if (current->parent == nullptr){
                        this->read_unlock_tree_ptr();
                    }
                    else{
                        current->parent->read_unlock();
                    }
                    current->read_unlock();
                    if (get_best_concept) return best_concept;
                    return current;
                }

                if (current->parent == nullptr){
                    this->read_unlock_tree_ptr();
                }
                else{
                    current->parent->read_unlock();
                }

                auto parent = current;
                current = nullptr;
                double best_logp = 0.0;

                for (auto &child: parent->children) {
                    child->read_lock();
                    double logp = child->log_prob_class_given_instance(instance, av_keys);
                    if (current == nullptr || logp > best_logp){
                        best_logp = logp;
                        if (current != nullptr){
                            current->read_unlock();
                        }
                        current = child;

                        double score = 0.0;
                        score = logp;

                        if (score > best_score){
                            best_score = score;
                            best_concept = current;
                        }
                    }
                    if (current != child){
                        child->read_unlock();
                    }
                }
            }
        }

        MultinomialCobwebNode* categorize_helper(const INSTANCE_TYPE &instance, bool get_best_concept){
            AV_COUNT_TYPE cached_instance;
            for (auto &[attr, val_map]: instance) {
                for (auto &[val, cnt]: val_map) {
                    cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
                }
            }
            return this->_cobweb_categorize(cached_instance, get_best_concept);
        }

        CategorizationFuture* categorize(const INSTANCE_TYPE instance, bool get_best_concept) {
            return new CategorizationFuture(this->categorize_helper(instance, get_best_concept));
        }

        CategorizationFuture* async_categorize(const INSTANCE_TYPE &instance, bool get_best_concept) {
            auto fut_result = pool.submit([this, instance, get_best_concept]() {
                auto* result = this->categorize_helper(instance, get_best_concept);
                return result;
            });
            return new CategorizationFuture(std::move(fut_result));
        }

};

inline CategorizationFuture::CategorizationFuture(std::future<MultinomialCobwebNode*> leaf_future): 
    leaf_future(std::move(leaf_future)), leaf(nullptr) {}

inline CategorizationFuture::CategorizationFuture(MultinomialCobwebNode* leaf_node):
    leaf_future(), leaf(leaf_node) {}

inline void CategorizationFuture::wait(){
    if (leaf == nullptr){
        leaf = leaf_future.get();
    }
}

inline std::unordered_map<ATTR_TYPE, std::unordered_map<VALUE_TYPE, double>> CategorizationFuture::predict(){
    std::unordered_map<ATTR_TYPE, std::unordered_map<VALUE_TYPE, double>> out;

    if (leaf == nullptr){
        leaf = leaf_future.get();
    }

    leaf->read_lock();

    AV_KEY_TYPE attr_vals = leaf->tree->attr_vals;

    for (auto &[attr, val_set]: attr_vals) {
        // std::cout << attr << std::endl;
        int num_vals = attr_vals.at(attr).size();
        float alpha = leaf->tree->alpha(num_vals);
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
    auto start_time = std::chrono::high_resolution_clock::now();
    node_mtx.lock_shared();
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
    read_wait_time += duration;
    // std::cout << "read locked: " << this << std::endl;
}

inline void MultinomialCobwebNode::read_unlock(){
    node_mtx.unlock_shared();
    // std::cout << "read unlocked: " << this << std::endl;
}

inline void MultinomialCobwebNode::write_lock(){
    auto start_time = std::chrono::high_resolution_clock::now();
    node_mtx.lock();
    // std::cout << "write locked: " << this << std::endl;
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
    write_wait_time += duration;
}

inline void MultinomialCobwebNode::write_unlock(){
    node_mtx.unlock();
    // std::cout << "write unlocked: " << this << std::endl;
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

inline double MultinomialCobwebNode::score_insert(const AV_COUNT_TYPE &instance,
        const VAL_COUNTS_TYPE &val_counts){
    std::unordered_set<ATTR_TYPE> all_attrs;
    for (auto &[attr, tmp]: instance){
        //if (attr[0] == '_') continue;
        if (attr.is_hidden()) continue;
        all_attrs.insert(attr);
    }
    for (auto &[attr, tmp]: this->av_counts){
        //if (attr[0] == '_') continue;
        if (attr.is_hidden()) continue;
        all_attrs.insert(attr);
    }

    return transform_reduce(PAR all_attrs.begin(), all_attrs.end(), 0.0,
            std::plus<>(), [&](const auto& attr_it){
                COUNT_TYPE attr_count = 0;
                std::unordered_set<VALUE_TYPE> all_vals;
                int num_vals = val_counts.at(attr_it);
                float alpha = this->tree->alpha(num_vals);
                // int num_vals = this->tree->num_vals_for_attr(attr_it);

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

inline double MultinomialCobwebNode::score_merge(MultinomialCobwebNode *other, const AV_COUNT_TYPE &instance,
        const VAL_COUNTS_TYPE &val_counts) {

    std::unordered_set<ATTR_TYPE> all_attrs;
    for (auto &[attr, tmp]: instance){
        // if (attr[0] == '_') continue;
        if (attr.is_hidden()) continue;
        all_attrs.insert(attr);
    }
    for (auto &[attr, tmp]: this->av_counts){
        // if (attr[0] == '_') continue;
        if (attr.is_hidden()) continue;
        all_attrs.insert(attr);
    }
    for (auto &[attr, tmp]: other->av_counts){
        // if (attr[0] == '_') continue;
        if (attr.is_hidden()) continue;
        all_attrs.insert(attr);
    }

    return transform_reduce(PAR all_attrs.begin(), all_attrs.end(), 0.0,
            std::plus<>(), [&](const auto& attr_it){
                COUNT_TYPE attr_count = 0;
                std::unordered_set<ATTR_TYPE> all_vals;
                int num_vals = val_counts.at(attr_it);
                float alpha = this->tree->alpha(num_vals);

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

inline MultinomialCobwebNode* MultinomialCobwebNode::get_best_level(
        const AV_COUNT_TYPE &instance, const AV_KEY_TYPE &av_keys){
    MultinomialCobwebNode* curr = this;
    MultinomialCobwebNode* best = this;
    double best_ll = this->log_prob_class_given_instance(instance, av_keys);

    while (curr->parent != nullptr) {
        curr = curr->parent;
        double curr_ll = curr->log_prob_class_given_instance(instance, av_keys);

        if (curr_ll > best_ll) {
            best = curr;
            best_ll = curr_ll;
        }
    }

    return best;
}

inline MultinomialCobwebNode* MultinomialCobwebNode::get_basic_level(const VAL_COUNTS_TYPE &val_counts){
    MultinomialCobwebNode* curr = this;
    MultinomialCobwebNode* best = this;
    double best_cu = this->category_utility(val_counts);

    while (curr->parent != nullptr) {
        curr = curr->parent;
        double curr_cu = curr->category_utility(val_counts);

        if (curr_cu > best_cu) {
            best = curr;
            best_cu = curr_cu;
        }
    }

    return best;
}

inline double MultinomialCobwebNode::score(const VAL_COUNTS_TYPE &val_counts) {

    return transform_reduce(PAR this->av_counts.begin(), this->av_counts.end(), 0.0,
            std::plus<>(), [&](const auto& attr_it){
                // if (attr_it.first[0] == '_') return 0.0;
                if (attr_it.first.is_hidden()) return 0.0;

                COUNT_TYPE attr_count = this->attr_counts.at(attr_it.first);
                int num_vals = val_counts.at(attr_it.first);
                float alpha = this->tree->alpha(num_vals);

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

inline double MultinomialCobwebNode::partition_utility(const VAL_COUNTS_TYPE &val_counts) {
    if (children.empty()) {
        return 0.0;
    }

    double children_score = 0.0;

    for (auto &child: children) {
        double p_of_child = (1.0 * child->count) / this->count;
        children_score += p_of_child * child->score(val_counts);
    }

    return ((this->score(val_counts) - children_score) / children.size());

}

inline std::tuple<double, std::string> MultinomialCobwebNode::get_best_operation(
        const AV_COUNT_TYPE &instance, MultinomialCobwebNode *best1,
        MultinomialCobwebNode *best2, double best1_pu,
        const VAL_COUNTS_TYPE &val_counts) {

    if (best1 == nullptr) {
        throw "Need at least one best child.";
    }
    std::vector<std::tuple<double, double, std::string>> operations;
    operations.push_back(std::make_tuple(best1_pu,
                custom_rand(),
                "best"));
    operations.push_back(std::make_tuple(pu_for_new_child(instance, val_counts),
                custom_rand(),
                "new"));
    if (children.size() > 2 && best2 != nullptr) {
        operations.push_back(std::make_tuple(pu_for_merge(best1, best2,
                        instance, val_counts),
                    custom_rand(),
                    "merge"));
    }

    if (best1->children.size() > 0) {
        operations.push_back(std::make_tuple(pu_for_split(best1, val_counts),
                    custom_rand(),
                    "split"));
    }

    sort(operations.rbegin(), operations.rend());

    OPERATION_TYPE bestOp = make_pair(std::get<0>(operations[0]), std::get<2>(operations[0]));
    return bestOp;
}

inline std::tuple<double, MultinomialCobwebNode *, MultinomialCobwebNode *> MultinomialCobwebNode::two_best_children(
        const AV_COUNT_TYPE &instance, const VAL_COUNTS_TYPE &val_counts) {

    if (children.empty()) {
        throw "No children!";
    }

    std::vector<std::tuple<double, double, double, MultinomialCobwebNode *>> relative_pu;
    for (auto &child: this->children) {
        relative_pu.push_back(
                std::make_tuple(
                    (child->count * child->score(val_counts)) -
                    ((child->count + 1) * child->score_insert(instance, val_counts)),
                    child->count,
                    custom_rand(),
                    child));
    }

    sort(relative_pu.rbegin(), relative_pu.rend());

    MultinomialCobwebNode *best1 = std::get<3>(relative_pu[0]);
    double best1_pu = pu_for_insert(best1, instance, val_counts);
    MultinomialCobwebNode *best2 = relative_pu.size() > 1 ? std::get<3>(relative_pu[1]) : nullptr;
    return std::make_tuple(best1_pu, best1, best2);
}

inline double MultinomialCobwebNode::pu_for_insert(MultinomialCobwebNode *child, const AV_COUNT_TYPE &instance,
        const VAL_COUNTS_TYPE &val_counts) {
    double children_score = 0.0;

    for (auto &c: this->children) {
        if (c == child) {
            double p_of_child = (c->count + 1.0) / (this->count + 1.0);
            children_score += p_of_child * c->score_insert(instance, val_counts);
        }
        else{
            double p_of_child = (1.0 * c->count) / (this->count + 1.0);
            children_score += p_of_child * c->score(val_counts);
        }
    }

    return ((this->score_insert(instance, val_counts) - children_score) / this->children.size());
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

inline double MultinomialCobwebNode::pu_for_new_child(const AV_COUNT_TYPE &instance,
        const VAL_COUNTS_TYPE &val_counts) {
    double children_score = 0.0;

    for (auto &c: this->children) {
        double p_of_child = (1.0 * c->count) / (this->count + 1.0);
        children_score += p_of_child * c->score(val_counts);
    }

    MultinomialCobwebNode new_child = MultinomialCobwebNode();
    new_child.parent = this;
    new_child.tree = this->tree;
    new_child.increment_counts(instance);
    double p_of_child = 1.0 / (this->count + 1.0);
    children_score += p_of_child * new_child.score(val_counts);

    return ((this->score_insert(instance, val_counts) - children_score) /
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
        MultinomialCobwebNode *best2, const AV_COUNT_TYPE &instance,
        const VAL_COUNTS_TYPE &val_counts) {

    double children_score = 0.0;

    for (auto &c: children) {
        if (c == best1 || c == best2){
            continue;
        }

        double p_of_child = (1.0 * c->count) / (this->count + 1.0);
        children_score += p_of_child * c->score(val_counts);
    }

    double p_of_child = (best1->count + best2->count + 1.0) / (this->count + 1.0);
    children_score += p_of_child * best1->score_merge(best2, instance, val_counts);

    return ((this->score_insert(instance, val_counts) - children_score) / (children.size()-1));
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

inline double MultinomialCobwebNode::pu_for_split(MultinomialCobwebNode *best,
        const VAL_COUNTS_TYPE &val_counts){
    double children_score = 0.0;

    for (auto &c: children) {
        if (c == best) continue;

        double p_of_child = (1.0 * c->count) / this->count;
        children_score += p_of_child * c->score(val_counts);
    }

    for (auto &c: best->children) {
        double p_of_child = (1.0 * c->count) / this->count;
        children_score += p_of_child * c->score(val_counts);
    }

    double pu = ((this->score(val_counts) - children_score) / (children.size() - 1 + best->children.size()));

    return pu;
}

inline bool MultinomialCobwebNode::is_exact_match(const AV_COUNT_TYPE &instance) {
    std::unordered_set<ATTR_TYPE> all_attrs;
    for (auto &[attr, tmp]: instance) all_attrs.insert(attr);
    for (auto &[attr, tmp]: this->av_counts) all_attrs.insert(attr);

    for (auto &attr: all_attrs) {
        // if (attr[0] == '_') continue;
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
        // if (attr[0] == '_') continue;
        if (attr.is_hidden()) continue;
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

inline double MultinomialCobwebNode::category_utility(const VAL_COUNTS_TYPE &val_counts){
    double p_of_c = (1.0 * this->count) / this->tree->get_root()->count;
    return (p_of_c * (this->tree->get_root()->score(val_counts) - this->score(val_counts)));
}

inline double MultinomialCobwebNode::log_prob_class_given_instance(const AV_COUNT_TYPE &instance, const AV_KEY_TYPE &av_keys){

    double log_prob = 0;

    for (auto &[attr, vAttr]: instance) {
        // bool hidden = attr[0] == '_';
        bool hidden = attr.is_hidden();
        // if (hidden || !this->tree->root->av_counts.count(attr)){
        if (hidden || !av_keys.count(attr)){
            continue;
        }

        double num_vals = av_keys.at(attr).size();
        // double num_vals = this->tree->root->av_counts.at(attr).size();

        for (auto &[val, cnt]: vAttr){
            // if (!this->tree->root->av_counts.at(attr).count(val)){
            if (!av_keys.at(attr).count(val)){
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

    log_prob += log((1.0 * this->count) / this->tree->get_root()->count);

    return log_prob;
}



int main(int argc, char* argv[]) {
    std::vector<AV_COUNT_TYPE> instances;
    std::vector<CategorizationFuture*> cfs;
    auto tree = MultinomialCobwebTree(true, 1.0, true, true);

    for (int i = 0; i < 1000; i++){
        INSTANCE_TYPE inst;
        inst["anchor"]["word" + std::to_string(i)] = 1;
        inst["anchor2"]["word" + std::to_string(i % 10)] = 1;
        inst["anchor3"]["word" + std::to_string(i % 20)] = 1;
        inst["anchor4"]["word" + std::to_string(i % 100)] = 1;
        cfs.push_back(tree.async_ifit(inst));
    }
    for (int i = 0; i < 1000; i++){
        cfs.at(i)->wait();
    }
     
    return 0;
}


PYBIND11_MODULE(multinomial_cobweb, m) {
    m.doc() = "concept_formation.multinomial_cobweb plugin"; // optional module docstring

    py::class_<CategorizationFuture>(m, "CategorizationFuture")
        .def("wait", &CategorizationFuture::wait, py::call_guard<py::gil_scoped_release>())
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
        .def_readonly("read_wait_time", &MultinomialCobwebNode::read_wait_time)
        .def_readonly("write_wait_time", &MultinomialCobwebNode::write_wait_time)
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
        .def("async_ifit", &MultinomialCobwebTree::async_ifit, py::call_guard<py::gil_scoped_release>())
        .def("ifit", &MultinomialCobwebTree::ifit)
        .def("fit", &MultinomialCobwebTree::fit,
                py::arg("instances") = std::vector<AV_COUNT_TYPE>(),
                py::arg("iterations") = 1,
                py::arg("randomizeFirst") = true)
        .def("categorize", &MultinomialCobwebTree::categorize,
                py::arg("instance") = std::vector<AV_COUNT_TYPE>(),
                py::arg("get_best_concept") = false, py::return_value_policy::reference)
        .def("async_categorize", &MultinomialCobwebTree::categorize,
                py::arg("instance") = std::vector<AV_COUNT_TYPE>(),
                py::arg("get_best_concept") = false, py::return_value_policy::reference)
        .def("clear", &MultinomialCobwebTree::clear)
        .def("__str__", &MultinomialCobwebTree::__str__)
        .def("dump_json", &MultinomialCobwebTree::dump_json)
        .def("load_json", &MultinomialCobwebTree::load_json)
        .def_readonly("av_key_wait_time", &MultinomialCobwebTree::av_key_wait_time)
        .def_readonly("write_wait_time", &MultinomialCobwebTree::write_wait_time)
        .def_readonly("root", &MultinomialCobwebTree::root, py::return_value_policy::reference);
}
