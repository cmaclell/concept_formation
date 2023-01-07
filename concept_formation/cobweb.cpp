#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <random>
#include <tuple>
#include <set>
#include "assert.h"

#define NULL_STRING "\0"

using namespace std;

typedef string ATTR_TYPE;
typedef string VALUE_TYPE;
typedef unsigned long COUNT_TYPE;
typedef unordered_map<ATTR_TYPE, unordered_map<VALUE_TYPE, COUNT_TYPE> > AV_COUNT_TYPE;
typedef unordered_map<ATTR_TYPE, VALUE_TYPE> INSTANCE_TYPE;
typedef pair<double, string> OPERATION_TYPE;

class CobwebTree;
class CobwebNode;

//utils functions
/**
 *
 * @return a number in range [0, 1)
 */
double custom_rand() {
    random_device rd;
    mt19937_64 gen(rd());
    uniform_real_distribution<double> unif(0, 1);
    return unif(gen);
}

/**
 *
 * @param s
 * @param n
 * @return string s repeated n times
 */
string repeat(string s, int n) {
    string res = "";
    for (int i = 0; i < n; i++) {
        res += s;
    }
    return res;
}


VALUE_TYPE most_likely_choice(vector<tuple<VALUE_TYPE, double>> choices) {
    cout << "most_likely_choice: Not implemented yet" << endl;
    return get<0>(choices[0]);
}

VALUE_TYPE weighted_choice(vector<tuple<VALUE_TYPE, double>> choices) {
    cout << "weighted_choice: Not implemented yet" << endl;
    return get<0>(choices[0]);
}


class CobwebNode {

    static unsigned long counter;
    int concept_id;
    unsigned long count;
    unsigned long long squared_counts;
    unsigned long attr_count;
    vector<CobwebNode *> children;
    CobwebNode *parent;
    CobwebTree *tree;

private:
    /**
     *
     * @return av_counts in json format
     */
    string av_countsToString() {
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

public:
    AV_COUNT_TYPE av_counts;

    CobwebNode() {
        concept_id = genSym();
        count = 0;
        squared_counts = 0;
        attr_count = 0;
        parent = NULL;
        tree = NULL;

    }

    CobwebNode(CobwebNode *otherNode) {
        concept_id = genSym();
        count = 0;
        squared_counts = 0;
        attr_count = 0;

        parent = otherNode->parent;
        tree = otherNode->tree;

        update_counts_from_node(otherNode);

        for (auto child: otherNode->children) {
            children.push_back(new CobwebNode(child));
        }

    }

    void increment_counts(INSTANCE_TYPE instance) {
        count += 1;

        for (auto &[attr, val]: instance) {
            bool hidden = attr[0] == '_';

            if (!hidden & !av_counts.count(attr)){
                attr_count += 1;
            }

            if (!hidden){
                squared_counts -= pow(av_counts[attr][val], 2);
            }

            av_counts[attr][instance[attr]]++;

            if (!hidden){
                squared_counts += pow(av_counts[attr][val], 2);
            }
        }
    }

    void update_counts_from_node(CobwebNode *node) {
        count += node->count;

        for (auto &[attr, tmp]: node->av_counts) {
            bool hidden = attr[0] == '_';

            if (!hidden & !av_counts.count(attr)){
                attr_count += 1;
            }

            for (auto&[val, tmp2]: node->av_counts.at(attr)) {

                if (!hidden){
                    squared_counts -= pow(av_counts[attr][val], 2);
                }

                av_counts[attr][val] += node->av_counts.at(attr).at(val);

                if (!hidden){
                    squared_counts += pow(av_counts[attr][val], 2);
                }
            }
        }
    }

    double expected_correct_guesses_insert(INSTANCE_TYPE instance){
        unsigned long attr_count = this->attr_count;
        unsigned long long squared_counts = this->squared_counts;

        for (auto &[attr, val]: instance) {
            if (attr[0] == '_'){
                continue;    
            }

            if (!av_counts.count(attr)){
                attr_count += 1;
            }

            int av_count = 0;
            try{
                av_count = av_counts.at(attr).at(val);
            }
            catch (const out_of_range& e) { }

            squared_counts -= pow(av_count, 2);
            squared_counts += pow(av_count + 1, 2);
        }

        return squared_counts / pow(count + 1, 2) / attr_count;
    }

    double expected_correct_guesses_merge(CobwebNode *other, INSTANCE_TYPE instance) {

        CobwebNode* big = this;
        CobwebNode* small = other;

        if (count < other->count){
            small = this;
            big = other;
        }

        unsigned long attr_count = big->attr_count;
        unsigned long long squared_counts = big->squared_counts;

        for (auto &[attr, tmp]: small->av_counts) {
            if (attr[0] == '_'){
                continue;
            }

            if (!big->av_counts.count(attr)){
                attr_count += 1;
            }

            for (auto&[val, tmp2]: small->av_counts.at(attr)) {
                int big_count = 0;
                try{
                    big_count = big->av_counts.at(attr).at(val);
                }
                catch (const out_of_range& e) { }

                squared_counts -= pow(big_count, 2); 
                squared_counts += pow(big_count + small->av_counts.at(attr).at(val), 2); 
            }
        }

        for (auto &[attr, val]: instance) {
            if (attr[0] == '_'){
                continue;    
            }

            if (!big->av_counts.count(attr) && !small->av_counts.count(attr)){
                attr_count += 1;
            }

            int big_count = 0;
            try{
                big_count = big->av_counts.at(attr).at(val);
            }
            catch (const out_of_range& e) { }

            int small_count = 0;
            try{
                small_count = small->av_counts.at(attr).at(val);
            }
            catch (const out_of_range& e) { }

            squared_counts -= pow(big_count + small_count, 2);
            squared_counts += pow(big_count + small_count + 1, 2);
        }

        return squared_counts / pow(big->count + small->count + 1, 2) / attr_count;

    }

    double expected_correct_guesses() {
        return squared_counts / pow(count, 2) / attr_count;
    }

    double category_utility() {
        if (children.empty()) {
            return 0.0;
        }
        double childCorrectGuesses = 0.0;
        for (auto &child: children) {
            double pOfChild = child->count / this->count;
            childCorrectGuesses += pOfChild * child->expected_correct_guesses();
        }
        return ((childCorrectGuesses - this->expected_correct_guesses()) / children.size());
    }

    tuple<double, string> get_best_operation(INSTANCE_TYPE instance, CobwebNode *best1,
                                             CobwebNode *best2, double best1Cu,
                                             bool best_op=true, bool new_op=true,
                                             bool merge_op=true, bool split_op=true) {

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

        /*
        cout << endl;
        for (auto &[cu, rand, op]: operations){
            cout << cu << ", " << rand << ", " << op << endl;
        }
        */

        OPERATION_TYPE bestOp = make_pair(get<0>(operations[0]), get<2>(operations[0]));
        return bestOp;
    };

    tuple<double, CobwebNode *, CobwebNode *> two_best_children(INSTANCE_TYPE instance) {
        if (children.empty()) {
            throw "No children!";
        }
        vector<tuple<double, double, double, CobwebNode *>> relative_cus;
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

        CobwebNode *best1 = get<3>(relative_cus[0]);
        double best1_cu = cu_for_insert(best1, instance);
        CobwebNode *best2 = relative_cus.size() > 1 ? get<3>(relative_cus[1]) : NULL;
        return make_tuple(best1_cu, best1, best2);
    }

    double cu_for_insert(CobwebNode *child, INSTANCE_TYPE instance) {
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

    CobwebNode *create_new_child(INSTANCE_TYPE instance) {
        CobwebNode *newChild = new CobwebNode();
        newChild->parent = this;
        newChild->tree = this->tree;
        newChild->increment_counts(instance);
        this->children.push_back(newChild);
        return newChild;
    };

    double cu_for_new_child(INSTANCE_TYPE instance) {
        double child_correct_guesses = 0.0;

        for (auto &c: children) {
            child_correct_guesses += (c->count * c->expected_correct_guesses()); 
        }

        // sum over all attr (at 100% prob) divided by num attr should be 1.
        child_correct_guesses += 1;

        child_correct_guesses /= (count + 1);
        double parent_correct_guesses = expected_correct_guesses_insert(instance);

        return ((child_correct_guesses - parent_correct_guesses) / (children.size()+1));
    }

    CobwebNode *merge(CobwebNode *best1, CobwebNode *best2) {
        CobwebNode *newChild = new CobwebNode();
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

    double cu_for_merge(CobwebNode *best1, CobwebNode *best2, INSTANCE_TYPE instance) {
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

    void split(CobwebNode *best) {
        children.erase(remove(children.begin(), children.end(), best), children.end());
        for (auto &c: best->children) {
            c->parent = this;
            c->tree = this->tree;
            children.push_back(c);
        }
        delete best;
    }

    double cu_for_split(CobwebNode *best){
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

    bool is_exact_match(INSTANCE_TYPE instance) {
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
                if (!this->av_counts.at(attr).count(instance[attr])) {
                    return false;
                }
                if (this->av_counts.at(attr).at(instance[attr]) != this->count) {
                    return false;
                }
            }
        }
        return true;
    }

    long _hash() {
        hash<string> hash_obj;
        return hash_obj("CobwebNode" + to_string(concept_id));
    }

    int genSym() {
        counter++;
        return counter;
    }

    friend ostream &operator<<(ostream &os, CobwebNode *node) {
        os << node->pretty_print();
        return os;
    }

    string pretty_print(int depth = 0) {
        string ret = repeat("\t", depth) + "|-" + av_countsToString() + ":" +
                     (to_string(this->count)) + " (" + to_string(concept_id) + ", " + to_string(attr_count) + ", " + to_string(this->squared_counts) + ", " + to_string(this->expected_correct_guesses()) + ")\n";

        for (auto &c: children) {
            ret += c->pretty_print(depth + 1);
        }

        return ret;
    }


    int depth() {
        if (this->parent) {
            return 1 + this->parent->depth();
        }
        return 0;
    }

    bool is_parent(CobwebNode *otherConcept) {
        CobwebNode *temp = otherConcept;
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

    int num_concepts() {
        int childrenCount = 0;
        for (auto &c: children) {
            childrenCount += c->num_concepts();
        }
        return 1 + childrenCount;
    }

    // TODO output_json()

    vector<tuple<VALUE_TYPE, double>> get_weighted_values(ATTR_TYPE attr, bool allowNone = true) {
        vector<tuple<VALUE_TYPE, double>> choices;
        if (!this->av_counts.count(attr)) {
            choices.push_back(make_tuple(NULL_STRING, 1.0));
        }
        double valCount = 0;
        for (auto &[val, tmp]: this->av_counts.at(attr)) {
            COUNT_TYPE count = this->av_counts.at(attr).at(val);
            choices.push_back(make_tuple(val, 1.0 * count / this->count));
            valCount += count;
        }
        if (allowNone) {
            choices.push_back(make_tuple(NULL_STRING, ((this->count - valCount) / this->count)));
        }
        return choices;
    }

    VALUE_TYPE predict(ATTR_TYPE attr, string choiceFn = "most likely", bool allowNone = true) {
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

    double probability(ATTR_TYPE attr, VALUE_TYPE val) {
        if (val == NULL_STRING) {
            double c = 0.0;
            if (this->av_counts.count(attr)) {
                for (auto &[attr, vAttr]: this->av_counts) {
                    for (auto&[val, cnt]: vAttr) {
                        c += cnt;
                    }
                }
                return 1.0 * (this->count - c) / this->count;
            }
        }
        if (this->av_counts.count(attr) && this->av_counts.at(attr).count(val)) {
            return 1.0 * this->av_counts.at(attr).at(val) / this->count;
        }
        return 0.0;
    }

    double log_likelihood(CobwebNode *childLeaf) {
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

    static int getCounter() {
        return counter;
    }

    static void setCounter(int counter) {
        CobwebNode::counter = counter;
    }

    int getConceptId() const {
        return concept_id;
    }

    void setConceptId(int concept_id) {
        CobwebNode::concept_id = concept_id;
    }

    unsigned long getCount() const {
        return count;
    }

    void setCount(unsigned long count) {
        CobwebNode::count = count;
    }

    const AV_COUNT_TYPE &getAvCounts() const {
        return av_counts;
    }

    void setAvCounts(const AV_COUNT_TYPE &av_counts) {
        CobwebNode::av_counts = av_counts;
    }

    vector<CobwebNode *> &getChildren() {
        return children;
    }

    void setChildren(const vector<CobwebNode *> &children) {
        CobwebNode::children = children;
    }

    CobwebNode *getParent() const {
        return parent;
    }

    void setParent(CobwebNode *parent) {
        CobwebNode::parent = parent;
    }

    CobwebTree *getTree() const {
        return tree;
    }

    void setTree(CobwebTree *tree) {
        CobwebNode::tree = tree;
    }

};

class CobwebTree {
    CobwebNode *root;

public:
    CobwebTree() {
        this->root = new CobwebNode();
        this->root->setTree(this);
    }

    string __str__(){
        return this->root->pretty_print(0);
    }

    void clear() {
        this->root = new CobwebNode();
        this->root->setTree(this);
    }

    friend ostream &operator<<(ostream &os, CobwebTree *tree) {
        os << tree->root;
        return os;
    }

    void _sanity_check_instance(INSTANCE_TYPE instance) {
        for (auto&[attr, v]: instance) {
            try {
//                hash(attr); todo:
                attr[0];
            } catch (string e) {
                throw "Invalid attribute: ";
            }
            try {
//                hash(instance[attr]); todo:
            } catch (string e) {
                throw "Invalid value";
            }

        }
    }

    CobwebNode *ifit(INSTANCE_TYPE instance) {
        this->_sanity_check_instance(instance);
        return this->cobweb(instance);
    }

    void fit(vector<INSTANCE_TYPE> instances, int iterations = 1, bool randomizeFirst = true) {
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

    CobwebNode *cobweb(INSTANCE_TYPE instance) {
        CobwebNode *current = this->root;
        while (current != NULL) {
            if (current->getChildren().empty() && (current->is_exact_match(instance) || current->getCount() == 0)) {
                current->increment_counts(instance);
                break;
            } else if (current->getChildren().empty()) {
                CobwebNode *newNode = new CobwebNode(current);
                current->setParent(newNode);
                newNode->getChildren().push_back(current);

                if (newNode->getParent() != NULL) {
                    newNode->getParent()->getChildren().erase(remove(newNode->getParent()->getChildren().begin(),
                                                                     newNode->getParent()->getChildren().end(),
                                                                     current),
                                                              newNode->getParent()->getChildren().end());
                    newNode->getParent()->getChildren().push_back(newNode);
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
                    CobwebNode *newChild = current->merge(best1, best2);
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

    CobwebNode *_cobweb_categorize(INSTANCE_TYPE instance) {
        auto current = this->root;
        while (current != NULL) {
            if (current->getChildren().empty()) {
                return current;
            }
            auto[_, best1, best2] = current->two_best_children(instance);
            current = best1;
        }

        return current;
    }

    INSTANCE_TYPE infer_missing(INSTANCE_TYPE instance, string choiceFn = "most likely", bool allowNone = true) {
        this->_sanity_check_instance(instance);
        INSTANCE_TYPE tempInstance;
        for (auto&[attr, v]: instance) {
            tempInstance[attr] = v;
        }
        auto concept = this->_cobweb_categorize(tempInstance);
        for (auto &[attr, tmp]: concept->av_counts) {
            if (tempInstance.count(attr)) {
                continue;
            }
            auto val = concept->predict(attr, choiceFn, allowNone);
            if (val != NULL_STRING) {
                tempInstance[attr] = val;
            }
        }
        return tempInstance;
    }

    CobwebNode *categorize(INSTANCE_TYPE instance) {
        this->_sanity_check_instance(instance);
        return this->_cobweb_categorize(instance);
    }
};


unsigned long CobwebNode::counter = 0;

/*
int main() {
    CobwebNode *node = new CobwebNode();
    cout << node->getConceptId();
    CobwebTree *tree = new CobwebTree();
    cout << tree << endl;
    return 0;
}
*/

PYBIND11_MODULE(cobweb, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::class_<CobwebNode>(m, "CobwebNode")
        .def(py::init())
        .def("pretty_print", &CobwebNode::pretty_print);

    py::class_<CobwebTree>(m, "CobwebTree")
        .def(py::init())
        .def("ifit", &CobwebTree::ifit, py::return_value_policy::copy)
        .def("fit", &CobwebTree::fit)
        .def("__str__", &CobwebTree::__str__);
//         .def_readonly("mean", &ContinuousValue::mean)
//         .def_readonly("meanSq", &ContinuousValue::meanSq);
}
