#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <iostream>
#include <vector>
#include <map>
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
typedef int COUNT_TYPE;
typedef map<ATTR_TYPE, map<VALUE_TYPE, COUNT_TYPE> > AV_COUNT_TYPE;
typedef map<ATTR_TYPE, VALUE_TYPE> INSTANCE_TYPE;
typedef pair<double, string> OPERATION_TYPE;

class CobwebTree;

class CobwebNode;


vector<string> DEFAULT_OPERATIONS = {"best", "new", "merge",
                                     "split"};
//utils functions
/**
 *
 * @return a number in range [0, 1)
 */
double customRand() {
    mt19937_64 rng;
    uniform_real_distribution<double> unif(0, 1);
    return unif(rng);
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


VALUE_TYPE mostLikelyChoice(vector<tuple<VALUE_TYPE, double>> choices) {
    cout << "mostLikelyChoice: Not implemented yet" << endl;
    return get<0>(choices[0]);
}

VALUE_TYPE weightedChoice(vector<tuple<VALUE_TYPE, double>> choices) {
    cout << "weightedChoice: Not implemented yet" << endl;
    return get<0>(choices[0]);
}


class CobwebNode {

    static int counter;
    int conceptId;
    double count;
    double squared_counts;
    double attr_counts;
    AV_COUNT_TYPE av_counts;
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
        for (auto &[attr, vAttr]: av_counts) {
            ret += "\"" + attr + "\": {";
            for (auto &[val, cnt]: vAttr) {
                ret += "\"" + val + "\": " + to_string(cnt) + ", ";
            }
            ret += "}, ";
        }
        ret += "}";
        return ret;
    }

public:
    CobwebNode() {
        conceptId = genSym();
        count = 0.0;
        squared_counts = 0.0;
        attr_counts = 0.0;
        parent = NULL;
        tree = NULL;

    }

    CobwebNode(CobwebNode *otherNode) {
        conceptId = genSym();
        count = 0.0;
        tree = otherNode->tree;
        parent = otherNode->parent;
        update_counts_from_node(otherNode);
        for (auto child: otherNode->children) {
            children.push_back(new CobwebNode(child));
        }

    }

    CobwebNode *shallowCopy() {
        CobwebNode *temp = new CobwebNode();
        temp->tree = tree;
        temp->parent = parent;
        temp->update_counts_from_node(this);
        return temp;
    };


    vector<string> attrs() {
        vector<string> res;
        for (auto &[attr, v]: av_counts) {
            if (attr.length() == 0 || attr[0] != '_') {
                res.push_back(attr);
            }
        }
        return res;
    }

    vector<string> attrs(string attr_filter) {
        vector<string> res;
        if (attr_filter == "all") {
            for (auto &[attr, v]: av_counts) {
                res.push_back(attr);
            }
        }
        return res;
    }

    vector<string> attrs(function<bool(string)> const &attr_filter) {
        vector<string> res;
        for (auto &[attr, v]: av_counts) {
            if (attr_filter(attr)) res.push_back(attr);
        }
        return res;
    }

    void increment_counts(INSTANCE_TYPE instance) {
        count++;

        for (auto &[attr, val]: instance) {
            bool hidden = attr[0] == '_';

            if (!hidden & !av_counts.count(attr)){
                attr_counts += 1;
            }

            if (!av_counts.count(attr)) {
                av_counts[attr] = {};
            }

            if (!av_counts[attr].count(instance[attr])) {
                av_counts[attr][instance[attr]] = 0;
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
                attr_counts += 1;
            }

            if (!av_counts.count(attr)) {
                av_counts[attr] = {};
            }

            for (auto&[val, tmp2]: node->av_counts[attr]) {
                if (!av_counts[attr].count(val)) {
                    av_counts[attr][val] = 0;
                }

                if (!hidden){
                    squared_counts -= pow(av_counts[attr][val], 2);
                }

                av_counts[attr][val] += node->av_counts[attr][val];

                if (!hidden){
                    squared_counts += pow(av_counts[attr][val], 2);
                }
            }
        }
    }

    double expected_correct_guesses() {
        return squared_counts / pow(count, 2) / attr_counts;
    }

    double expected_correct_guesses_insert(INSTANCE_TYPE instance){
        double attr_counts = this->attr_counts;
        double squared_counts = this->squared_counts;

        for (auto &[attr, val]: instance) {
            if (attr[0] == '_'){
                continue;    
            }

            if (!av_counts.count(attr)){
                attr_counts += 1;
            }

            squared_counts -= pow(av_counts[attr][val], 2);
            squared_counts += pow(av_counts[attr][val] + 1, 2);
        }

        return squared_counts / pow(count+1, 2) / attr_counts;
    }

    double expected_correct_guesses_merge(CobwebNode *other, INSTANCE_TYPE instance) {

        CobwebNode* big = this;
        CobwebNode* small = other;

        if (count < other->count){
            small = this;
            big = other;
        }

        attr_counts = big->attr_counts;
        squared_counts = big->squared_counts;

        for (auto &[attr, tmp]: small->av_counts) {
            if (attr[0] == '_'){
                continue;
            }

            if (!big->av_counts.count(attr)){
                attr_counts += 1;
            }

            for (auto&[val, tmp2]: small->av_counts[attr]) {
                squared_counts -= pow(big->av_counts[attr][val], 2); 
                squared_counts += pow(big->av_counts[attr][val] +
                                      small->av_counts[attr][val], 2); 
            }
        }

        for (auto &[attr, val]: instance) {
            if (attr[0] == '_'){
                continue;    
            }

            if (!big->av_counts.count(attr) && !small->av_counts.count(attr)){
                attr_counts += 1;
            }

            squared_counts -= pow(big->av_counts[attr][val] +
                                  small->av_counts[attr][val], 2); 
            squared_counts += pow(big->av_counts[attr][val] +
                                  small->av_counts[attr][val] + 1, 2); 
        }

        return squared_counts / pow(count+1, 2) / attr_counts;

    }

    double expectedCorrectGuesses() {
        double correctGuesses = 0.0;
        int attrCount = 0;
        for (auto &attr: attrs()) {
            attrCount++;
            if (av_counts.count(attr)) {
                for (auto&[val, tmp]: av_counts[attr]) {
                    double prob = av_counts[attr][val] / count;
                    correctGuesses += prob * prob;
                }
            }
        }
        return 1.0 * correctGuesses / attrCount;
    }

    double categoryUtility() {
        if (children.empty()) {
            return 0.0;
        }
        double childCorrectGuesses = 0.0;
        for (auto &child: children) {
            double pOfChild = 1.0 * child->count / this->count;
            childCorrectGuesses += pOfChild * child->expected_correct_guesses();
        }
        return ((childCorrectGuesses - this->expected_correct_guesses()) / children.size());
    }

    tuple<double, string>
    getBestOperation(INSTANCE_TYPE instance, CobwebNode *best1, CobwebNode *best2, double best1Cu,
                     vector<string> possibleOps = DEFAULT_OPERATIONS) {
        if (best1 == NULL) {
            throw "Need at least one best child.";
        }
        vector<tuple<double, double, string>> operations;
        if (find(possibleOps.begin(), possibleOps.end(), "best") != possibleOps.end()) {
            operations.push_back(make_tuple(best1Cu, customRand(), "best"));
        }
        if (find(possibleOps.begin(), possibleOps.end(), "new") != possibleOps.end()) {
            operations.push_back(make_tuple(cu_for_new_child(instance), customRand(), "new"));
        }
        if (find(possibleOps.begin(), possibleOps.end(), "merge") != possibleOps.end() && children.size() > 2 &&
            best2 != NULL) {
            operations.push_back(make_tuple(cu_for_merge(best1, best2, instance), customRand(), "merge"));
        }
        if (find(possibleOps.begin(), possibleOps.end(), "split") != possibleOps.end() && best1->children.size() > 0) {
            operations.push_back(make_tuple(cu_for_split(best1), customRand(), "split"));
        }
        sort(operations.rbegin(), operations.rend());
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
                    customRand(),
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

    CobwebNode *createNewChild(INSTANCE_TYPE instance) {
        CobwebNode *newChild = new CobwebNode();
        newChild->parent = this;
        newChild->tree = this->tree;
        newChild->increment_counts(instance);
        this->children.push_back(newChild);
        return newChild;
    };

    CobwebNode *createChildWithCurrentCounts() {
        if (this->count > 0) {
            CobwebNode *newChild = new CobwebNode();
            newChild->parent = this;
            newChild->tree = this->tree;
            this->children.push_back(newChild);
            return newChild;
        }
        return NULL;
    }

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

    bool isExactMatch(INSTANCE_TYPE instance) {
        set<ATTR_TYPE> allAttrs;
        for (auto &[attr, v]: instance) allAttrs.insert(attr);
        for (auto &attr: this->attrs()) allAttrs.insert(attr);
        for (auto &attr: allAttrs) {
            if (attr[0] == '_') continue;
            if (instance.count(attr) && !this->av_counts.count(attr)) {
                return false;
            }
            if (this->av_counts.count(attr) && !instance.count(attr)) {
                return false;
            }
            if (this->av_counts.count(attr) && instance.count(attr)) {
                if (!this->av_counts[attr].count(instance[attr])) {
                    return false;
                }
                if (this->av_counts[attr][instance[attr]] != this->count) {
                    return false;
                }
            }
        }
        return true;
    }
    long _hash() {
        hash<string> hash_obj;
        return hash_obj("CobwebNode" + to_string(conceptId));
    }

    int genSym() {
        counter++;
        return counter;
    }

    string prettyPrint(int depth = 0) {
        string ret = repeat("\t", depth) + "|-" + av_countsToString() + ":" +
                     (to_string(this->count)) + '\n';

        for (auto &c: children) {
            ret = ret + c->prettyPrint(depth + 1);
        }

        return ret;
    }

    friend ostream &operator<<(ostream &os, CobwebNode *node) {
        os << node->prettyPrint();
        return os;
    }


    int depth() {
        if (this->parent) {
            return 1 + this->parent->depth();
        }
        return 0;
    }

    bool isParent(CobwebNode *otherConcept) {
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

    int numConcepts() {
        int childrenCount = 0;
        for (auto &c: children) {
            childrenCount += c->numConcepts();
        }
        return 1 + childrenCount;
    }

    map<string, string> outputJson() {
        cout << "outputJson: not supported!" << endl;
        map<string, string> tmp;
        return tmp;
    }

    vector<tuple<VALUE_TYPE, double>> getWeightedValues(ATTR_TYPE attr, bool allowNone = true) {
        vector<tuple<VALUE_TYPE, double>> choices;
        if (!this->av_counts.count(attr)) {
            choices.push_back(make_tuple(NULL_STRING, 1.0));
        }
        double valCount = 0;
        for (auto &[val, tmp]: this->av_counts) {
            COUNT_TYPE count = this->av_counts[attr][val];
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
            choose = mostLikelyChoice;
        } else if (choiceFn == "sampled" || choiceFn == "s") {
            choose = weightedChoice;
        } else throw "Unknown choice_fn";
        if (!this->av_counts.count(attr)) {
            return NULL_STRING;
        }
        vector<tuple<VALUE_TYPE, double>> choices = this->getWeightedValues(attr, allowNone);
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
        if (this->av_counts.count(attr) && this->av_counts[attr].count(val)) {
            return 1.0 * this->av_counts[attr][val] / this->count;
        }
        return 0.0;
    }

    double logLikelihood(CobwebNode *childLeaf) {
        set<ATTR_TYPE> allAttrs;
        for (auto &attr: this->attrs()) allAttrs.insert(attr);
        for (auto &attr: childLeaf->attrs()) allAttrs.insert(attr);
        double ll = 0;
        for (auto &attr: allAttrs) {
            set<VALUE_TYPE> vals;
            vals.insert(NULL_STRING);
            if (this->av_counts.count(attr)) {
                for (auto &[val, tmp]: this->av_counts[attr]) vals.insert(val);
            }
            if (childLeaf->av_counts.count(attr)) {
                for (auto &[val, tmp]: childLeaf->av_counts[attr]) vals.insert(val);
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
        return conceptId;
    }

    void setConceptId(int conceptId) {
        CobwebNode::conceptId = conceptId;
    }

    double getCount() const {
        return count;
    }

    void setCount(double count) {
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

    void clear() {
        this->root = new CobwebNode();
        this->root->setTree(this);
    }

    friend ostream &operator<<(ostream &os, CobwebTree *tree) {
        os << tree->root;
        return os;
    }

    void _sanityCheckInstance(INSTANCE_TYPE instance) {
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
        this->_sanityCheckInstance(instance);
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
            if (current->getChildren().empty() && (current->isExactMatch(instance)) || current->getCount() == 0) {
                current->increment_counts(instance);
                break;
            } else if (current->getChildren().empty()) {
                CobwebNode *newNode = new CobwebNode(current);
                current->setParent(newNode);
                newNode->getChildren().push_back(newNode);
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
                current = newNode->createNewChild(instance);
                break;
            } else {
                auto[best1_cu, best1, best2] = current->two_best_children(instance);
                auto[_, bestAction] = current->getBestOperation(instance, best1, best2, best1_cu);
                if (bestAction == "best") {
                    current->increment_counts(instance);
                    current = best1;
                } else if (bestAction == "new") {
                    current->increment_counts(instance);
                    current = current->createNewChild(instance);
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

    CobwebNode *_cobwebCategorize(INSTANCE_TYPE instance) {
        auto current = this->root;
        while (current != NULL) {
            if (current->getChildren().empty()) {
                return current;
            }
            auto[_, best1, best2] = current->two_best_children(instance);
            current = best1;
        }
    }

    INSTANCE_TYPE inferMissing(INSTANCE_TYPE instance, string choiceFn = "most likely", bool allowNone = true) {
        this->_sanityCheckInstance(instance);
        INSTANCE_TYPE tempInstance;
        for (auto&[attr, v]: instance) {
            tempInstance[attr] = v;
        }
        auto concept = this->_cobwebCategorize(tempInstance);
        for (auto attr: concept->attrs("all")) {
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
        this->_sanityCheckInstance(instance);
        return this->_cobwebCategorize(instance);
    }
};

int CobwebNode::counter = 0;

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
        .def(py::init());

    py::class_<CobwebTree>(m, "CobwebTree")
        .def(py::init())
        .def("ifit", &CobwebTree::ifit);
//         .def_readonly("num", &ContinuousValue::num)
//         .def_readonly("mean", &ContinuousValue::mean)
//         .def_readonly("meanSq", &ContinuousValue::meanSq);
}
