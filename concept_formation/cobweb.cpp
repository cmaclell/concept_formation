#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

using namespace std;

class CobwebTree {

};

class CobwebNode {

    private:

        static int counter;

        int conceptID;
        double count = 0.0;
        int avCounts;
        vector<CobwebNode> children;
        CobwebNode parent;

        int gensym() {
            return ++CobwebNode::counter;
        }

    CobwebNode() {
        conceptID = gensym();
    }
    // CobwebNode parent;
    // CobwebTree tree;

};

int CobwebNode::counter = 0;

int main() {

    return 0;
}



