#include <iostream>
#include <tuple>
#include <vector>
#include <string>
#include <random>
using namespace std;

int main() {
    return 0;
}

string weighted_choice(vector<tuple<string, double>> choices) {
    if(choices.size() <= 0) {
        throw invalid_argument("Choices cannot be an empty list");
    }
    double upto, total = 0.0;
    uniform_real_distribution<double> dist(0.0, 1.0);
    for(tuple<string, double> c: choices) {
        total += get<1>(c);
    }
    double r = dist.a();
    for(tuple<string, double> c: choices) {
        if(get<1>(c) < 0) {
            throw invalid_argument("All weights must be greater than or equal to 0.");
        }
        if(upto + get<1>(c) > r) {
            return get<0>(c);
        }
        upto += get<1>(c);
    }
    return "";
}

string most_likely_choice(vector<tuple<string, double>> choices) {
    if(choices.size() <= 0) {
        throw invalid_argument("Choices cannot be an empty list");
    }
    if(choices.size() == 1) {
        return get<0>(choices[0]);
    }
    double upto, total = 0.0;
    tuple<string, double> bC = choices[0];
    for(int i = 1; i < choices.size(); ++i) {
        if(get<1>(choices[i]) < 0) {
            throw invalid_argument("All weights must be greater than or equal to 0.");
        }
        if(get<1>(choices[i]) > get<1>(bC)) {
            bC = choices[i];
        }
    }
    return get<0>(bC);
}