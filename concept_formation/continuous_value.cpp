#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <math.h>
#include <map>
#include <stdexcept>

namespace py = pybind11;

class ContinuousValue {

public:

    int num;
    double mean;
    double meanSq;

    ContinuousValue() {
        num = 0;
        mean = 0.0;
        meanSq = 0.0;     
    }

    ContinuousValue(const int num, const double mean, const double meanSq) : ContinuousValue() {
        this->num = num;
        this->mean = mean;
        this->meanSq = meanSq;     
    }

    bool is_cpp() const {
        return true;
    }

    double c4(const int n) const {
        if (n <= 1){
            throw std::invalid_argument("Cannot apply correction for a sample size of 1.");
        }

        switch(n) {
            case 2:
                return 0.7978845608028654;
            case 3:
                return 0.886226925452758;
            case 4:
                return 0.9213177319235613;
            case 5:
                return 0.9399856029866254;
            case 6:
                return 0.9515328619481445;
            case 7:
                return 0.9593687886998328;
            case 8:
                return 0.9650304561473722;
            case 9:
                return 0.9693106997139539;
            case 10:
                return 0.9726592741215884;
            case 11:
                return 0.9753500771452293;
            case 12:
                return 0.9775593518547722;
            case 13:
                return 0.9794056043142177;
            case 14:
                return 0.9809714367555161;
            case 15:
                return 0.9823161771626504;
            case 16:
                return 0.9834835316158412;
            case 17:
                return 0.9845064054718315;
            case 18:
                return 0.985410043808079;
            case 19:
                return 0.9862141368601935;
            case 20:
                return 0.9869342675246552;
            case 21:
                return 0.9875829288261562;
            case 22:
                return 0.9881702533158311;
            case 23:
                return 0.988704545233999;
            case 24:
                return 0.9891926749585048;
            case 25:
                return 0.9896403755857028;
            case 26:
                return 0.9900524688409107;
            case 27:
                return 0.990433039209448;
            case 28:
                return 0.9907855696217323;
            case 29:
                return 0.9911130482419843;
            default:
                return 1.0;
        }
    }

    ContinuousValue copy() const {
        return ContinuousValue(num, mean, meanSq);
    }

    double unbiased_mean() const {
        return mean;
    }

    double scaled_unbiased_mean(const double shift, double scale) const {
        if (scale <= 0){
            scale = 1.0;
        }
        return (mean - shift) / scale;
    }

    double biased_std() const {
        return sqrt(meanSq / num);
    }

    double unbiased_std() const {
        if (num < 2){
            return 0.0;
        }
        return sqrt(meanSq / (num - 1)) / c4(num);
    }

    double scaled_biased_std(double scale) const {
        if (scale <= 0){
            scale  = 1.0;
        }
        return biased_std() / scale;
    }

    double scaled_unbiased_std(double scale) const {
        if (scale <= 0){
            scale  = 1.0;
        }
        return unbiased_std() / scale;
    }

    long __hash__() const {
        return 8636487271284131744;
    } 

    std::string __repr__() const {
        return "" + std::to_string(unbiased_mean()) + " (" + std::to_string(unbiased_std()) + ") [" + std::to_string(num) + "]";
    } 

    int __len__() const {
        return 1;
    } 

    void update(const double x){
        num += 1;
        double delta = x - mean;
        mean += delta / num;
        meanSq += delta * (x - mean);
    }

    void update_batch(const std::vector<double> data){
        for (double ele : data){
            update(ele);
        }
    }

    void combine(const ContinuousValue other){
        double delta = other.mean - mean;
        meanSq = (meanSq + other.meanSq + delta * delta * ((num * other.num) / (num + other.num)));
        mean = ((num * mean + other.num * other.mean) / (num + other.num));
        num += other.num;
    }

    double integral_of_gaussian_product(const ContinuousValue other) const {
        double mu1 = this->unbiased_mean();
        double sd1 = this->unbiased_std();

        double mu2 = other.unbiased_mean();
        double sd2 = other.unbiased_std();

        double noisy_sd_squared = 1.0 / (4 * M_PI);
        sd1 = sqrt(sd1 * sd1 + noisy_sd_squared);
        sd2 = sqrt(sd2 * sd2 + noisy_sd_squared);

        return ((1 / sqrt(2 * M_PI * (sd1 * sd1 + sd2 * sd2))) *
                exp(-1 * (mu1 - mu2) * (mu1 - mu2) /
                    (2 * (sd1 * sd1 + sd2 * sd2))));

    }
    
    std::map<std::string, double> output_json() const {
        std::map<std::string, double> out;
        out.insert(std::pair<std::string, double>("mean", this->unbiased_mean()));
        out.insert(std::pair<std::string, double>("std", this->unbiased_std()));
        out.insert(std::pair<std::string, double>("n", static_cast<double>(num)));
        return out;
    }

};

PYBIND11_MODULE(continuous_value, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::class_<ContinuousValue>(m, "ContinuousValue")
        .def(py::init())
        .def("is_cpp", &ContinuousValue::is_cpp)
        .def("c4", &ContinuousValue::c4)
        .def("copy", &ContinuousValue::copy)
        .def("unbiased_mean", &ContinuousValue::unbiased_mean)
        .def("scaled_unbiased_mean", &ContinuousValue::scaled_unbiased_mean)
        .def("biased_std", &ContinuousValue::biased_std)
        .def("unbiased_std", &ContinuousValue::unbiased_std)
        .def("scaled_biased_std", &ContinuousValue::scaled_biased_std)
        .def("scaled_unbiased_std", &ContinuousValue::scaled_unbiased_std)
        .def("__hash__", &ContinuousValue::__hash__)
        .def("__repr__", &ContinuousValue::__repr__)
        .def("__len__", &ContinuousValue::__len__)
        .def("update", &ContinuousValue::update)
        .def("update_batch", &ContinuousValue::update_batch)
        .def("combine", &ContinuousValue::combine)
        .def("integral_of_gaussian_product", &ContinuousValue::integral_of_gaussian_product)
        .def("output_json", &ContinuousValue::output_json)
        .def_readonly("num", &ContinuousValue::num)
        .def_readonly("mean", &ContinuousValue::mean)
        .def_readonly("meanSq", &ContinuousValue::meanSq);
}
