#include "LinearSvmModel.hpp"

#include "detector_model.pb.h"

#include "helpers/Log.hpp"

#include <boost/filesystem.hpp>

/*#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/support_multi_pass.hpp>
#include <boost/spirit/include/phoenix.hpp>
#include <boost/bind.hpp>
*/

#include <stdexcept>
#include <fstream>
#include <iterator>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "LinearSvmModel");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "LinearSvmModel");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "LinearSvmModel");
}

} // end of anonymous namespace


namespace doppia {

using namespace std;
using namespace boost;

//namespace qi = boost::spirit::qi;

LinearSvmModel::LinearSvmModel()
{
    // w is left empty
    bias = 0;
    return;
}

LinearSvmModel::LinearSvmModel(const doppia_protobuf::DetectorModel &model)
{

    if(model.has_detector_name())
    {
        log_info() << "Parsing model " << model.detector_name() << std::endl;
    }

    if(model.detector_type() != doppia_protobuf::DetectorModel::LinearSvm)
    {
        throw std::runtime_error("Received model is not of the expected type LinearSvm");
    }

    if(model.has_linear_svm_model() == false)
    {
        throw std::runtime_error("The model content does not match the model type");
    }

    throw std::runtime_error("LinearSvmModel from doppia_protobuf::DetectorModel not yet implemented");

    return;
}

LinearSvmModel::LinearSvmModel(const Eigen::VectorXf &w_, const float bias_)
{
    w = w_;
    bias = bias_;
    return;
}

LinearSvmModel::LinearSvmModel(const std::string &filename)
{

    if(filesystem::exists(filename) == false)
    {
        log_error() << "Could not find the linear SVM model file " << filename << std::endl;
        throw std::invalid_argument("Could not find linear SVM model file");
    }

    log_info() << "Parsing linear svm model " << filename << std::endl;

    // open and parse the file
    std::ifstream model_file(filename.c_str());
    if(model_file.is_open() == false)
    {
        log_error() << "Could not open the linear SVM model file " << filename << std::endl;
        throw std::runtime_error("Failed to open the  linear SVM model file");
    }
    parse_libsvm_model_file(model_file);
    return;
}

LinearSvmModel::~LinearSvmModel()
{
    // nothing to do here
    return;
}


float LinearSvmModel::get_bias() const
{
    return bias;
}

const Eigen::VectorXf &LinearSvmModel::get_w() const
{
    return w;
}

/*
struct libsvm_grammar_result
{
    float bias;
    Eigen::VectorXf w;
};

void resize_weight_vector(Eigen::VectorXf &w, int size)
{
    w = Eigen::VectorXf::Zero(size);
    return;
}


void add_to_weight_vector(int &i, Eigen::VectorXf &w, float &value)
{
    w(i) = value;
    i += 1;
    return;
}


void print_int(const int& i)
{
    log_info() << "print_int " << i << std::endl;
}

void print_string(const string& s)
{
    log_info() << "print_string " << s << std::endl;
}

void std_vector_to_eigen(const vector<float> &std_v, Eigen::VectorXf &w)
{
    w.setZero(std_v.size());

    for(unsigned int i=0; i < std_v.size(); i+=1)
    {
        w(i) = std_v[i];
    }

    return;
}


template <typename Iterator>
struct libsvm_grammar : qi::grammar<Iterator>
{

    libsvm_grammar(libsvm_grammar_result &result, int &w_size) : libsvm_grammar::base_type(start)
    {
        using qi::eps;
        using qi::lit;
        using qi::_val;
        using qi::_1;
        using qi::lexeme;
        using qi::eol;
        using qi::int_;
        using qi::float_;
        using qi::char_;
        using boost::phoenix::push_back;
        using boost::phoenix::ref;

        w_size = 0;

        solver_type = lit("solver_type") >> +(char_ - eol) >> eol;
        nr_classes = lit("nr_class") >> int_ >> eol;
        labels = lit("label") >> +int_[push_back(_val, _1)]; // >> eol;
        nr_features = lit("nr_feature") >> int_ ; // >> eol;
        bias = lit("bias") >>  float_ ; // >> eol;
        weights = lit("w") >> eol >> +float_ ; // >> eol;

        start = (
                    solver_type[print_string] >>
                    nr_classes[print_int] >>
                    labels >>
                    nr_features[boost::bind(&resize_weight_vector, boost::ref(result.w), ::_1)] >>
                    nr_features[print_int] >>
                    bias [ ref(result.bias) = _1]>>
                    weights[boost::bind(&std_vector_to_eigen, ::_1, boost::ref(result.w))] >>
                    *char_ >>
                    qi::eoi );

        //rl = alpha[_a = _1] >> char_(_a); // get two identical characters
    }


    qi::rule<Iterator> start;
    qi::rule<Iterator, string() > solver_type;
    qi::rule<Iterator, int() > nr_classes;
    qi::rule<Iterator, vector<int>() > labels;
    qi::rule<Iterator, int() > nr_features;
    qi::rule<Iterator, float() > bias;
    qi::rule<Iterator, vector<float>() > weights;

};

void parse_using_boost_spirit(std::ifstream &model_file, float &bias, Eigen::VectorXf &w)
{

    typedef std::istreambuf_iterator<char> file_iterator_t;
    spirit::multi_pass<file_iterator_t>
            first = spirit::make_default_multi_pass(file_iterator_t(model_file)),
            last = spirit::make_default_multi_pass(file_iterator_t());

    // multipass seems to fail, using in memory string instead (slower, heavier solution)
    std::stringstream string_buffer;
    string_buffer << model_file.rdbuf();
    string file_content = string_buffer.str();


    {
         string test_content = "solver_type L2R_L2LOSS_SVC_DUAL\n";
         string::iterator first = test_content.begin();
         string solver_type;
         const bool result = qi::phrase_parse(
                     first, test_content.end(),
                     (
                       qi::lit("solver_type") >> (+qi::char_)[print_string] >> qi::eoi
                     ), spirit::ascii::space, solver_type);
         log_info() << "test_content solver_type == " << solver_type << std::endl;
         log_info() << "test_content result == " << result << std::endl;


         test_content = "a\nb\nc\n";
         first = test_content.begin();
         const bool result11 = qi::phrase_parse(
                     first, test_content.end(),
                     (
                       *((*qi::char_("a-z")) >> qi::eol) >> qi::eoi
                     ), spirit::ascii::space);
         log_info() << "test_content result1.1 == " << result11 << std::endl;

         int w_size;
         libsvm_grammar_result parsed_model;
         //libsvm_grammar<string::iterator> parser;
         libsvm_grammar<string::iterator> parser(parsed_model, w_size);

         qi::rule<string::iterator, string() > solver_type_rule;
         qi::rule<string::iterator, int() > nr_classes;
         qi::rule<string::iterator, vector<int>() > labels;
         qi::rule<string::iterator> start_rule;

         solver_type_rule = qi::lit("solver_type") >> +(qi::char_("a-zA-Z0-9")| qi::char_("_")) ;
         nr_classes = qi::lit("nr_class") >> qi::int_;
         labels = qi::lit("label") >> +qi::int_[phoenix::push_back(qi::_val, qi::_1)];

         start_rule = ( solver_type_rule[print_string] >> qi::eol >>
                       nr_classes[print_int] >> qi::eol >>
                       labels >> qi::eol >>
                       *qi::char_ >> qi::eoi );

         test_content = "solver_type L2R_L2LOSS_SVC_DUAL\nnr_class 2\nlabel -1 +1\n";
         first = test_content.begin();
         const bool result2 = qi::phrase_parse(
                     first, test_content.end(),
                     //parser,
                     start_rule,
                     spirit::ascii::space);
         log_info() << "test_content result2 == " << result2 << std::endl;

    }

    //log_debug() << "file_content == " << file_content << std::endl;
    string::iterator content_begin = file_content.begin(),
            content_end = file_content.end();

    {
        int w_size;
        libsvm_grammar_result parsed_model;
        //libsvm_grammar<string::iterator> parser;
        libsvm_grammar<string::iterator> parser(parsed_model, w_size);

        log_info() << "Starting file parsing" << std::endl;
        const bool result = qi::phrase_parse(
                    //first,last,
                    content_begin, content_end,
                    parser, spirit::ascii::space); // parsed_model);


        assert(parsed_model.w.size() == w_size);
        bias = parsed_model.bias;
        w = parsed_model.w;

        log_debug() << "Read w of size " << w.size() << std::endl << w << std::endl;

        //if (not result or first != last)
        //if (not result or content_begin != content_end)
        if (result == false)
        { // fail if we did not get a full match
            throw std::runtime_error("LibSVM model parsing failed");
        }
    }

    return;
}
*/

void hardcoded_parsing(std::ifstream &model_file, float &bias, Eigen::VectorXf &w)
{
    //solver_type L2R_L2LOSS_SVC_DUAL
    //nr_class 2
    //label 1 -1
    //nr_feature 5120
    //bias -1
    //w
    //-9.928359971228299e-05
    // -1.046861188325508e-05
    // ...


    string t_string, solver_type;
    int nr_class, nr_feature;
    vector<int> labels;

    model_file >> t_string >> solver_type;
    model_file >> t_string >> nr_class;
    labels.resize(nr_class);
    model_file >> t_string; // label
    for(int i=0;i < nr_class; i+=1)
    {
        model_file >> labels[i];
    }
    model_file >> t_string >> nr_feature;
    w.setZero(nr_feature);
    model_file >> t_string >> bias;
    model_file >> t_string; // w
    for(int i=0;i < nr_feature; i+=1)
    {
        model_file >> w(i);
    }

    log_debug() << "Read a model trained using == " << solver_type << std::endl;
    log_debug() << "Read bias == " << bias << std::endl;
    log_debug() << "Read w of size " << w.size() << std::endl; // << w << std::endl;

    return;
}


void LinearSvmModel::parse_libsvm_model_file(std::ifstream &model_file)
{

    //parse_using_boost_spirit(model_file, bias, w);
    hardcoded_parsing(model_file, bias, w);

    return;
}

float LinearSvmModel::compute_score(const Eigen::VectorXf &feature_vector) const
{
    assert(w.size() > 0);
    return feature_vector.dot(w) - bias;
}

} // end of namespace doppia
