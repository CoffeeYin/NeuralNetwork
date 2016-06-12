/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   A I R F O I L   S E L F - N O I S E   A P P L I C A T I O N                                                */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - The artificial intelligence company                                                            */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// This is a function regression problem.

// System includes

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{
   try
   {
      std::cout << "OpenNN. Airfoil Self-Noise Application." << std::endl;

      srand((unsigned)time(NULL));

      // Data set

      DataSet data_set;

      data_set.set_data_file_name("../data/mytrain.dat");

      data_set.set_separator("Tab");

      data_set.load_data();
       
       std::ofstream f("../data/MSE.dat");

      // Variables

      Variables* variables_pointer = data_set.get_variables_pointer();

      Vector< Variables::Item > variables_items(3);

      variables_items[0].name = "x1";
      variables_items[0].units = "none";
      variables_items[0].use = Variables::Input;

      variables_items[1].name = "x2";
      variables_items[1].units = "none";
      variables_items[1].use = Variables::Input;

      variables_items[2].name = "(x1^2+x2^2)/2";
      variables_items[2].units = "meters";
      variables_items[2].use = Variables::Target;
       
       std::vector<double> MSE;

      variables_pointer->set_items(variables_items);

      const Matrix<std::string> inputs_information = variables_pointer->arrange_inputs_information();
      const Matrix<std::string> targets_information = variables_pointer->arrange_targets_information();

      // Instances

      Instances* instances_pointer = data_set.get_instances_pointer();

      instances_pointer->split_random_indices();

      const Vector< Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();
      const Vector< Statistics<double> > targets_statistics = data_set.scale_targets_minimum_maximum();

      // Neural network
       for (int i = 5;i<16;i++){
           const size_t inputs_number = variables_pointer->count_inputs_number();
           const size_t hidden_perceptrons_number = i;
           const size_t outputs_number = variables_pointer->count_targets_number();
           
           NeuralNetwork neural_network(inputs_number, hidden_perceptrons_number, outputs_number);
           
           Inputs* inputs = neural_network.get_inputs_pointer();
           
           inputs->set_information(inputs_information);
           
           Outputs* outputs = neural_network.get_outputs_pointer();
           
           outputs->set_information(targets_information);
           
           neural_network.construct_scaling_layer();
           
           ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
           
           scaling_layer_pointer->set_statistics(inputs_statistics);
           
           scaling_layer_pointer->set_scaling_method(ScalingLayer::NoScaling);
           
           neural_network.construct_unscaling_layer();
           
           UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();
           
           unscaling_layer_pointer->set_statistics(targets_statistics);
           
           unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::NoUnscaling);
           
           // Performance functional
           
           PerformanceFunctional performance_functional(&neural_network, &data_set);
           
           performance_functional.set_regularization_type(PerformanceFunctional::NEURAL_PARAMETERS_NORM);
           
           // Training strategy object
           
           TrainingStrategy training_strategy(&performance_functional);
           
           QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();
           
           quasi_Newton_method_pointer->set_maximum_iterations_number(1000);
           quasi_Newton_method_pointer->set_display_period(10);
           
           quasi_Newton_method_pointer->set_minimum_performance_increase(1.0e-7);  //original 1.0e-6
           
           quasi_Newton_method_pointer->set_reserve_performance_history(true);
           
           TrainingStrategy::Results training_strategy_results = training_strategy.perform_training();
           
           // Testing analysis
           
           TestingAnalysis testing_analysis(&neural_network, &data_set);
           
           TestingAnalysis::LinearRegressionResults linear_regression_results = testing_analysis.perform_linear_regression_analysis();
           
           
           // Check Neural Network
           DataSet check_data_set;
           Matrix<double> check_matrix_set;
           check_matrix_set.load("../data/mytraining_check.dat");
           
           Matrix<double> check_matrix_set_target;
           check_matrix_set_target.load("../data/mytraining_check.dat");
           check_matrix_set_target.subtract_column(0);
           check_matrix_set_target.subtract_column(0);

           check_data_set.set_data_file_name("../data/mytraining_check.dat");
           check_data_set.set_separator("Space");
           check_data_set.load_data();
           
           Variables* check_variables_pointer = check_data_set.get_variables_pointer();
           
           Vector< Variables::Item > check_variables_items(3);
           
           check_variables_items[0].name = "x1";
           check_variables_items[0].units = "none";
           check_variables_items[0].use = Variables::Input;
           
           check_variables_items[1].name = "x2";
           check_variables_items[1].units = "none";
           check_variables_items[1].use = Variables::Input;
           
           check_variables_pointer->set_items(check_variables_items);
           
           const Matrix<std::string> check_inputs_information = check_variables_pointer->arrange_inputs_information();
           
           
           Instances* check_instances_pointer = check_data_set.get_instances_pointer();
           
           check_instances_pointer->split_random_indices();
           
           //const Vector< Statistics<double> > check_inputs_statistics = check_data_set.scale_inputs_minimum_maximum();
           
           const Matrix< double > check_result = neural_network.calculate_output_data(check_matrix_set);
           Vector<double> myresult = check_result.calculate_rows_sum();
           
           //check_matrix_set_target.append_column(myresult);
//           double mySum;
//           for(int j=0;j<inputs_number;j++){
//               mySum = (check_matrix_set[0][j]-check_matrix_set[2][j])^2;
//           }
           //Matrix<double> target_and_result = check_matrix_set_target;
           
           double myMse = check_matrix_set_target.calculate_sum_squared_error(check_result)/inputs_number;
           
           f << i  <<"    "<< myMse << '\n';
           
           std::string save_check_file_name = "../data/mytraning_check_result" + std::to_string(i);
           
           check_result.save(save_check_file_name);
           
           //check_result.save("../data/mytraning_check_result");
           
           scaling_layer_pointer->set_scaling_method(ScalingLayer::MinimumMaximum);
           unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::MinimumMaximum);
           
           data_set.save("../data/mytraining_data_set.xml"+ std::to_string(i));
           
           neural_network.save("../data/mytraining_neural_network.xml"+ std::to_string(i));
           neural_network.save_expression("../data/mytraining_expression.txt"+ std::to_string(i));
           
           performance_functional.save("../data/mytraining_performance_functional.xml"+ std::to_string(i));
           
           training_strategy.save("../data/mytraining_training_strategy.xml"+ std::to_string(i));
           training_strategy_results.save("../data/mytraining_training_strategy_results.dat"+ std::to_string(i));
           
           linear_regression_results.save("../data/mytraining_linear_regression_analysis_results.dat"+ std::to_string(i));
       }
       
       
       f.close();
       
       
       return 0;
       // Save results
//
//      scaling_layer_pointer->set_scaling_method(ScalingLayer::MinimumMaximum);
//      unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::MinimumMaximum);
//
//      data_set.save("../data/mytraining_data_set.xml");
//
//      neural_network.save("../data/mytraining_neural_network.xml");
//      neural_network.save_expression("../data/mytraining_expression.txt");
//
//      performance_functional.save("../data/mytraining_performance_functional.xml");
//
//      training_strategy.save("../data/mytraining_training_strategy.xml");
//      training_strategy_results.save("../data/mytraining_training_strategy_results.dat");
//
//      linear_regression_results.save("../data/mytraining_linear_regression_analysis_results.dat");

      return(0);
   }
   catch(std::exception& e)
   {
      std::cerr << e.what() << std::endl;

      return(1);
   }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2015 Roberto Lopez.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
