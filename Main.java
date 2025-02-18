
//Matthew Lutz CS457, 10/4/2024


import java.io.*;
import java.util.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;



public class Main{
	

  public static void main(String args[]){
      String filename = null;
      //System.out.println(System.getProperty("user.dir"));

      
      int k = 1; //k-fold value
      int degreeMin = 1;
      int degreeMax = 1; 
      double alpha = 0.005; //learning rate
      int epochLimit = 10000;
      int batchSize = 0; 
      boolean randomize = false;
      int verbosity = 1;
      

      //cmd line args
      try {
          for (int i = 0; i < args.length; i++) {
              switch (args[i]) {
                  case "-f":
                      if (++i < args.length) {
                    	  filename = args[i];
                        System.out.println(filename);
                      }
                      else throw new IllegalArgumentException("Expected filename after -f");
                      break;
                  case "-k":
                      if (++i < args.length) k = Integer.parseInt(args[i]);
                      else throw new IllegalArgumentException("Expected integer k-fold value after -k");
                      break;
                  case "-d":
                      if (++i < args.length) degreeMin = Integer.parseInt(args[i]);
                      else throw new IllegalArgumentException("Expected integer min degree after -d");
                      break;
                  case "-D":
                      if (++i < args.length) degreeMax = Integer.parseInt(args[i]);
                      else throw new IllegalArgumentException("Expected integer max degree after -D");
                      break;
                  case "-a":
                      if (++i < args.length) alpha = Double.parseDouble(args[i]);
                      else throw new IllegalArgumentException("Expected double alpha value after -a");
                      break;
                  case "-e":
                      if (++i < args.length) epochLimit = Integer.parseInt(args[i]);
                      else throw new IllegalArgumentException("Expected integer epoch limit after -e");
                      break;
                  case "-m":
                      if (++i < args.length) batchSize = Integer.parseInt(args[i]);
                      else throw new IllegalArgumentException("Expected integer batch size after -m");
                      break;
                  case "-r":
                      randomize = true;
                      break;
                  case "-v":
                      if (++i < args.length) verbosity = Integer.parseInt(args[i]);
                      else throw new IllegalArgumentException("Expected integer verbosity level after -v");
                      break;
                  default:
                      System.out.println("Unknown or incomplete argument: " + args[i]);
                      break;
              }
          }
      } catch (NumberFormatException e) {
          System.err.println("Error parsing numerical argument: " + e.getMessage());
          return;
      } catch (IllegalArgumentException e) {
          System.err.println("Argument error: " + e.getMessage());
          return;
      }

      if (filename == null) {
          System.err.println("Filename must be specified with -f.");
          return;
      }
      
      if (degreeMax < degreeMin) {
          System.err.println("Maximum degree must be greater than or equal to minimum degree.");
          return;
      }
      
      List<DataPoint> data = null;

      //Read in the file
      if (filename != null) {
          data = readDataFromFile(filename);
          int numData = data.size();
          if (data != null) {
              //System.out.println("Data loaded successfully.");
              List<DataPoint> augmentedData = augmentData(data, degreeMax);
              double[] weights = miniBatchGradientDescent(augmentedData, verbosity, epochLimit, alpha, 0, batchSize, k, degreeMin, numData);

          } else {
              System.out.println("Failed to load data.");
              return;
          }   
      } else {
          System.out.println("No filename specified. Exiting.");
          return;
      }
      
      if (k > 1) {  //cross validation is specified
    	    List<List<DataPoint>> folds = splitIntoFolds(data, k, randomize);
    	    for (int degree = degreeMin; degree <= degreeMax; degree++) {
    	    	System.out.printf("Using %d fold cross-validation %n", k);
    	        System.out.println("-----------------------------");
    	        System.out.printf("* Using model of degree %d%n", degree);
    	        double totalTrainingError = 0;
    	        double totalValidationError = 0;

    	        for (int foldIndex = 0; foldIndex < k; foldIndex++) {
    	            List<DataPoint> trainingData = new ArrayList<>();
    	            List<DataPoint> validationData = folds.get(foldIndex);

    	            for (int i = 0; i < folds.size(); i++) {
    	                if (i != foldIndex) {
    	                    trainingData.addAll(folds.get(i));
    	                }
    	            }

    	            //augment data
    	            List<DataPoint> augmentedTrainingData = augmentData(trainingData, degree);
    	            List<DataPoint> augmentedValidationData = augmentData(validationData, degree);
    	            
    	            double[] weights = miniBatchGradientDescent(augmentedTrainingData, verbosity, epochLimit, alpha, 0, batchSize, k, degree, trainingData.size());
    	            
    	            double trainingError = calculateTrainingError(augmentedTrainingData, weights);
    	            double validationError = calculateTrainingError(augmentedValidationData, weights);
    	            
    	            totalTrainingError += trainingError;
    	            totalValidationError += validationError;

    	            System.out.printf("  * Training on all data except Fold %d (%d examples):%n", foldIndex + 1, trainingData.size());
    	            System.out.printf("  * Training and validation errors: %.6f %.6f%n", trainingError, validationError);
    	        }
    	        
    	        //calculate averages
    	        System.out.printf("  * Average errors across the folds: %.6f %.6f%n", totalTrainingError / k, totalValidationError / k);
    	    }
    	}

      
     
  }

  //method to read data from the file
  private static List<DataPoint> readDataFromFile(String filename) {
          List<DataPoint> data = new ArrayList<>();
          try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
              String line;
              while ((line = br.readLine()) != null) {
                  if (line.startsWith("#")) {
                      continue; //skip comments
                  }
                  String[] tokens = line.split("\\s+");
                  double[] features = new double[tokens.length - 1];
                  for (int i = 0; i < tokens.length - 1; i++) {
                      features[i] = Double.parseDouble(tokens[i]);
                  }
                  double output = Double.parseDouble(tokens[tokens.length - 1]);
                  data.add(new DataPoint(features, output));
              }
          } catch (IOException e) {
              e.printStackTrace();
              return null;
          }
          return data;
      }

  private static List<DataPoint> augmentData(List<DataPoint> originalData, int degree){
          List<DataPoint> augmentedData = new ArrayList<>();

          for (DataPoint point : originalData) {
              double[] originalFeatures = point.features;
              List<Double> augmentedFeatures = new ArrayList<>();
              
              augmentedFeatures.add(1.0);
              
              for (double feature : originalFeatures) {
                  augmentedFeatures.add(feature);
              }

              for (int d = 2; d <= degree; d++) {
                  for (double feature : originalFeatures) {
                      augmentedFeatures.add(Math.pow(feature, d));
                  }
              }

              double[] augmentedArray = augmentedFeatures.stream().mapToDouble(Double::doubleValue).toArray();
              
              augmentedData.add(new DataPoint(augmentedArray, point.output));
          }
          
          return augmentedData;
      }

  private static double[] miniBatchGradientDescent(List<DataPoint> data, int verbosity, int epochLimit, double alpha, int epoch, int batchSize, int k, int d, int numData){
      int numFeatures = data.get(0).features.length;
      double[] weights = new double[numFeatures]; 
      double totalCost = 0;
      double previousCost = Double.MAX_VALUE;
      long startTime = System.currentTimeMillis();
      int iterations =0;
      String sc = "";

      
      Random rand = new Random();
      for (int i = 0; i < numFeatures; i++) {
          weights[i] = rand.nextDouble() * 0.01; // init weights to small random vals
      }
      
      double initialCost = calculateTrainingError(data, new double[numFeatures]);
      
      List<Double> costs = new ArrayList<>();
      List<String> models = new ArrayList<>();

      //loop over epochs
      for(epoch = 0; epoch < epochLimit; epoch++){
          Collections.shuffle(data);
          List<List<DataPoint>> batches = createMiniBatches(data, batchSize);  //split data into batches
          double batchCost = 0;


          //loop through batches
          for(List<DataPoint> batch : batches){
              double[] gradient = new double[numFeatures];

              //loop through each data point
              for(DataPoint point : batch){
                  double prediction = 0.0;
                  for (int j = 0; j < numFeatures; j++) {
                      prediction += weights[j] * point.features[j];  //calculate predicted value based on the current weights
                  }
                  double error = prediction - point.output; //find the error
                  batchCost += error * error;

                  //update gradient
                  for(int j =0; j < numFeatures; j++){
                      gradient[j] += error * point.features[j];
                  }

                  //update weights using the gradient and alpha
                  for (int j = 0; j < numFeatures; j++) {
                      weights[j] -= alpha * gradient[j] / batch.size(); 
                  }
                  


              }
              /**
              for (int j = 0; j < numFeatures; j++) {
                  weights[j] -= alpha * gradient[j] / batch.size(); // Scale gradient by batch size
              }
              **/
              iterations += batch.size();
              
              totalCost += batchCost / batch.size();

          }
          
          totalCost /= batches.size();

          if (verbosity >= 3 && verbosity != 5 && (epoch % 1000 == 0)) {
              costs.add(totalCost/batches.size()); //store current cost
              models.add(modelToString(weights));
          }
          if(verbosity == 5) {
        	  costs.add(totalCost/batches.size()); //store current cost
              models.add(modelToString(weights));
          }
          
          if (Math.abs(previousCost - totalCost) < 1e-10) {
        	  sc = String.format("      GD Stop condtion: Delta cost ~= 0");
        	  costs.add(totalCost);
        	  models.add(modelToString(weights));
              break; 
          }else if( epoch > 10000) {
        	  sc = String.format("      GD Stop condtion: Iter > 10000");
        	  costs.add(totalCost);
        	  models.add(modelToString(weights));
        	  break;
          }
          previousCost = totalCost;

      }
      costs.add(totalCost);
      models.add(modelToString(weights));
      double trainingError = calculateTrainingError(data, weights);
      if(k <= 1) {
          outputTrainingProgress(costs, models, epoch, weights, alpha, sc, initialCost, totalCost, epochLimit, startTime, verbosity, k, d, numData, trainingError, batchSize, iterations);
      }

      return weights;

  }

  private static void outputTrainingProgress(List<Double> costs, List<String> models, int epoch, double[] weights, double alpha, String sc, double initialCost, double cost, int totalIterations, long startTime, int verbosity, int k, int d, int numData, double trainingError, int batchSize, int iterations) {
	  if(k < 1) {
		  System.out.println("Skipping cross-validation.");		  
	  }else {
		  System.out.println("Using " + k + "-fold cross-validation.");
	  }
	  System.out.println("-----------------------------");
      if (verbosity >= 1) {
          System.out.println("* Using model of degreee "+ d);
          System.out.println("  * Training on all data (" + numData + " examples)");
          if(verbosity >= 2) {
        	  System.out.println("   * Beginning mini-batch gradient descent");
        	  System.out.println("    (alpha=" + alpha + ", epochLimit=" + totalIterations + ", batchSize=" + batchSize);
        	  if(verbosity >=3) {
        		  System.out.println("    Initial model with zero weights   : Cost =    " + initialCost);
        		  for (int i = 0; i < costs.size(); i++) {
        			  if (verbosity == 5) {
        				  System.out.printf("    After %d epochs ( %d iter.): Cost = %.9f Model: %s%n", i , (i + 1), costs.get(i), models.get(i));
        			  }
        			  else if(verbosity >= 4) {
        				  System.out.printf("    After %d epochs ( %d iter.): Cost = %.9f Model: %s%n", i * 1000, (i + 1) * 1000, costs.get(i), models.get(i));
        			  }else{
            			  System.out.printf("    After %d epochs ( %d iter.): Cost = %.9f\n", i * 1000, (i + 1) * 1000, costs.get(i));
        			  }
        		  }
        	  }
        	  long endTime = System.currentTimeMillis();            
              System.out.println("   * Done with fitting! ");
              System.out.println("      Training took " + (endTime - startTime) + "ms, " + totalIterations + " epochs, " + iterations + " iterations (" + (endTime-startTime)/iterations + "ms / iteration) ");
          }
          if(verbosity >= 2) {
        	  System.out.println(sc);
          }
          if (verbosity >= 4) {
              System.out.printf("      Model: Y = %s%n", modelToString(weights));
          }
          System.out.println("  * Training error:      " + trainingError);
      }
  }
  
  private static String modelToString(double[] weights) {
	  if (weights == null || weights.length == 0) {
		  return "No model available.";
	  }

	    StringBuilder model = new StringBuilder();
	    if (weights.length > 0) {
	        model.append(String.format("%.4f", weights[0]));
	    }

	    for (int i = 1; i < weights.length; i++) {
	        if (weights[i] == 0) {
	            continue; 
	        }
	        if (weights[i] > 0) {
	            model.append(" + ");
	        } else {
	            model.append(" ");
	        }
	        if (Math.abs(weights[i]) == 1) {
	            model.append(String.format("%sX%d", (weights[i] == -1 ? "-" : ""), i));
	        } else {
	            model.append(String.format("%.4fX%d", weights[i], i));
	        }
	    }
	    return model.toString();
	}

  //this method creates the mini batches 
  public static List<List<DataPoint>> createMiniBatches(List<DataPoint> data, int batchSize) {
	    List<List<DataPoint>> batches = new ArrayList<>();
	    if (batchSize <= 0) {
	        batches.add(data);
	        return batches;
	    }

	    int start = 0;
	    while (start < data.size()) {
	        int end = Math.min(data.size(), start + batchSize);
	        batches.add(new ArrayList<>(data.subList(start, end)));
	        start = end;
	    }

	    return batches;
  }
  
  //this method calculates the training error
  private static double calculateTrainingError(List<DataPoint> data, double[] weights) {
	  double sumError = 0;
	  System.out.println();
	  for (DataPoint point : data) {
		  double prediction = 0.0;
		  for (int j = 0; j < weights.length; j++) {
			  prediction += weights[j] * point.features[j];
		  }
		  double error = prediction - point.output;
		  sumError += error * error;
		  //System.out.printf("), Prediction: %.4f, Actual: %.4f, Error: %.4f%n", prediction, point.output, error);

	  }
	  return sumError / data.size(); //mse
  }

  //this method splits the data in k folds 
  public static List<List<DataPoint>> splitIntoFolds(List<DataPoint> data, int k, boolean randomize) {
	  if (randomize) {
		  Collections.shuffle(data);
	  }

	  List<List<DataPoint>> folds = new ArrayList<>();
	  int n = data.size();
	  int foldSize = n / k;
	  int remainder = n % k;
	  
	  int start = 0;
	  for (int i = 0; i < k; i++) {
		  int end = start + foldSize + (i < remainder ? 1 : 0);
		  folds.add(new ArrayList<>(data.subList(start, end)));
		  start = end;
	  }
	  return folds;
  }
  
  
  static class DataPoint {
      double[] features;
      double output;

      DataPoint(double[] features, double output) {
          this.features = features;
          this.output = output;
      }
  }
  
}
