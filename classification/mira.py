# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation
import util
import math
import random
import sys
PRINT = True

class MiraClassifier:
  """
  Mira classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mira"
    self.automaticTuning = False 
    self.C = 0.001
    self.legalLabels = legalLabels
    self.max_iterations = max_iterations
    self.initializeWeightsToZero()

  def initializeWeightsToZero(self):
    "Resets the weights of each label to zero vectors" 
    self.weights = {}
    for label in self.legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method."  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    
    if (self.automaticTuning):
        Cgrid = [0.002, 0.004, 0.008]
    else:
        Cgrid = [self.C]
        
    return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
    """
    This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
    then store the weights that give the best accuracy on the validationData.
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    representing a vector of values.
    """
    "*** YOUR CODE HERE ***"
    
    validation_passes = util.Counter()  # Counter mapping C-> validation testing successes
    validation_weights = {} # Counter mapping C-> counter representing the weight vector for C 
    for c_value in Cgrid:
      validation_weights[c_value] = [util.Counter() for i in range(len(self.legalLabels))]

    for c_index in range(len(Cgrid)):
      c_weights = [util.Counter() for i in range(len(self.legalLabels))] # Used to test each C value and run validation data
      for iteration in range(self.max_iterations):
        print "Starting iteration ", iteration, " for C = ", Cgrid[c_index], "..."
        # Randomize training data
        list_of_indices = [i for i in range(len(trainingData))]
        while len(list_of_indices) > 0:
          "*** YOUR CODE HERE ***"
          random_datum_index = random.choice(list_of_indices) # Choose a datum to analyze
          list_of_indices.remove(random_datum_index) # Do not analyze this datum again
          training_datum = trainingData[random_datum_index] # Counter for a datum
          training_true_label = trainingLabels[random_datum_index] # True label for a datum
          
          ##### MAX SCORE ARG CALCULATION
          score_counter = util.Counter()
          for label in self.legalLabels:
            score_counter[label] = training_datum * c_weights[label]
          computed_label = score_counter.argMax()
          ##### END MAX SCORE ARG CALCULATION

          #computed_label = self.find_max_score_label(training_datum)
          
          if (computed_label == training_true_label):
            #print "Correctly identified label ", computed_label, "!"
            pass
	  else:
            #print "Error: predicted ", computed_label, ", actual: ", training_true_label
            # Adjust weights for future iterations
            feature_scale_factor_num = (c_weights[computed_label] - c_weights[training_true_label]) * training_datum + 1.0
            raw_norm = 0
            for key in training_datum.keys():
              raw_norm += math.pow(training_datum[key], 2)
            feature_scale_factor_denom = 2 * raw_norm
            
            #print "Numer: ", feature_scale_factor_num
            #print "Denom: ", feature_scale_factor_denom
            feature_scale_factor = feature_scale_factor_num / feature_scale_factor_denom
            #print "Num/denom: ", feature_scale_factor
            feature_scale_factor = feature_scale_factor if (feature_scale_factor < Cgrid[c_index]) else Cgrid[c_index] 

            #Scale weight vectors accordingly
            c_weights[training_true_label] += training_datum.multiply_by_scalar(feature_scale_factor)
            c_weights[computed_label] -= training_datum.multiply_by_scalar(feature_scale_factor)
        
      # Weight vector for a specific C value is complete; run validation tests
      for l in self.legalLabels:
        self.weights[l] = c_weights[l] # Set for testing
      guesses = self.classify(validationData)
      validation_passes[Cgrid[c_index]] = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True) # Store results in validation_passes
      for j in self.legalLabels:  # Store weight vector in validation_weights
        validation_weights[Cgrid[c_index]][j] = c_weights[j]
    
    # All training and validation is done -> set optimal weight vector and C value
    self.C = validation_passes.argMax()
    for label in self.legalLabels:
      self.weights[label] = validation_weights[self.C][label]
    
    #util.raiseNotDefined()

  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns a list of the 100 features with the greatest difference in feature values
                     w_label1 - w_label2

    """
    featuresOdds = []

    "*** YOUR CODE HERE ***"
    diff_weights = self.weights[label1] - self.weights[label2]
    for key in diff_weights:
	diff_weights[key] = math.fabs(diff_weights[key]) # Account for negative counter values
    sorted_keys = diff_vector.sortedKeys()
    for key in sortedKeys[:100]:
	featuresOdds.append(diff_weights[key])

    return featuresOdds

