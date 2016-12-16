# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation
import util
import random
PRINT = True

class PerceptronClassifier:
  """
  Perceptron classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "perceptron"
    self.max_iterations = max_iterations
    self.weights = {}
    isDigit = True if len(legalLabels) == 10 else False
    for label in legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
      #Initialize with random weights
      if (isDigit):
        for col in range(28):
          for row in range(28):
            self.weights[label][(col,row)] = random.uniform(0,1)
            #print "Initial weight: ", self.weights[label][(col,row)]
      else:
        for row in range(74):
          for col in range(60):
            self.weights[label][(col,row)] = random.uniform(0,1)
            #print "Initial weight: ", self.weights[label][(col,row)]
            
  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels);
    self.weights == weights;
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """
    #print "trainingData: "
    #print trainingData
    #print "validationData: "
    #print validationData
    #print type(trainingData[0])

    self.features = trainingData[0].keys() # could be useful later
    # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
    # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."
      list_of_indices = [i for i in range(len(trainingData))]
      while len(list_of_indices) > 0:
          "*** YOUR CODE HERE ***"
          random_datum_index = random.choice(list_of_indices) # Choose a datum to analyze
          list_of_indices.remove(random_datum_index) # Do not analyze this datum again
          training_datum = trainingData[random_datum_index] # Counter for a datum
          training_true_label = trainingLabels[random_datum_index] # True label for a datum
          computed_label = self.find_max_score_label(training_datum)
          if (computed_label == training_true_label):
            print "Correctly identified label ", computed_label, "!"
          else:
            print "Error: predicted ", computed_label, ", actual: ", training_true_label
            # Adjust weights for future iterations
            self.weights[training_true_label] += training_datum
            self.weights[computed_label] -= training_datum

          # util.raiseNotDefined()

  # Find label yielding max score for a given datum
  def find_max_score_label(self, my_datum):
    max_score = self.compute_score(my_datum, self.legalLabels[0])
    max_score_label = self.legalLabels[0]
    print "Initial max score: ", max_score
    print "Initial max label: ", max_score_label
    for i in range(1,len(self.legalLabels)):
      new_score = self.compute_score(my_datum, self.legalLabels[i])
      if new_score > max_score:
        max_score = new_score
        max_score_label = self.legalLabels[i];
        print "Updated max label: ", max_score_label
        print "Updated max score: ", max_score
    print "Max label: ", max_score_label
    print "Max score: ", max_score
    return max_score_label

  # Compute score for a given label
  def compute_score(self, feature_list, true_label):
    score = feature_list * self.weights[true_label]
    print "Computed score: ", score, "true label: ", true_label
    return score

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

  
  def findHighWeightFeatures(self, label):
    """
    Returns a list of the 100 features with the greatest weight for some label
    """
    featuresWeights = []

    "*** YOUR CODE HERE ***"
    features_list = self.weights[label]
    featuresWeights = features_list.sortedKeys()[:100]
    return featuresWeights # Sort by values and fetch top 100 features
    #util.raiseNotDefined()
    return featuresWeights

