import libsvm._

/**
 * A simple scala wrapper for LibSVM
 */
class SvmClassifier {
  /**
   * Create a SVM problem.  Basically just inserting input features and outputs into an object.
   * @param outputs output classes.
   * @param inputs input features.
   * @return svm_problem object.
   */
  def createSVMProblem(outputs: IndexedSeq[Double], inputs: IndexedSeq[IndexedSeq[Double]]): svm_problem = {

    //must have same number of inputs and outputs
    require(outputs.size == inputs.size)

    val dataSize = inputs.size
    val numFeatures = inputs.head.size

    val prob = new svm_problem

    prob.l = dataSize

    //declare arrays for inputs and outputs
    prob.x = new Array[Array[svm_node]](dataSize)
    prob.y = new Array[Double](dataSize)

    (0 to dataSize - 1).foreach { exampleIdx =>

      prob.x(exampleIdx) = new Array[svm_node](numFeatures)

      (0 to numFeatures - 1).foreach { featureIdx =>

        val node = new svm_node()

        node.index = featureIdx
        node.value = inputs(exampleIdx)(featureIdx)
        prob.x(exampleIdx)(featureIdx) = node
      }

      prob.y(exampleIdx) = outputs(exampleIdx)
    }

    prob
  }


  /**
   * Train an SVM model with some training inputs and outputs.
   * @param outputs output classes.
   * @param inputs input features.
   * @return svm_model object.
   */
  def trainModel(outputs: IndexedSeq[Double], inputs: IndexedSeq[IndexedSeq[Double]]): svm_model = {

    val svmProb = createSVMProblem(outputs, inputs)

    //svm parameters
    val param = new svm_parameter()
    param.probability = 1
    param.gamma = 0.5
    param.nu = 0.5
    param.C = 1
    param.svm_type = svm_parameter.C_SVC
    param.kernel_type = svm_parameter.LINEAR
    param.cache_size = 20000
    param.eps = 0.001

    svm.svm_train(svmProb, param)
  }

  /**
   * Perform cross-validation
   * @param outputs output classes.
   * @param inputs input features
   * @param numFolds number of cross-validation folds.
   */
  def xValidation(outputs: IndexedSeq[Double], inputs: IndexedSeq[IndexedSeq[Double]], numFolds: Int) = {

    val svmProb = createSVMProblem(outputs, inputs)

    //svm paramters
    val param = new svm_parameter()
    param.probability = 1
    param.gamma = 0.5
    param.nu = 0.5
    param.C = 1
    param.svm_type = svm_parameter.C_SVC
    param.kernel_type = svm_parameter.LINEAR
    param.cache_size = 20000
    param.eps = 0.001

    val target = new Array[Double](svmProb.l)
    svm.svm_cross_validation(svmProb, param, numFolds, target)

    var totalError = 0.0
    var totalCorrect = 0.0

    if (param.svm_type == svm_parameter.EPSILON_SVR || param.svm_type == svm_parameter.NU_SVR) {

      for (i <- (0 to svmProb.l - 1)) {

        val y = svmProb.y(i)
        val v = target(i)

        totalError += (v - y) * (v - y)
      }

      println(s"Cross-Validation Mean Squared Error = ${totalError / svmProb.l}")

    }
    else {

      for (j <- (0 to svmProb.l - 1)) {
        if (target(j) == svmProb.y(j)) {
          totalCorrect += 1
        }
      }
      println(s"Cross-Validation Accuracy = ${100.0 * totalCorrect / svmProb.l}")
    }

  }

  /**
   * Predict class given input features and model.
   * @param model trained svm model
   * @param inputFeatures input features to predict from
   * @return predicted class
   */
  def svmPredict(model: svm_model, inputFeatures: IndexedSeq[Double]): Double = {

    val inputArray = new Array[svm_node](inputFeatures.size)

    (0 to inputFeatures.size - 1).foreach { idx =>

      val node = new svm_node
      node.index = idx
      node.value = inputFeatures(idx)

      inputArray(idx) = node
    }

    svm.svm_predict(model, inputArray)
  }


}
