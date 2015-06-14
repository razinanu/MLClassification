package org.weka;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.trees.J48;
import weka.core.Instances;

/**
 * \brief This class is for using two different machine learning algorithms to
 * classify the sets.
 * The methods are J48, which is based on decision trees, and
 * Support Vector Machines.
 * @param Instances
 *            trainSet
 * @param Instances
 *            testSet
 */
public class Classifiers {

	
	public void classifier(Instances trainSet, Instances testSet)
			throws Exception {

		buildSVMmodel(trainSet, testSet);

		buildDecisionTreemodel(trainSet, testSet);
	}

	/**
	 * \brief Building the decision tree model and classify the train and test
	 * sets.
	 * 
	 * 
	 * It uses two fold cross validation to classify the train set and builds the
	 * model. In order to estimate the classification quality, two methods have
	 * been used. The multi-class logarithmic loss and the evaluation function
	 * from WEKA library.
	 * 
	 * @param Instances
	 *            trainSet
	 * @param Instances
	 *            testSet
	 */
	private void buildDecisionTreemodel(Instances trainSet, Instances testSet)
			throws Exception, IOException, FileNotFoundException {
		Evaluation validationJ48 = new Evaluation(trainSet);
		try {

			// read the model from local file
			Classifier clsVal = (Classifier) weka.core.SerializationHelper
					.read("lib/J48.model");
			System.out
					.print("multi-class logarithmic loss of decision tree is: ");
			// Evaluate the classification quality with multi-class logarithmic
			// loss
			evaluateModelLogLoss(clsVal, trainSet);

			// Evaluate the decision tree model
			validationJ48.evaluateModel(clsVal, trainSet);

			System.out.println(validationJ48.toSummaryString(
					"\nResults of decision tree classifier\n======\n", false));
			// predict the label for test data
			labeleJ48TestSet(testSet, clsVal);

		} catch (FileNotFoundException exp) {

			System.out.println("Build the decision tree model...");
			// Decision tree classifier
			Classifier cls = new J48();
			cls.buildClassifier(trainSet);
			// cross validation using 2 folds to estimate the classification
			// quality,it picks randomly the instances
			validationJ48.crossValidateModel(cls, trainSet, 2, new Random(1));
			System.out
					.print("multi-class logarithmic loss of decision tree is: ");
			// Evaluate the classification quality with multi-class logarithmic
			// loss
			evaluateModelLogLoss(cls, trainSet);
			validationJ48.evaluateModel(cls, trainSet);
			System.out.println(validationJ48.toSummaryString(
					"\nResults of decision tree classifier\n======\n", false));
			// validationJ48.evaluateModelOnceAndRecordPrediction(cls, testSet);
			// save the model
			ObjectOutputStream oos = new ObjectOutputStream(
					new FileOutputStream("lib/J48.model"));
			oos.writeObject(cls);
			oos.flush();
			oos.close();
			// predict the label for test data
			labeleJ48TestSet(testSet, cls);

		}
	}

	/**
	 * \brief Estimating the quality of decision tree classifier based on
	 * multi-class logarithmic loss.
	 * 
	 * 
	 * @param Instances
	 *            trainset
	 * @param Classifier
	 *            clsVal
	 */
	private void evaluateModelLogLoss(Classifier clsVal, Instances trainSet)
			throws Exception {

		// set class attribute
		trainSet.setClassIndex(trainSet.numAttributes() - 1);
		// create copy
		Instances labeled = new Instances(trainSet);
		double sumQuality = 0;
		double epsilon = 0.000000000000001;
		double logPred = 0;
		

		for (int i = 0; i < trainSet.numInstances(); i++) {

			double value = clsVal.classifyInstance(trainSet.instance(i));
			String classAtt = trainSet.classAttribute().value((int) value);
			int numClass = findNumberClass(classAtt);
			double[] preds = clsVal
					.distributionForInstance(labeled.instance(i));

			double origpredic = preds[numClass];

			logPred = Math.log(Math.max(origpredic, epsilon));
			
			sumQuality += logPred;

		}
		System.out.println(Double.toString(-1 * sumQuality
				/ trainSet.numInstances()));

	}

	/**
	 * \brief Convert the class attribute to an integer.
	 * 
	 * @param String
	 *            classAtt
	 * @return int numClass
	 */
	private int findNumberClass(String classAtt) {
		int numClass = 0;
		switch (classAtt) {
		case "Class_1":
			numClass = 0;
			break;
		case "Class_2":
			numClass = 1;
			break;
		case "Class_3":
			numClass = 2;
			break;
		case "Class_4":
			numClass = 3;
			break;
		case "Class_5":
			numClass = 4;
			break;
		case "Class_6":
			numClass = 5;
			break;
		case "Class_7":
			numClass = 6;
			break;
		case "Class_8":
			numClass = 7;
			break;
		case "Class_9":
			numClass = 8;
			break;
		default:
			throw new IllegalArgumentException(
					"Invalid number of class attribute: " + classAtt);

		}
		return numClass;
	}

	/**
	 * \brief Building the SVM model to classify the train set.
	 * 
	 * 
	 * It uses two fold cross validation to classify the train set and builds the
	 * model. In order to estimate the classification quality, two methods have
	 * been used. The multi-class logarithmic loss and the evaluation function
	 * from WEKA library.
	 * 
	 * @param Instances
	 *            trainset
	 * @param Instances
	 *            trainset
	 * @throws Exception
	 */
	private void buildSVMmodel(Instances trainSet, Instances testSet)
			throws Exception, IOException, FileNotFoundException {
		Evaluation validationSvm = new Evaluation(trainSet);

		try {

			// read the model from local file
			LibSVM svmVal = (LibSVM) weka.core.SerializationHelper
					.read("lib/svm.model");
			// Evaluate the classification quality with multi-class logarithmic
			// loss
			System.out.print("multi-class logarithmic loss of SVM model is: ");
			evaluateModelLogLoss(svmVal, trainSet);
			System.out.print("Evaluate the SVM model...");
			validationSvm.evaluateModel(svmVal, trainSet);

			System.out.println(validationSvm.toSummaryString(
					"\nResults of SVM classifier\n======\n", false));

			// test the unlabeled data and assign them to one class

			labeleSVMTestSet(testSet, svmVal);

		} catch (FileNotFoundException exp) {

			System.out.println("Build the SVM model...");
			LibSVM svm = new LibSVM();
			svm.buildClassifier(trainSet);

			// // cross validation using 2 folds to estimate the classification
			// // quality,it picks randomly the instances
			validationSvm.crossValidateModel(svm, trainSet, 2, new Random(1));
			// Evaluate the classification quality with multi-class logarithmic
			// loss
			System.out.print("multi-class logarithmic loss of SVM model is: ");
			evaluateModelLogLoss(svm, trainSet);
			System.out.print("Evaluate the SVM model...");
			validationSvm.evaluateModel(svm, trainSet);
			System.out.println(validationSvm.toSummaryString(
					"\nResults of SVM classifier\n======\n", false));
			ObjectOutputStream oos = new ObjectOutputStream(
					new FileOutputStream("lib/svm.model"));
			oos.writeObject(svm);
			oos.flush();
			oos.close();
			labeleSVMTestSet(testSet, svm);

		}
	}

	/**
	 * \brief Predict label class for test set.
	 * 
	 * It uses the decision tree model, which has been built over train set to
	 * predict the label for each instance in test set and save it in local
	 * directory
	 * 
	 * @param Instances
	 *            trainset
	 * @param Classifier
	 *            clsVal
	 * @throws Exception
	 *             IOException
	 */
	private void labeleJ48TestSet(Instances testSet, Classifier clsVal)
			throws Exception, IOException {

		testSet.setClassIndex(testSet.numAttributes() - 1);

		BufferedWriter writer = new BufferedWriter(new FileWriter(
				"lib/J48labeled.csv"));

		// label instances
		writer.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9");
		writer.newLine();

		for (int i = 0; i < testSet.numInstances(); i++) {

			// get the predicted probabilities
			double[] prediction = clsVal.distributionForInstance(testSet
					.instance(i));

			writer.write(Integer.toString(i + 1));
			// output predictions
			for (int c = 0; c < prediction.length; c++) {
				writer.write(",");
				writer.write(Double.toString(prediction[c]));

			}
			writer.newLine();

		}
		writer.flush();
		writer.close();

	}

	/**
	 * \brief Predict label class for test set.
	 * 
	 * It uses the SVM model, which has been built over train set to predict the
	 * label for each instance in test set and save it in local directory
	 * 
	 * @param Instances
	 *            trainset
	 * @param LibSVM
	 *            svmVal
	 * @throws Exception
	 *             IOException
	 */
	private void labeleSVMTestSet(Instances testSet, LibSVM svmVal)
			throws Exception, IOException {

		testSet.setClassIndex(testSet.numAttributes() - 1);

		BufferedWriter writer = new BufferedWriter(new FileWriter(
				"lib/SVMlabeled.csv"));

		// label instances
		writer.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9");
		writer.newLine();

		for (int i = 0; i < testSet.numInstances(); i++) {

			// get the predicted probabilities
			double[] prediction = svmVal.distributionForInstance(testSet
					.instance(i));
			// output predictions

			writer.write(Integer.toString(i + 1));
			for (int c = 0; c < prediction.length; c++) {
				writer.write(",");
				writer.write(Double.toString(prediction[c]));

			}
			writer.newLine();

		}
		writer.flush();
		writer.close();

	}
}
