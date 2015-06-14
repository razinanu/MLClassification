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

public class Classifiers {

	public void classifier(Instances trainSet, Instances testSet)
			throws Exception {

		buildSVMmodel(trainSet, testSet);

		buildDecisionTreemodel(trainSet, testSet);
	}

	private void buildDecisionTreemodel(Instances trainSet, Instances testSet)
			throws Exception, IOException, FileNotFoundException {
		Evaluation validationJ48 = new Evaluation(trainSet);
		try {

			// read the model from local file
			Classifier clsVal = (Classifier) weka.core.SerializationHelper
					.read("lib/J48.model");

			System.out.println("Validation of exiting J48 model... ");
			// Evaluate the decision tree model
			validationJ48.evaluateModel(clsVal, trainSet);

			System.out.println(validationJ48.toSummaryString(
					"\nResults of J48 classifier\n======\n", false));
			// predict the label for test data
			labeleJ48TestSet(testSet, clsVal);

		} catch (FileNotFoundException exp) {

			System.out.println("Build the J48 model...");
			// Decision tree classifier
			Classifier cls = new J48();
			cls.buildClassifier(trainSet);
			// cross validation using 2 folds to estimate the classification
			// quality,it picks randomly the instances
			validationJ48.crossValidateModel(cls, trainSet, 2, new Random(1));
			validationJ48.evaluateModel(cls, trainSet);
			System.out.println(validationJ48.toSummaryString(
					"\nResults of J48 classifier\n======\n", false));
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

	private void buildSVMmodel(Instances trainSet, Instances testSet)
			throws Exception, IOException, FileNotFoundException {
		Evaluation validationSvm = new Evaluation(trainSet);

		try {

			// read the model from local file
			LibSVM svmVal = (LibSVM) weka.core.SerializationHelper
					.read("lib/svm.model");

			// System.out.println("Validation of exiting SVM model... ");
			// validationSvm.evaluateModel(svmVal, trainSet);
			//
			// System.out.println(validationSvm.toSummaryString(
			// "\nResults of SVM classifier\n======\n", false));

			// test the unlabeled data and assign them to one class
			// labeleSVMTestSet(testSet, svmVal);

		} catch (FileNotFoundException exp) {

			System.out.println("Build the SVM model...");
			LibSVM svm = new LibSVM();
			svm.buildClassifier(trainSet);

			// // cross validation using 2 folds to estimate the classification
			// // quality,it picks randomly the instances
			validationSvm.crossValidateModel(svm, trainSet, 2, new Random(1));
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
