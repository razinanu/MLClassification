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

		buildDecisionTreemodel(trainSet);
	}

	private void buildDecisionTreemodel(Instances trainSet) throws Exception,
			IOException, FileNotFoundException {
		Evaluation validationJ48 = new Evaluation(trainSet);
		try {

			System.out.println("Validation of exiting J48 model... ");
			// read the model from local file
			Classifier clsVal = (Classifier) weka.core.SerializationHelper
					.read("lib/J48.model");
			// Evaluate the decision tree model
			validationJ48.evaluateModel(clsVal, trainSet);

			System.out.println(validationJ48.toSummaryString(
					"\nResults\n======\n", false));

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
					"\nResults\n======\n", false));
			ObjectOutputStream oos = new ObjectOutputStream(
					new FileOutputStream("lib/J48.model"));
			oos.writeObject(cls);
			oos.flush();
			oos.close();

		}
	}

	private void buildSVMmodel(Instances trainSet, Instances testSet)
			throws Exception, IOException, FileNotFoundException {
		Evaluation validationSvm = new Evaluation(trainSet);

		try {
			LibSVM svmVal = (LibSVM) weka.core.SerializationHelper
					.read("lib/svm.model");

			// System.out.println("Validation of exiting SVM model... ");
			// validationSvm.evaluateModel(svmVal, trainSet);
			//
			// System.out.println(validationSvm.toSummaryString(
			// "\nResults\n======\n", false));
			// test the unlabeled data and assign them to one class
			labeleSVMTestSet(testSet, svmVal);

		} catch (FileNotFoundException exp) {

			System.out.println("Build the SVM model...");
			LibSVM svm = new LibSVM();
			svm.buildClassifier(trainSet);

			// // cross validation using 2 folds to estimate the classification
			// // quality,it picks randomly the instances
			validationSvm.crossValidateModel(svm, trainSet, 2, new Random(1));
			validationSvm.evaluateModel(svm, trainSet);
			System.out.println(validationSvm.toSummaryString(
					"\nResults\n======\n", false));
			ObjectOutputStream oos = new ObjectOutputStream(
					new FileOutputStream("lib/svm.model"));
			oos.writeObject(svm);
			oos.flush();
			oos.close();
			labeleSVMTestSet(testSet, svm);

		}
	}

	private void labeleSVMTestSet(Instances testSet, LibSVM svmVal)
			throws Exception, IOException {

		testSet.setClassIndex(testSet.numAttributes() - 1);

		// create copy
		Instances labeled = new Instances(testSet);

		// label instances
		for (int i = 0; i < 5; i++) {
			// System.out.println("in foreach!");
			double clsLabel = svmVal.classifyInstance(testSet.instance(i));
			labeled.instance(i).setClassValue(clsLabel);
		}

		// save labeled data
		try {
			System.out.println("open the buffer");
			BufferedWriter writer = new BufferedWriter(new FileWriter(
					"lib/labeled.arff"));
			for (int i = 0; i < labeled.numInstances(); i++) {

				writer.write("Instance is" + i);
				writer.newLine();
				writer.write(labeled.instance(i).toString());
				
				writer.newLine();

			}

			// writer.newLine();
			writer.flush();
			writer.close();
		} catch (Exception exc) {
			System.out.println("error");
		}
	}
}
