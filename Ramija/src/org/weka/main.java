package org.weka;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.attributeSelection.PrincipalComponents;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Add;

public class main {

	public static void main(String[] args) throws Exception{
		
		System.out.println("Hello Weka!");
		Instances training = loadFiles("lib/trainSet.arff");
		Instances test = loadFiles("lib/test.arff");
		
		Classifier[] models = { 
				new J48(), // a decision tree
				new DecisionTable(),//decision table majority classifier
				new DecisionStump() //one-level decision tree
		};
		
	
		PrincipalComponents pca = new PrincipalComponents();
		pca.buildEvaluator(training);
//		String[] bla = new String[4];
//		bla[0] = "-R";
//		bla[1] = "0.95";
//		bla[2] = "-A";
//		bla[3] = "-1";
//		pca.setOptions(bla);
		Instances transformed_train = pca.transformedData(training);
		System.out.println("old: " + training.numAttributes() + " new: " + transformed_train.numAttributes());
		
//		String[] arr = pca.getOptions();
//		for(int i=0; i<arr.length; i++)
//			System.out.println(pca.getOptions()[i]);
		
//		for(int i=0; i< transformed_train.numAttributes(); i++)
//			System.out.println(transformed_train.attribute(i).name());
		
		Add filter = new Add();
		
		Instances transformed_test = pca.transformedData(test);
		System.out.println("old: " + test.numAttributes() + " new: " + transformed_test.numAttributes());
		System.out.println("done");
	}

	private static Instances loadFiles(String address) throws Exception
	{
		DataSource source = new DataSource(address);
		Instances data = source.getDataSet();
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		System.out.println("loaded files: " + address);
		return data;
	}
	
}
