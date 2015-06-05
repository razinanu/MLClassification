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

public class main {

	public static void main(String[] args) throws Exception{
		
		System.out.println("Hello Weka!");
		Instances training = loadFiles("lib/trainSet.arff");
		Instances test = loadFiles("lib/test.csv");
		
		Classifier[] models = { 
				new J48(), // a decision tree
				new DecisionTable(),//decision table majority classifier
				new DecisionStump() //one-level decision tree
		};
		
	
		PrincipalComponents pc = new PrincipalComponents();
		pc.buildEvaluator(training);
		Instances transformed_train = pc.transformedData(training);
	}

	private static Instances loadFiles(String address) throws Exception
	{
		DataSource source = new DataSource(address);
		Instances data = source.getDataSet();
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		System.out.println("loaded files. " + address);
		return data;
	}
	
}
