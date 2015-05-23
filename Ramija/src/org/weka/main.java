package org.weka;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;

public class main {

	public static void main(String[] args) {
		Classifier[] models = { 
				new J48(), // a decision tree
				new DecisionTable(),//decision table majority classifier
				new DecisionStump() //one-level decision tree
		};
 
		
System.out.println("Hello Weka!");
	}

}
