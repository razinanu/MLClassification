package org.weka;

import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.PrincipalComponents;

public class PreProcess {
	public Instances newTrain;
	public Instances newTest;
	

	public void AttributeSelect(Instances trainSet, Instances testSet) {

		Instances newTestSet=null;
		newTestSet=addClassinTestSet(testSet);
		PrincipalComponents pca = new PrincipalComponents();
		try {
			pca.setInputFormat(trainSet);
			newTrain = Filter.useFilter(trainSet, pca);
			newTest = Filter.useFilter(newTestSet, pca);
			System.out.println("Number of att in Train: "+newTrain.numAttributes());
			System.out.println("Number of att in Test: "+newTest.numAttributes());
//			for (int i = 0; i < newTrain.numAttributes(); i++) {
//				System.out.println("Train: "+newTrain.attribute(i));
//				
//			}
//			for (int i = 0; i < newTest.numAttributes(); i++) {
//				System.out.println("Test: "+newTest.attribute(i));
//				
//			}

		} catch (Exception exc) {
			System.out.println(exc.getMessage());
		}
	}

	public Instances addClassinTestSet(Instances testSet) {

		Add filter = new Add();
		Instances newTestSet=null;

		filter.setAttributeIndex("last");
		filter.setNominalLabels("Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9");
		filter.setAttributeName("class");
		try {
			filter.setInputFormat(testSet);
			newTestSet = Filter.useFilter(testSet, filter);
//			System.out.println("Test set num Attribute: "
//					+ newTestSet.numAttributes());
		} catch (Exception exc) {
			System.out.println(exc.getMessage());
		}
		return newTestSet;

	}

}
