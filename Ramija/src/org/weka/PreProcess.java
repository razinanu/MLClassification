package org.weka;



import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.PrincipalComponents;

/**
 * \brief This class is for analyzing the train and test sets for selecting an appropriate subset of the attribute 
 * @param train set
 * @param test set
 */
public class PreProcess {
	public Instances newTrain;
	public Instances newTest;
	/**
	 * \brief Select an appropriate subset of the attributes based on Principal Component Analysis.
	 * 
	 * convert a set of possibly correlated attributes into a set of values of linearly uncorrelated attributes
	 * This process will be done for both train set and test set
	 * 
	 * 
	 * @param train set
	 * @param test set
	 */

	public void AttributeSelect(Instances trainSet, Instances testSet) {

		Instances newTestSet = null;
		newTestSet = addClassinTestSet(testSet);
		PrincipalComponents pca = new PrincipalComponents();
		try {
			pca.setInputFormat(trainSet);
			newTrain = Filter.useFilter(trainSet, pca);
			newTest = Filter.useFilter(newTestSet, pca);
			System.out.println("Number of att in Train: "
					+ newTrain.numAttributes());
			System.out.println("Number of att in Test: "
					+ newTest.numAttributes());

		} catch (Exception exc) {
			System.out.println(exc.getMessage());
		}
	}
	/**
	 * \brief Add the class attribute to the test set. 
	 * 
	 * To have same attribute analyze with PCA both train and test sets must contain same number of attributes. 
	 * As the test set doesn't contains the class attribute, it adds just a new column "class" with empty value to test set.
	
	 * @param test set
	 * @throws Exception
	 */

	public Instances addClassinTestSet(Instances testSet) {

		Add filter = new Add();
		Instances newTestSet = null;

		filter.setAttributeIndex("last");
		filter.setNominalLabels("Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9");
		filter.setAttributeName("class");
		try {

			filter.setInputFormat(testSet);
			newTestSet = Filter.useFilter(testSet, filter);

//			for (int i = 0; i < newTestSet.numInstances(); i++) {
//				newTestSet.instance(i).setValue(newTestSet.numAttributes() - 1,
//						-1);
//			}

		} catch (Exception exc) {
			System.out.println(exc.getMessage());
		}
		return newTestSet;

	}

}
