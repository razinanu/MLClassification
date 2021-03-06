package org.weka;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class main {

	/**
	 * \brief classify a data set with three different methods and write the output in .csv
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		Instances trainSet = loadFiles("lib/trainSet.arff");
		Instances testSet = loadFiles("lib/test.arff");

		PreProcess attSelect = new PreProcess();

		attSelect.AttributeSelect(trainSet, testSet);
		Classifiers c = new Classifiers();
		c.classifier(attSelect.newTrain, attSelect.newTest);
		
		SemiSupervised s = new SemiSupervised();
		s.classify(attSelect.newTrain, attSelect.newTest);
	}

	/**
	 * \brief load the training and test set.
	 * 
	 * @param address
	 * @return
	 * @throws Exception
	 */
	private static Instances loadFiles(String address) throws Exception {
		DataSource source = new DataSource(address);
		Instances data = source.getDataSet();
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);

		System.out.println("loaded files: " + address);
		return data;
	}

}
