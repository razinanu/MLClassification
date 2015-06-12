package org.weka;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Add;


public class main {
	
	

	public static void main(String[] args) throws Exception {

		
		 Instances trainSet = loadFiles("lib/trainSet.arff");
	     Instances testSet = loadFiles("lib/test.arff");

		PreProcess attSelect = new PreProcess();

		attSelect.AttributeSelect(trainSet, testSet);

		SemiSupervised s = new SemiSupervised();
		s.classify(trainSet, testSet);
	}

	private static Instances loadFiles(String address) throws Exception {
		DataSource source = new DataSource(address);
		Instances data = source.getDataSet();
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		System.out.println("loaded files: " + address);
		return data;
	}

}
