package org.weka;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;


import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class SemiSupervised {

	private boolean[] used;
	
	public SemiSupervised()
	{
	}
	
	public void classify(Instances labeled, Instances unlabeled) throws Exception
	{
		System.out.println("start semi-supervised classification...");
		
		used = new boolean[unlabeled.numInstances()];
		int iter = 0;
		
		while(iter < unlabeled.numInstances())
		{		
			//System.out.println("iteration " + iter++ + "(count: " + labeled.numInstances() + ")");

			IBk classifier = new IBk(5);
			classifier.buildClassifier(labeled);

			int random_instance = findRandomInstance();
			
			double cl = classifier.classifyInstance(unlabeled.instance(random_instance));
			unlabeled.instance(random_instance).setClassValue(cl);
			labeled.add((Instance)unlabeled.instance(random_instance).copy());
		}
		
		labelSemiSup(unlabeled);
	}
		
	private int findRandomInstance()
	{
		int random_instance = (int)(Math.random() * (used.length));
		
		while (used[random_instance])
		{
			random_instance++;
			if(random_instance > used.length)	random_instance = 0;
		}
		
		used[random_instance] = true;
		
		return random_instance;
	}
	
	private void labelSemiSup(Instances testSet) throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(
				"lib/Semilabeled.csv"));

		// label instances
		writer.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9");
		writer.newLine();

		for (int i = 0; i < testSet.numInstances(); i++) {

			// get the predicted probabilities
			double[] prediction = new double[9];
			prediction[(int)testSet.instance(i).classValue()] = 1.0;

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
	
}
