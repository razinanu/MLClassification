package org.weka;

import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class SemiSupervised {

	public SemiSupervised()
	{
		
	}
	
	public void classify(Instances labeled, Instances unlabeled) throws Exception
	{
		System.out.println("start semi-supervised classification.");
		
		int iter = 0;
		while(unlabeled.numInstances() > 0)
		{
			IBk classifier = new IBk(5);
			
			System.out.println("iteration " + iter++ + "(count: " + labeled.numInstances() + " " + unlabeled.numInstances() + ")");
			
			classifier.buildClassifier(labeled);

			int random_instance = (int)(Math.random() * (unlabeled.numInstances()-1));
			double cl = classifier.classifyInstance(unlabeled.instance(random_instance));
			unlabeled.instance(random_instance).setClassValue(cl);
			labeled.add((Instance)unlabeled.instance(random_instance).copy());
			unlabeled.delete(random_instance);		
		}
		
		for(int i=0; i<labeled.numInstances(); i++)
		{
			System.out.println(labeled.instance(i).classValue());
		}
	}
	
	
}
