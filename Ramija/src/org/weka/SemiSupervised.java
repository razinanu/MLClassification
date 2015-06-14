package org.weka;

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
		System.out.println("start semi-supervised classification.");
		
		used = new boolean[unlabeled.numInstances()];
		int iter = 0;
		
		while(unlabeled.numInstances() > 0)
		{		
			System.out.println("iteration " + iter++ + "(count: " + labeled.numInstances() + " " + unlabeled.numInstances() + ")");

			IBk classifier = new IBk(5);
			classifier.buildClassifier(labeled);

			int random_instance = findRandomInstance();
			
			double cl = classifier.classifyInstance(unlabeled.instance(random_instance));
			unlabeled.instance(random_instance).setClassValue(cl);
			labeled.add((Instance)unlabeled.instance(random_instance).copy());
		}
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
	
	
}
