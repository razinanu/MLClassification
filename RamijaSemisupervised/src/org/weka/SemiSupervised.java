package org.weka;

import weka.core.Instances;
import weka.classifiers.collective.meta.YATSI;
import weka.classifiers.collective.meta.CollectiveEM;

public class SemiSupervised {

	public SemiSupervised()
	{
		
	}
	
	public void other_classifier(Instances labeled, Instances unlabeled) throws Exception
	{
		CollectiveEM em = new CollectiveEM();
		
		em.buildClassifier(labeled, unlabeled);
		
		System.out.println("done");
		
		for(int i=0; i<unlabeled.numInstances(); i++)
			System.out.println(unlabeled.instance(i).classValue());
	}
	
	public void classify(Instances labeled, Instances unlabeled) throws Exception
	{
		other_classifier(labeled, unlabeled);
		
//		System.out.println();
//	    // configure classifier
//	    YATSI yatsi = new YATSI();
//	    yatsi.setKNN(3);	//k-nearest neighbors
//	    yatsi.setNoWeights(true);
//
////	    // build classifier
//	    yatsi.buildClassifier(labeled, unlabeled);
////		
//		for(int i=0; i<unlabeled.numInstances(); i++)
//			System.out.println(unlabeled.instance(i).classValue());
	}
	
}
