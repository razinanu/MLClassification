package org.weka;

import weka.core.Instances;
import weka.classifiers.collective.meta.YATSI;

public class SemiSupervised {

	public SemiSupervised()
	{
		
	}
	
	public void classify(Instances labeled, Instances unlabeled) throws Exception
	{
	    // configure classifier
	    YATSI yatsi = new YATSI();
	    yatsi.setKNN(10);	//k-nearest neighbors
	    yatsi.setNoWeights(true);

	    
	    
//	    // build classifier
	    yatsi.buildClassifier(labeled, unlabeled);
//		
		for(int i=0; i<unlabeled.numInstances(); i++)
			System.out.println(unlabeled.instance(i).classValue());
		
	}
	
}
