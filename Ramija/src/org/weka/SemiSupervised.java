package org.weka;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

/**
 * \brief This class tries to classify the test set with a semi supervised method.
 *
 */
public class SemiSupervised {

	private boolean[] used;	///represents the unlabeled (false) and labeled(true) instances of the test set
	
	///the default constructor
	public SemiSupervised()
	{
	}
	
	/**
	 * \brief Classifies the unlabeled set via semi-supervised self-training.
	 * 
	 * Random instances of the unlabeled set are classified with the IBk-Algorithm. This is
	 * a k-nearest neighbour approach. After labeling on instance, it is added to the training
	 * set and therefore used in the next iteration of k-nearest neighbour classification.
	 * 
	 * @param training
	 * @param test
	 * @throws Exception
	 */
	public void classify(Instances training, Instances test) throws Exception
	{

		System.out.println("Start semi-supervised classification. This will probably take a long time (> 2h). ");

		
		used = new boolean[test.numInstances()];
		int iter = 0;
		
		Instances labeled = new Instances(training);
		Instances unlabeled = new Instances(test);	//copy the input, so it won't be changed
		
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
		
		System.out.println("Semi-supervised classification is done. Results are found in lib/Semilabeled.csv");
	}
	
	/**
	 * \brief searches for a random unlabeled instances.
	 * 
	 * Since the original order of the test set shall not be destroyed, a newly classified
	 * instance cannot be removed from the test set. To avoid classifying one instance multiple
	 * times, a boolean array stores which instances have already been classified. If the random 
	 * function wants to return an instance which has been already used, the next free instance 
	 * will be found.
	 * 
	 * @return
	 */
	private int findRandomInstance()
	{
		int random_instance = (int)(Math.random() * (used.length));
		
		while (used[random_instance])
		{
			random_instance++;
			if(random_instance >= used.length)	random_instance = 0;
		}
		
		used[random_instance] = true;
		
		return random_instance;
	}
	
	/**
	 * \brief writes the labeled instances of the test set into a .csv
	 * 
	 * @param testSet
	 * @throws IOException
	 */
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
