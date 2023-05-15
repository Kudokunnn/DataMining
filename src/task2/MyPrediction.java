package task2;

import weka.core.Instances;
import weka.classifiers.trees.J48;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class MyPrediction {

    public static void main(String[] args) throws Exception {

        // Load data set
        DataSource source = new DataSource("stroke_prediction.arff");
        Instances data = source.getDataSet();
        
        // Convert class variable to nominal
        NumericToNominal filter = new NumericToNominal();
        filter.setAttributeIndices("last");
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);

        // Set class index to the last attribute
        data.setClassIndex(data.numAttributes() - 1);

        // Split data into training and testing sets
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;
        Instances trainData = new Instances(data, 0, trainSize);
        Instances testData = new Instances(data, trainSize, testSize);

        // Build decision tree classifier
        J48 classifier = new J48();
        classifier.buildClassifier(trainData);

        // Evaluate classifier on testing set
        double correct = 0;
        for (int i = 0; i < testData.numInstances(); i++) {
            double pred = classifier.classifyInstance(testData.instance(i));
            if (pred == testData.instance(i).classValue()) {
                correct++;
            }
        }
        double accuracy = 100 * correct / testData.numInstances();
        System.out.println("Accuracy of Decision Tree: " + accuracy + "%");
    }
}
