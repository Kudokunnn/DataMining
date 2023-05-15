package task2;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Classifier;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.Evaluation;

public class MyClassifier {

    public static void main(String[] args) throws Exception {

        // Load data set
        DataSource source = new DataSource("stroke_prediction.arff");
        Instances data = source.getDataSet();

        // Set class attribute
        data.setClassIndex(data.numAttributes() - 1);

        // Randomize the data
        Randomize randomizeFilter = new Randomize();
        randomizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, randomizeFilter);

        // Convert class attribute to nominal
        NumericToNominal nominalFilter = new NumericToNominal();
        nominalFilter.setAttributeIndices("" + (data.classIndex() + 1));
        nominalFilter.setInputFormat(data);
        data = Filter.useFilter(data, nominalFilter);

        // Split the data into training and testing sets
        int trainSize = (int) Math.round(data.numInstances() * 0.7);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);

        // Build classifier
        Classifier cls = new J48();
        cls.buildClassifier(train);

        // Evaluate classifier
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(cls, test);	

        // Print results
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println(eval.toClassDetailsString());
    }
}