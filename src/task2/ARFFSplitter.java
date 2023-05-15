package task2;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

import java.io.File;

public class ARFFSplitter {

    public static void main(String[] args) throws Exception {

        // Load data set
        Instances data = DataSource.read("stroke_prediction.arff");
        data.setClassIndex(data.numAttributes() - 1);

        // Randomize the data
        Randomize randomizeFilter = new Randomize();
        randomizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, randomizeFilter);

        // Split the data into training and testing sets
        int trainSize = (int) Math.round(data.numInstances() * 0.7);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);

        // Save the training set to file
        ArffSaver trainSaver = new ArffSaver();
        trainSaver.setInstances(train);
        trainSaver.setFile(new File("train_file.arff"));
        trainSaver.writeBatch();

        // Save the testing set to file
        ArffSaver testSaver = new ArffSaver();
        testSaver.setInstances(test);
        testSaver.setFile(new File("test_file.arff"));
        testSaver.writeBatch();
    }
}

