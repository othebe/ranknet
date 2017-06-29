package ranknet;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class DataSetGenerator {
    private int numFeatures;
    private INDArray idealWeights;

    public DataSetGenerator(int numFeatures, INDArray idealWeights) {
        this.numFeatures = numFeatures;
        this.idealWeights = idealWeights;
    }

    public List<List<Data>> generate(int numDataSets, int numDataPerSet) {
        List<List<Data>> dataSets = new LinkedList<List<Data>>();

        for (int i = 0; i < numDataSets; i++) {
            dataSets.add(generateDataSet(numDataPerSet));
        }

        return dataSets;
    }

    private List<Data> generateDataSet(int numData) {
        INDArray dataRows = null;

        for (int i = 0; i < numData; i++) {
            INDArray features = Nd4j.rand(numFeatures, 1);
            INDArray score = idealWeights.mmul(features);
            INDArray scoredFeatures = Nd4j.vstack(score, features);

            if (dataRows == null) {
                dataRows = scoredFeatures;
            } else {
                dataRows = Nd4j.hstack(dataRows, scoredFeatures);
            }
        }

        int[] columns = new int[numData];
        for (int i = 1; i <= numData; i++) {
            columns[i - 1] = i;
        }
        INDArray sorted =  Nd4j.sortRows(dataRows, 0, false);
        INDArray sortedFeatures = sorted.getRows(columns);

        List<Data> dataSet = new LinkedList<Data>();
        for (int i = 0; i < numData; i++) {
            double computeScore = sorted.getRow(i).getDouble(0);
            INDArray features = sortedFeatures.getRow(i);
            Data data = new Data(numData - i, features, computeScore);
            dataSet.add(data);
        }

        return dataSet;
    }
}
