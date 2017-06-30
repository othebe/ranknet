package ranknet;

import com.sun.tools.javac.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

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
        List<Data> dataSet = new LinkedList<Data>();
        List<Pair<INDArray, INDArray>> scoredFeaturePairList = new ArrayList<Pair<INDArray, INDArray>>();

        for (int i = 0; i < numData; i++) {
            INDArray features = Nd4j.rand(1, numFeatures);
            INDArray score = features.mmul(idealWeights.transpose());
            scoredFeaturePairList.add(new Pair<INDArray, INDArray>(score, features));
        }

        Collections.sort(scoredFeaturePairList, new Comparator<Pair<INDArray, INDArray>>() {
            public int compare(Pair<INDArray, INDArray> o1, Pair<INDArray, INDArray> o2) {
                return Double.compare(o2.fst.getDouble(0), o1.fst.getDouble(0));
            }
        });

        for (int i = 0; i < scoredFeaturePairList.size(); i++) {
            Pair<INDArray, INDArray> scoredFeaturePair = scoredFeaturePairList.get(i);
            dataSet.add(new Data(numData - i, scoredFeaturePair.snd, scoredFeaturePair.fst.getDouble(0)));
        }

        return dataSet;
    }
}
