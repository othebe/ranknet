package ranknet;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Iterator;
import java.util.List;

public class Evaluator {
    private NeuralRankNet net;

    public Evaluator(NeuralRankNet net) {
        this.net = net;
    }

    public void evaluate(List<List<Data>> dataSets) {
        int mismatchedCount = 0;
        int dataCount = 0;

        for (List<Data> dataSet : dataSets) {
            Iterator<Data> dataIterator = dataSet.iterator();
            double prevScore = getScore(dataIterator.next());

            while (dataIterator.hasNext()) {
                double currScore = getScore(dataIterator.next());
                if (currScore > prevScore) {
                    mismatchedCount++;
                }
                dataCount++;
            }
        }

        System.out.printf("Mismatch/Total = %d/%d [%f]\n", mismatchedCount, dataCount, mismatchedCount * 1.0f / dataCount);
    }

    private double getScore(Data data) {
        List<INDArray> feedForward = net.feedForward(data.getFeatures());
        return feedForward.get(feedForward.size() - 1).getDouble(0);
    }
}
