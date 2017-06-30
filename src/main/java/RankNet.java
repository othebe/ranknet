import neuralnet.activationfunction.IdentityActivationFunction;
import neuralnet.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import ranknet.Data;
import ranknet.DataSetGenerator;
import ranknet.NeuralRankNet;

import java.util.List;

public class RankNet {
    private static final int NUM_FEATURES = 10;
    private static final int NUM_DATA_SETS = 10;
    private static final int NUM_DATA_PER_SET = 55;
    private static final int EPOCHS = 50;

    public static void main(String[] args) {
        DataSetGenerator generator = new DataSetGenerator(NUM_FEATURES, getIdealWeights());

        NeuralRankNet net = NeuralRankNet.Builder()
                .setLearningRate(.001)
                .addLayer(Layer.Builder().setInCount(NUM_FEATURES).setOutCount(NUM_FEATURES * 5).setActivationFunction(IdentityActivationFunction.INSTANCE).build())
                .addLayer(Layer.Builder().setInCount(NUM_FEATURES * 5).setOutCount(1).setActivationFunction(IdentityActivationFunction.INSTANCE).build())
                .build();

        train(net, generator);

        test(net, generator);
    }

    private static void train(NeuralRankNet net, DataSetGenerator generator) {
        List<List<Data>> dataSets = generator.generate(NUM_DATA_SETS, NUM_DATA_PER_SET);

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            for (List<Data> dataSet : dataSets) {
                for (int i = 0; i < dataSet.size() - 1; i++) {
                    net.train(dataSet.get(i).getFeatures(), dataSet.get(i + 1).getFeatures(), Nd4j.scalar(1));
                }
            }
        }
    }

    private static void test(NeuralRankNet net, DataSetGenerator generator) {
        List<List<Data>> dataSets = generator.generate(NUM_DATA_SETS, NUM_DATA_PER_SET);

        for (List<Data> dataSet : dataSets) {
            for (Data data : dataSet) {
                List<INDArray> feedForwardList = net.feedForward(data.getFeatures());
                double score = feedForwardList.get(feedForwardList.size() - 1).getDouble(0);
                System.out.printf("%f ", score);
            }
            System.out.println();
        }
    }

    private static INDArray getIdealWeights() {
        return Nd4j.rand(1, NUM_FEATURES);
    }
}
