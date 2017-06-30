import neuralnet.activationfunction.IdentityActivationFunction;
import neuralnet.activationfunction.SigmoidActivationFunction;
import neuralnet.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import ranknet.Data;
import ranknet.DataSetGenerator;
import ranknet.Evaluator;
import ranknet.NeuralRankNet;

import java.util.List;

public class RankNet {
    private static final int NUM_FEATURES = 64;
    private static final int HIDDEN_NODES = NUM_FEATURES * 2;
    private static final int NUM_DATA_SETS = 100;
    private static final int NUM_DATA_PER_SET = 5;
    private static final int EPOCHS = 20;

    public static void main(String[] args) {
        DataSetGenerator generator = new DataSetGenerator(NUM_FEATURES, getIdealWeights());

        NeuralRankNet net = NeuralRankNet.Builder()
                .setLearningRate(.1)
                .addLayer(Layer.Builder().setInCount(NUM_FEATURES).setOutCount(HIDDEN_NODES).setActivationFunction(SigmoidActivationFunction.INSTANCE).build())
                .addLayer(Layer.Builder().setInCount(HIDDEN_NODES).setOutCount(HIDDEN_NODES).setActivationFunction(SigmoidActivationFunction.INSTANCE).build())
                .addLayer(Layer.Builder().setInCount(HIDDEN_NODES).setOutCount(1).setActivationFunction(IdentityActivationFunction.INSTANCE).build())
                .build();

        train(net, generator);

//        test(net, generator);
    }

    private static void train(NeuralRankNet net, DataSetGenerator generator) {
        List<List<Data>> dataSets = generator.generate(NUM_DATA_SETS, NUM_DATA_PER_SET);
        List<List<Data>> testDataSets = generator.generate(NUM_DATA_SETS, NUM_DATA_PER_SET);

        Evaluator evaluator = new Evaluator(net);

        System.out.printf("*** PreTrain ***\n");
        evaluator.evaluate(testDataSets);

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            for (List<Data> dataSet : dataSets) {
                for (int i = 0; i < dataSet.size() - 1; i++) {
                    net.train(dataSet.get(i).getFeatures(), dataSet.get(i + 1).getFeatures(), Nd4j.scalar(1));
                }
            }
            System.out.printf("Epoch %d: ", epoch);

            evaluator.evaluate(testDataSets);
        }
    }

    private static void test(NeuralRankNet net, DataSetGenerator generator) {
        List<List<Data>> dataSets = generator.generate(NUM_DATA_SETS, NUM_DATA_PER_SET);

        Evaluator evaluator = new Evaluator(net);
        evaluator.evaluate(dataSets);
    }

    private static INDArray getIdealWeights() {
        return Nd4j.rand(1, NUM_FEATURES);
    }
}
