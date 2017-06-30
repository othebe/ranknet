import neuralnet.activationfunction.IdentityActivationFunction;
import neuralnet.activationfunction.SigmoidActivationFunction;
import neuralnet.layer.Layer;
import org.nd4j.linalg.factory.Nd4j;
import ranknet.Data;
import ranknet.Evaluator;
import ranknet.LetorReader;
import ranknet.NeuralRankNet;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class RankNetLetor {
    private static final String DATASET_TYPE_TRAIN = "letor/train.txt";
    private static final String DATASET_TYPE_TEST = "letor/train.txt";
    private static final int DATASET_LETOR_FEATURE_COUNT = 136;

    private static final String DATASET_TYPE_TRAIN_SMALL = "letor_small/train.txt";
    private static final String DATASET_TYPE_TEST_SMALL = "letor_small/train.txt";
    private static final int DATASET_LETOR_FEATURE_COUNT_SMALL = 46;

    private static final double LEARNING_RATE = .01f;
    private static final int EPOCHS = 10;

    private static final ClassLoader CLASSLOADER = ClassLoader.getSystemClassLoader();

    public static void main(String[] args) throws FileNotFoundException {
        int inputCount = DATASET_LETOR_FEATURE_COUNT_SMALL;
        int hiddenCount = inputCount * 3;

        NeuralRankNet net = NeuralRankNet.Builder()
                .setLearningRate(LEARNING_RATE)
                .addLayer(Layer.Builder().setInCount(inputCount).setOutCount(hiddenCount).setActivationFunction(SigmoidActivationFunction.INSTANCE).build())
                .addLayer(Layer.Builder().setInCount(hiddenCount).setOutCount(hiddenCount).setActivationFunction(SigmoidActivationFunction.INSTANCE).build())
                .addLayer(Layer.Builder().setInCount(hiddenCount).setOutCount(1).setActivationFunction(SigmoidActivationFunction.INSTANCE).build())
                .build();

        train(net);
    }

    public static void train(NeuralRankNet net) throws FileNotFoundException {
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            File file = new File(CLASSLOADER.getResource(DATASET_TYPE_TRAIN_SMALL).getFile());
            int featureCount = DATASET_LETOR_FEATURE_COUNT_SMALL;
            LetorReader reader = new LetorReader(file, featureCount);

            System.out.printf("*** Epoch %d ***\n", epoch);

            int count = 1;
            while (reader.hasNext()) {
                test(net);
                System.out.printf("Training dataset %d\n", count);
                count++;

                List<Data> dataSet = reader.next();
                Iterator<Data> iteratorI = dataSet.iterator();
                while (iteratorI.hasNext()) {
                    Data dataI = iteratorI.next();
                    double rankI = dataI.getRankScore();

                    Iterator<Data> iteratorJ = dataSet.iterator();
                    while (iteratorJ.hasNext()) {
                        Data dataJ = iteratorJ.next();
                        double rankJ = dataJ.getRankScore();

                        if (rankI > rankJ) {
                            net.train(dataI.getFeatures(), dataJ.getFeatures(), Nd4j.scalar(1));
                        }
                    }
                }
            }

            System.out.printf("Trained Epoch %d\n\n", epoch);
        }
    }

    public static void test(NeuralRankNet net) throws FileNotFoundException {
        File file = new File(CLASSLOADER.getResource(DATASET_TYPE_TEST_SMALL).getFile());
        int featureCount = DATASET_LETOR_FEATURE_COUNT_SMALL;
        LetorReader reader = new LetorReader(file, featureCount);

        Evaluator evaluator = new Evaluator(net);

        List<List<Data>> dataSets = new LinkedList<List<Data>>();
        while (reader.hasNext()) {
            dataSets.add(reader.next());
        }

        evaluator.evaluate(dataSets);
    }
}
