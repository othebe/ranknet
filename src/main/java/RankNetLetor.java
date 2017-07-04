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
    private static final int EPOCHS = 30;

    private static final ClassLoader CLASSLOADER = ClassLoader.getSystemClassLoader();

    private static final String dataSetTrain = DATASET_TYPE_TRAIN_SMALL;
    private static final String dataSetTest = DATASET_TYPE_TEST_SMALL;
    private static final int dataSetFeatureCount = DATASET_LETOR_FEATURE_COUNT_SMALL;
    private static final int dataSetHiddenCount = dataSetFeatureCount * 2;

    public static void main(String[] args) throws FileNotFoundException {
        NeuralRankNet net = NeuralRankNet.Builder()
                .setLearningRate(LEARNING_RATE)
                .addLayer(Layer.Builder().setInCount(dataSetFeatureCount).setOutCount(dataSetHiddenCount).setActivationFunction(SigmoidActivationFunction.INSTANCE).build())
                .addLayer(Layer.Builder().setInCount(dataSetHiddenCount).setOutCount(dataSetHiddenCount).setActivationFunction(SigmoidActivationFunction.INSTANCE).build())
                .addLayer(Layer.Builder().setInCount(dataSetHiddenCount).setOutCount(1).setActivationFunction(SigmoidActivationFunction.INSTANCE).build())
                .build();

        train(net);
    }

    public static void train(NeuralRankNet net) throws FileNotFoundException {
        System.out.printf("*** PRETRAIN ***\n");
        test(net);
        System.out.println();

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            File file = new File(CLASSLOADER.getResource(dataSetTrain).getFile());
            int featureCount = dataSetFeatureCount;
            LetorReader reader = new LetorReader(file, featureCount);

            System.out.printf("*** Epoch %d ***\n", epoch);

            while (reader.hasNext()) {
                List<Data> dataSet = reader.next();

                // COMPREHENSIVE: Build all possible permutations from dataset.
                for (int i = 0; i < dataSet.size(); i++) {
                    Data dataI = dataSet.get(i);

                    for (int j = i + 1; j < dataSet.size(); j++) {
                        Data dataJ = dataSet.get(j);

                        if (dataI.getRankScore() > dataJ.getRankScore()) {
                            net.train(dataI.getFeatures(), dataJ.getFeatures(), Nd4j.scalar(1));
                        } else if (dataI.getRankScore() < dataJ.getRankScore()) {
                            net.train(dataJ.getFeatures(), dataI.getFeatures(), Nd4j.scalar(1));
                        }
                    }
                }

                // QUICK: Linearly build pairs from dataset.
                /*Iterator<Data> iteratorI = dataSet.iterator();
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
                }*/
            }

            test(net);
            System.out.println();
        }
    }

    public static void test(NeuralRankNet net) throws FileNotFoundException {
        File file = new File(CLASSLOADER.getResource(dataSetTest).getFile());
        int featureCount = dataSetFeatureCount;
        LetorReader reader = new LetorReader(file, featureCount);

        Evaluator evaluator = new Evaluator(net);

        List<List<Data>> dataSets = new LinkedList<List<Data>>();
        while (reader.hasNext()) {
            dataSets.add(reader.next());
        }

        evaluator.evaluate(dataSets);
    }
}
