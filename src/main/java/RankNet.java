import neuralnet.NeuralNet;
import neuralnet.activationfunction.SigmoidActivationFunction;
import neuralnet.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import ranknet.Data;
import ranknet.DataSetGenerator;

import java.util.Iterator;
import java.util.List;

public class RankNet {
    private static final int NUM_FEATURES = 5;
    private static final int EPOCHS = 10;

    public static void main(String[] args) {
        DataSetGenerator generator = new DataSetGenerator(NUM_FEATURES, getIdealWeights());

        NeuralNet net = NeuralNet.Builder()
                .setLearningRate(.01)
                .addLayer(Layer.Builder().setInCount(NUM_FEATURES).setOutCount(NUM_FEATURES * 2).build())
                .addLayer(Layer.Builder().setInCount(NUM_FEATURES * 2).setOutCount(NUM_FEATURES * 2).setActivationFunction(SigmoidActivationFunction.INSTANCE).build())
                .addLayer(Layer.Builder().setInCount(NUM_FEATURES * 2).setOutCount(NUM_FEATURES * 2).setActivationFunction(SigmoidActivationFunction.INSTANCE).build())
                .addLayer(Layer.Builder().setInCount(NUM_FEATURES * 2).setOutCount(1).build())
                .build();

        train(net, generator);
    }

    private static void train(NeuralNet net, DataSetGenerator generator) {
        List<List<Data>> dataSets = generator.generate(5, 5);

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            for (List<Data> dataSet : dataSets) {
                Iterator<Data> dataSetIterator = dataSet.iterator();
                Data dataI = dataSetIterator.next();

                while (dataSetIterator.hasNext()) {
                    Data dataJ = dataSetIterator.next();

                    INDArray outI = net.feedForward(dataI.getFeatures()).get(0);
                    INDArray outJ = net.feedForward(dataJ.getFeatures()).get(0);

                    INDArray outIJ = outI.sub(outJ);
                    INDArray pIJ = SigmoidActivationFunction.INSTANCE.output(outIJ);

                    INDArray error = SigmoidActivationFunction.INSTANCE.derivative(outIJ)
                            .mul(Nd4j.ones(pIJ.shape()).sub(pIJ));

                    net.backpropagateAndUpdate(dataI.getFeatures(), error);
                }
            }
        }
    }

    private static INDArray getIdealWeights() {
        return Nd4j.rand(1, NUM_FEATURES);
    }
}
